"""Shared DSPy-driven orchestration logic for lightweight clinical agents."""

from __future__ import annotations

import json
import os
import logging
import re
import textwrap
from abc import ABC, abstractmethod
from datetime import datetime
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Type

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from .signatures import AssessAndSelectSkills, BuildToolPlan, DraftFinalAnswer, ToolExecutionStep
from .toolkit import BaseTool, ToolOutput
from .skills.registry import build_skill_prompt, render_skill_context, get_skill_catalog
from .skills.policy_engine import (
    apply as apply_policy,
    extract_target_concepts,
    normalise_nodes_to_evidence,
)
from .agent_tools import fetch_patient_lab_keys

logger = logging.getLogger(__name__)


class DSPyAgentBase(ABC):
    """Base class that coordinates plan → tool execution → summary via DSPy signatures."""

    _AUTO_TOP_K_CAP = 30

    def __init__(self, *, tools: Optional[Iterable[BaseTool]] = None, max_tool_rounds: int = 6) -> None:
        self.tools: List[BaseTool] = list(tools or [])
        self._max_tool_rounds = max_tool_rounds
        # Default to full context; can be turned off with AGENT_FULL_CONTEXT_FOR_STEPS=false
        self._use_full_context_for_steps = self._env_flag("AGENT_FULL_CONTEXT_FOR_STEPS", True)
        logger.info(
            "Tool execution context mode: %s",
            "full_context" if self._use_full_context_for_steps else "summaries",
        )
        self._event_handler: Optional[Callable[[Dict[str, Any]], None]] = None
        self._auto_top_k_cap = self._AUTO_TOP_K_CAP
        self._skill_prompt_cache: Optional[str] = None
        self._skill_lookup_cache: Dict[str, Dict[str, Any]] = {}
        # Max context tokens - read from env with reasonable default and safety margin
        # Actual model limit is ~182k, but we keep headroom for system prompts + skills + output
        self._MAX_CONTEXT_TOKENS = int(os.getenv("VLLM_CONTEXT_TOKENS", "120000"))
        logger.info("Context token limit set to: %d", self._MAX_CONTEXT_TOKENS)
        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._history: List[ChatMessage] = []
        self._default_filters: Dict[str, Any] = {}
        self._plan_text: str = ""
        self._plan_steps: List[Dict[str, Any]] = []
        self._analysis_text: str = ""
        self._required_information: List[str] = []
        self._missing_information: List[str] = []
        self._global_stop_conditions: List[str] = []
        self._event_handler = None
        self._query_failure_counts: Dict[str, int] = {}
        self._consecutive_retrieve_failures = 0
        self._response_level: str = "1"
        self._response_requirements: Dict[str, Any] = {}
        self._active_skills: List[str] = []
        self._active_policy_skills: List[str] = []
        self._skills_context_text: str = ""
        self._style_context_text: str = ""
        self._lab_key_catalog: List[str] = []
        # Cache for tool results to avoid duplicate calls with identical arguments
        self._tool_result_cache: Dict[str, Tuple[List[Dict[str, Any]], Dict[str, Any], str]] = {}
        # Track queries that returned no results to avoid re-querying
        self._negative_query_cache: Set[str] = set()

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @property
    def history(self) -> Sequence[ChatMessage]:
        return tuple(self._history)

    # ------------------------------------------------------------------
    # Event streaming helpers
    # ------------------------------------------------------------------
    def _emit_event(self, event_type: str, **payload: Any) -> None:
        if not self._event_handler:
            return
        event = {
            "type": event_type,
            "payload": payload,
            "timestamp": time.time(),
        }
        try:
            self._event_handler(event)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to dispatch agent event %s", event_type)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def answer_with_rag(
        self,
        question: str,
        *,
        conversation_history: str = "",
        patient_context: str = "",
        answer_schema: Optional[str] = None,
        report_type: Optional[str] = None,
        report_date: Optional[str] = None,
        reference_date: Optional[str] = None,
        top_k: int = 5,
        event_handler: Optional[Callable[[Dict[str, Any]], None]] = None,
        answer_language: str = "English",
    ) -> ChatMessage:
        """Plan, execute tools, and summarise a clinical answer."""

        history_text = (conversation_history or "").strip()
        base_question = question.strip()
        answer_schema = (answer_schema or "").strip()
        question = base_question
        if history_text:
            question = (
                "The user is continuing a conversation. Previous exchanges:\n"
                f"{history_text}\n\nCurrent user question:\n{base_question}"
            )

        self.reset()
        self._event_handler = event_handler
        self._reference_date = reference_date

        default_time_scope = self._infer_time_scope(report_date)
        time_scope_fields = self._derive_time_scope_fields(report_date, default_time_scope)
        self._default_filters = {
            "query": question.strip(),
            "answer_schema": answer_schema,
            "top_k": top_k,
            "report_type": report_type,
            "report_date": report_date,
            "time_scope": default_time_scope,
            **time_scope_fields,
        }

        self._emit_event(
            "run_started",
            question=question,
            patient_context=patient_context,
            answer_schema=answer_schema,
            report_type=report_type,
            report_date=report_date,
            reference_date=reference_date,
            time_scope=default_time_scope,
            top_k=top_k,
        )

        intro_message = self._format_intro_message(
            question,
            patient_context,
            answer_schema,
            report_type,
            report_date,
            reference_date,
            default_time_scope,
            top_k,
        )
        self._append_user_message(intro_message)

        allowed_tools_payload = self._allowed_tools_payload(full=True)
        allowed_tool_names_json = self._allowed_tools_payload(full=False)
        inferred_patient_id = self._infer_patient_id()
        inferred_db_path = self._infer_db_path()
        self._lab_key_catalog = (
            fetch_patient_lab_keys(inferred_patient_id, db_path=inferred_db_path) if inferred_patient_id else []
        )
        assessment_result = self._run_signature(
            AssessAndSelectSkills,
            {
                "question": question,
                "patient_context": patient_context or "No additional patient data provided.",
                "answer_schema": answer_schema,
                "default_filters": json.dumps(self._format_tool_filters(self._default_filters), ensure_ascii=False),
                "allowed_tools": allowed_tools_payload,
                "skill_summaries": self._skill_prompt_text(),
                "lab_key_catalog": json.dumps(self._lab_key_catalog, ensure_ascii=False),
            },
        )

        self._analysis_text = self._normalize_text(assessment_result.get("analysis")).strip()
        self._required_information = self._parse_json_list(assessment_result.get("required_information"))
        self._missing_information = self._parse_json_list(assessment_result.get("missing_information"))
        self._active_skills = self._normalize_skill_ids(
            self._parse_json_list(assessment_result.get("selected_skills"))
        )
        # Heuristic hints to keep style selection robust when question_type metadata is absent.
        for hint in self._heuristic_style_suggestions(question):
            if self._skill_entry(hint) and hint not in self._active_skills:
                self._active_skills.append(hint)
        # Always enforce structured output format.
        if self._skill_entry("style.structured_annotation") and "style.structured_annotation" not in self._active_skills:
            self._active_skills.append("style.structured_annotation")
        if "style.base" not in self._active_skills and self._skill_entry("style.base"):
            self._active_skills.append("style.base")
        # Swap base style skills for language-specific variants when available.
        self._localize_skill_ids(answer_language)
        self._active_policy_skills = self._select_policy_skills(self._active_skills)
        workflow_categories = {"workflows", "parsing", "knowledge"}
        style_categories = {"style"}
        self._skills_context_text = render_skill_context(self._active_skills, categories=workflow_categories)
        self._style_context_text = render_skill_context(self._active_skills, categories=style_categories)
        self._response_level = self._normalize_text(assessment_result.get("response_level")).strip() or "1"
        self._response_requirements = self._parse_json_response(
            assessment_result.get("response_requirements")
        ) or {}
        if answer_schema:
            self._response_requirements["answer_schema"] = answer_schema
        style_requirements = self._collect_style_response_requirements()
        if style_requirements.get("response_level_hint") and not self._response_level:
            self._response_level = str(style_requirements["response_level_hint"])
        self._response_requirements = self._merge_style_requirements(
            self._response_requirements,
            style_requirements,
        )

        valid_plan = False
        plan_attempts = 0
        plan_repair_note = ""
        while plan_attempts < 3 and not valid_plan:
            plan_attempts += 1
            plan_question = question
            if plan_repair_note:
                plan_question = f"{question}\n\n{plan_repair_note}"
            plan_result = self._run_signature(
                BuildToolPlan,
                {
                    "question": plan_question,
                    "patient_context": patient_context or "No additional patient data provided.",
                    "answer_schema": answer_schema,
                    "default_filters": json.dumps(self._format_tool_filters(self._default_filters), ensure_ascii=False),
                    "allowed_tools": allowed_tools_payload,
                    "skills_context": self._skills_context_text or "",
                    "lab_key_catalog": json.dumps(self._lab_key_catalog, ensure_ascii=False),
                },
            )

            plan_raw = plan_result.get("tool_plan")
            self._plan_text = self._stringify_plan(plan_raw).strip()
            plan_object = self._parse_plan_object(self._plan_text)
            self._plan_steps = self._extract_plan_steps(plan_object)
            self._global_stop_conditions = self._parse_json_list(plan_object.get("global_stop_conditions")) if isinstance(plan_object, dict) else []
            valid_plan = bool(self._plan_steps) and all(step.get("tool_name") for step in self._plan_steps)

            if not valid_plan and plan_attempts < 3:
                plan_repair_note = (
                    "REPAIR: The previous tool plan was invalid. Return valid JSON with "
                    "a non-empty steps array. Each step must include step_number, objective, "
                    "tool_name, arguments (JSON object), evidence_required (array), and stop_if."
                )
                self._emit_event(
                    "plan_retry",
                    attempt=plan_attempts,
                    reason="BuildToolPlan returned an empty or invalid plan.",
                    plan_text=self._plan_text,
                )

        if not valid_plan:
            raise RuntimeError("BuildToolPlan did not return a valid tool plan.")

        self._emit_event(
            "plan_ready",
            analysis=self._analysis_text,
            required_information=self._required_information,
            missing_information=self._missing_information,
            answer_schema=answer_schema,
            skills=self._active_skills,
            policy_skills=self._active_policy_skills,
            skill_workflow_context=self._skills_context_text,
            skill_style_context=self._style_context_text,
            lab_key_catalog=self._lab_key_catalog,
            plan_text=self._plan_text,
            plan_steps=self._plan_steps,
            global_stop_conditions=self._global_stop_conditions,
        )

        plan_overview_message = [
            "Tool plan initialized.",
            "\nPlan (JSON):",
            self._plan_text or "{}",
        ]
        self._append_assistant_message("\n".join(plan_overview_message), signature="BuildToolPlan")

        collected_nodes: List[Dict[str, Any]] = []
        executed_actions: List[Dict[str, Any]] = []
        last_action_record: Dict[str, Any] | None = None
        previous_summaries: List[str] = [self._analysis_text] if self._analysis_text else []
        applied_filter_notes: List[str] = []

        step_index = 0
        executed_tool_this_step = False
        consecutive_empty_rounds = 0
        executed_queries: set[tuple[str, str]] = set()
        tool_rounds_exhausted = False  # set True when loop ends without a natural 'finish'
        stop_conditions_json = json.dumps(self._global_stop_conditions, ensure_ascii=False)

        for round_idx in range(self._max_tool_rounds):
            total_steps = len(self._plan_steps)
            plan_chunk_obj = self._plan_steps[step_index] if step_index < total_steps else {}
            plan_chunk = json.dumps(plan_chunk_obj, ensure_ascii=False) if plan_chunk_obj else "{}"
            plan_chunk_pretty = (
                json.dumps(plan_chunk_obj, ensure_ascii=False, indent=2) if plan_chunk_obj else "{}"
            )

            previous_text = self._build_execution_context(
                collected_nodes=collected_nodes,
                missing_information=self._missing_information,
                last_action_record=last_action_record,
                previous_summaries=previous_summaries,
            )

            self._emit_event(
                "execution_step",
                round=round_idx + 1,
                step_index=step_index,
                total_steps=total_steps,
                plan_chunk=plan_chunk_obj,
            )

            action_result = self._run_signature(
                ToolExecutionStep,
                {
                    "question": question,
                    "patient_context": patient_context or "",
                    "answer_schema": answer_schema,
                    "skills_context": self._skills_context_text or "",
                    "lab_key_catalog": json.dumps(self._lab_key_catalog, ensure_ascii=False),
                    "plan_chunk": plan_chunk,
                    "previous_results": previous_text,
                    "evidence_required": json.dumps(plan_chunk_obj.get("evidence_required", []), ensure_ascii=False)
                    if plan_chunk_obj
                    else "[]",
                    "required_information": json.dumps(self._required_information, ensure_ascii=False),
                    "missing_information": json.dumps(self._missing_information, ensure_ascii=False),
                    "allowed_tools": allowed_tool_names_json,
                    "global_stop_conditions": stop_conditions_json,
                },
            )
            action_label = self._normalize_text(action_result.get("action")).strip().lower()
            tool_name = self._normalize_text(action_result.get("tool_name")).strip()
            raw_arguments = action_result.get("arguments", "")
            rationale = self._normalize_text(action_result.get("rationale")).strip()

            if not action_label:
                retry_note = (
                    "VALIDATION ERROR: The prior ToolExecutionStep output did not include a valid "
                    "\"action\". Return JSON with action in {\"call_tool\",\"skip\",\"finish\"}. "
                    "If action is call_tool, include tool_name and arguments."
                )
                action_result = self._run_signature(
                    ToolExecutionStep,
                    {
                        "question": question,
                        "patient_context": patient_context or "",
                        "answer_schema": answer_schema,
                        "skills_context": self._skills_context_text or "",
                        "lab_key_catalog": json.dumps(self._lab_key_catalog, ensure_ascii=False),
                        "plan_chunk": plan_chunk,
                        "previous_results": f"{previous_text}\n\n{retry_note}",
                        "evidence_required": json.dumps(plan_chunk_obj.get("evidence_required", []), ensure_ascii=False)
                        if plan_chunk_obj
                        else "[]",
                        "required_information": json.dumps(self._required_information, ensure_ascii=False),
                        "missing_information": json.dumps(self._missing_information, ensure_ascii=False),
                        "allowed_tools": allowed_tool_names_json,
                        "global_stop_conditions": stop_conditions_json,
                    },
                )
                action_label = self._normalize_text(action_result.get("action")).strip().lower()
                tool_name = self._normalize_text(action_result.get("tool_name")).strip()
                raw_arguments = action_result.get("arguments", "")
                rationale = self._normalize_text(action_result.get("rationale")).strip()

            if not action_label:
                raise RuntimeError("ToolExecutionStep did not return an action to execute.")

            self._emit_event(
                "execution_decision",
                round=round_idx + 1,
                step_index=step_index,
                action=action_label,
                tool_name=tool_name,
                arguments_raw=raw_arguments,
                rationale=rationale,
            )

            logger.debug(
                "Plan progress: step %d/%d, action=%s, tool=%s, stop_conditions=%s",
                step_index + 1,
                total_steps,
                action_label or "(leer)",
                tool_name or "(keins)",
                self._global_stop_conditions,
            )

            action_summary = self._format_action_summary(
                action_label,
                tool_name,
                raw_arguments,
                rationale,
                plan_chunk_pretty,
            )
            self._append_assistant_message(action_summary, signature="ToolExecutionStep", round=round_idx + 1)

            if action_label in {"", "finish"}:
                # Guard: do not allow finish before at least one tool call for this step,
                # BUT allow finish when all planned steps have already been completed.
                if action_label == "finish" and not executed_tool_this_step and step_index < total_steps:
                    # No tool has been executed in this step; force a retry.
                    retry_note = (
                        "REPAIR: Do not choose 'finish' before executing the planned tool at least once. "
                        "Call the tool for this step or explicitly 'skip' it."
                    )
                    previous_text = f"{previous_text}\n\n{retry_note}"
                    continue
                if action_label == "finish" and rationale:
                    # Preserve the finish rationale so it appears in the plan overview.
                    previous_summaries.append(f"Finish rationale: {rationale}")
                tool_rounds_exhausted = False  # natural finish
                break

            if action_label == "skip":
                step_index += 1
                executed_tool_this_step = False
                continue

            if action_label != "call_tool":
                previous_summaries.append(f"Unbekannte Aktion '{action_label}'.")
                step_index += 1
                continue

            arguments = self._parse_arguments(raw_arguments)

            # Fix B: Block duplicate queries (check both raw and sanitized forms)
            query_value = arguments.get("query", "") if isinstance(arguments, dict) else ""
            query_key = (tool_name, query_value)
            sanitized_key = (tool_name, self._sanitize_query(query_value)) if query_value else query_key
            if query_value and (query_key in executed_queries or sanitized_key in executed_queries):
                logger.warning(
                    "Duplicate query blocked for %s: '%s'",
                    tool_name,
                    query_value[:80],
                )
                previous_summaries.append(
                    f"DUPLICATE BLOCKED: Query '{query_value[:80]}' was already executed with "
                    f"tool '{tool_name}'. Reformulate with different keywords/scope or choose 'finish'."
                )
                # Do NOT count as consecutive empty round — give the model a chance to reformulate
                continue
            if query_value:
                executed_queries.add(query_key)
                if sanitized_key != query_key:
                    executed_queries.add(sanitized_key)

            self._emit_event(
                "tool_started",
                round=round_idx + 1,
                step_index=step_index,
                tool_name=tool_name,
                arguments=arguments,
            )
            nodes, action_record, summary_text = self._call_tool(tool_name, arguments)
            self._emit_event(
                "tool_finished",
                round=round_idx + 1,
                step_index=step_index,
                tool_name=tool_name,
                action_record=action_record,
                nodes=nodes,
                summary=summary_text,
            )
            executed_actions.append(action_record)
            last_action_record = action_record
            executed_tool_this_step = True

            error_occurred = action_record.get("is_error", False)

            if summary_text:
                previous_summaries.append(summary_text)

            if nodes:
                consecutive_empty_rounds = 0
                nodes_added_this_round = len(nodes)
                collected_nodes.extend(nodes)
                # Check context budget - stop retrieving if approaching limit
                current_tokens = self._estimate_context_tokens(collected_nodes)
                if current_tokens >= self._MAX_CONTEXT_TOKENS:
                    logger.warning(
                        "Context budget reached (%d tokens >= %d limit), stopping retrieval early",
                        current_tokens,
                        self._MAX_CONTEXT_TOKENS,
                    )
                    self._emit_event(
                        "context_budget_reached",
                        current_tokens=current_tokens,
                        max_tokens=self._MAX_CONTEXT_TOKENS,
                        round=round_idx + 1,
                    )
                    previous_summaries.append(
                        f"Context budget reached ({current_tokens} tokens), proceeding to answer."
                    )
                    break

                # Heuristic sufficiency check: inform model about context size
                total_nodes = len(collected_nodes)
                if round_idx >= 2 and total_nodes >= 20:  # After 3+ rounds with 20+ nodes
                    sufficiency_hint = (
                        f"CONTEXT STATUS: You now have {total_nodes} context nodes ({current_tokens} tokens). "
                        f"This round added {nodes_added_this_round} nodes. "
                        "Consider whether critical evidence is still missing before continuing."
                    )
                    previous_summaries.append(sufficiency_hint)
                    logger.info(
                        "Sufficiency check: %d nodes, %d tokens after round %d",
                        total_nodes,
                        current_tokens,
                        round_idx + 1,
                    )
            else:
                # Track consecutive empty rounds for soft hints
                if not error_occurred:
                    consecutive_empty_rounds += 1
                # Soft hint: nudge the agent to consider finishing, but let it decide
                if collected_nodes and round_idx >= 1:
                    if consecutive_empty_rounds >= 2:
                        exhaustion_hint = (
                            f"NOTE: {consecutive_empty_rounds} consecutive queries returned no new results. "
                            f"You have {len(collected_nodes)} existing context nodes. "
                            "Consider finishing if the available evidence is sufficient, or try "
                            "substantially different search terms if critical evidence is still missing."
                        )
                    else:
                        exhaustion_hint = (
                            f"This query returned no new results. You have {len(collected_nodes)} existing nodes. "
                            "Try different search terms if critical evidence is still missing, or finish if sufficient."
                        )
                    previous_summaries.append(exhaustion_hint)
                    logger.info(
                        "No new nodes from query in round %d (consecutive_empty=%d), hinting agent",
                        round_idx + 1,
                        consecutive_empty_rounds,
                    )

            if error_occurred:
                names = self._available_tool_names()
                if names:
                    previous_summaries.append("Verfügbare Werkzeuge: " + ", ".join(names))
                # Compare against the entry *before* the one we just appended.
                if len(executed_actions) >= 2:
                    prev = executed_actions[-2]
                    if prev.get("is_error") and prev.get("tool") == tool_name and prev.get("arguments") == action_record.get("arguments"):
                        logger.warning(
                            "Duplicate failed tool call detected for %s with arguments %s — skipping step",
                            tool_name,
                            action_record.get("arguments"),
                        )
                        previous_summaries.append(
                            f"Tool '{tool_name}' failed twice with same arguments — skipping step."
                        )
                        step_index += 1
                        executed_tool_this_step = False
                        continue
                continue
            filters_text = self._summarise_filters_from_action(action_record)
            if filters_text:
                applied_filter_notes.append(filters_text)

            step_index += 1
            executed_tool_this_step = False

        # If no context nodes were retrieved, do not fail the run; allow the model to
        # return a structured "nicht berechenbar" answer with an explicit rationale.
        if not collected_nodes:
            deduped_nodes = []
            plan_execution_summary = (
                "\n".join(previous_summaries)
                or "No context retrieved; tool calls returned no results."
            )
            applied_filters_summary = "\n".join(applied_filter_notes) or "Default filters applied without changes."
            context_snippets = "No context snippets available (no tool results)."
            citations_metadata = []
            # Relax citation requirement and nudge the formatter to emit a structured
            # "Nicht berechenbar" answer when nothing could be retrieved.
            self._response_requirements["citations_required"] = False
            self._style_context_text += (
                "\nREPAIR: No evidence was found in tool calls; respond with "
                "'Answer: Nicht berechenbar' and a short Reasoning that the required "
                "labs/reports are missing in the searched window."
            )
            self._emit_event(
                "context_compiled",
                executed_actions=executed_actions,
                applied_filters_summary=applied_filters_summary,
                context_nodes=deduped_nodes,
                context_token_total=0,
                plan_execution_summary=plan_execution_summary,
                citations=citations_metadata,
            )
            policy_result: Dict[str, Any] = {"resolution_action": "select", "final_claim": None}
            policy_trace: Dict[str, Any] = {"notice": "No evidence available to rank."}
        else:
            deduped_nodes = self._assign_citation_ids(self._deduplicate_context(collected_nodes))
            plan_execution_summary = "\n".join(previous_summaries) or "No tools were required."
            applied_filters_summary = "\n".join(applied_filter_notes) or "Default filters applied without changes."
            context_snippets = self._format_context_snippets(
                deduped_nodes,
                limit=len(deduped_nodes),
                max_chars=None,
            )
            citations_metadata = self._build_citations_metadata(deduped_nodes)

            self._emit_event(
                "context_compiled",
                executed_actions=executed_actions,
                applied_filters_summary=applied_filters_summary,
                context_nodes=deduped_nodes,
                context_token_total=self._estimate_context_tokens(deduped_nodes),
                plan_execution_summary=plan_execution_summary,
                citations=citations_metadata,
            )

        overview_header = [
            f"Analysis: {self._analysis_text or '(none)'}",
            "Required information: " + (", ".join(self._required_information) if self._required_information else "(none)"),
            "Remaining gaps: " + (", ".join(self._missing_information) if self._missing_information else "(none)"),
            "",
            "Execution summary:",
            plan_execution_summary,
        ]
        plan_overview_text = "\n".join(overview_header)

        # ---------------- Policy engine -----------------
        policy_result: Dict[str, Any] = locals().get("policy_result", {})
        policy_trace: Dict[str, Any] = locals().get("policy_trace", {})
        if deduped_nodes:
            policy_result = {}
            policy_trace = {}
            try:
                question_type = self._infer_question_type_from_skills()
                target_concepts, concept_search_terms = extract_target_concepts(question)
                evidence_items = normalise_nodes_to_evidence(
                    deduped_nodes, target_concepts or None, concept_search_terms or None
                )
                policy_result, policy_trace = apply_policy(
                    self._active_policy_skills,
                    evidence_items,
                    question_type,
                    target_concepts=target_concepts,
                    reference_date=self._reference_date,
                )
            except Exception as exc:  # pragma: no cover - defensive
                policy_result = {"resolution_action": "select", "final_claim": None, "error": str(exc)}
                policy_trace = {"error": str(exc)}

            self._emit_event(
                "policy_applied",
                policy_skills=self._active_policy_skills,
                policy_result=policy_result,
                policy_trace=policy_trace,
            )

        if deduped_nodes:
            self._log_context_metrics(deduped_nodes)

        if tool_rounds_exhausted:
            self._style_context_text += (
                f"\nTOOL LIMIT: All {self._max_tool_rounds} retrieval rounds have been used. "
                "You must now commit to the best answer supported by the evidence collected. "
                "Do not output 'Abstain' or leave the answer blank. "
                "If evidence is incomplete, choose the most defensible answer and clearly express "
                "uncertainty in the Reasoning field."
            )

        summary_result: Dict[str, Any] = {}
        final_answer = ""
        max_summary_attempts = 2
        for attempt in range(max_summary_attempts):
            summary_result = self._run_signature(
                DraftFinalAnswer,
                {
                    "question": question,
                    "answer_schema": answer_schema,
                    "answer_language": answer_language,
                    "patient_context": patient_context or "No additional patient data provided.",
                    "plan_overview": plan_overview_text,
                    "applied_filters": applied_filters_summary,
                    "context_snippets": context_snippets or "No context snippets available.",
                    "outstanding_information": json.dumps(self._missing_information, ensure_ascii=False),
                    "citations": json.dumps(
                        [{k: v for k, v in c.items() if k != "snippet"} for c in citations_metadata],
                        ensure_ascii=False,
                    ),
                    "response_level": self._response_level,
                    "response_requirements": json.dumps(self._response_requirements, ensure_ascii=False),
                    "style_context": self._style_context_text or "",
                    "policy_result": json.dumps(policy_result, ensure_ascii=False),
                    "policy_trace": json.dumps({"ranked_ids": policy_trace.get("ranked_ids", [])}, ensure_ascii=False),
                },
            )
            final_answer = self._normalize_text(summary_result.get("final_answer")).strip()
            if not final_answer and isinstance(summary_result.get("content"), str):
                final_answer = summary_result.get("content", "").strip()
            # Enforce structured template when present.
            def _is_structured(text: str) -> bool:
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                return len(lines) >= 2 and lines[0].startswith("Answer:") and lines[1].startswith("Reasoning:")

            if final_answer and _is_structured(final_answer):
                break
            if attempt < max_summary_attempts - 1:
                self._emit_event(
                    "summary_retry",
                    attempt=attempt + 1,
                    reason="format_invalid" if final_answer else "empty_final_answer",
                    draft=final_answer,
                )
                # Append repair note to style context for next attempt.
                self._style_context_text += "\nREPAIR: Format must be exactly two lines labeled 'Answer:' and 'Reasoning:'; no extra lines; citations only in Reasoning."
                final_answer = ""
                continue
            self._emit_event(
                "summary_retry",
                attempt=attempt + 1,
                reason="empty_final_answer",
            )
        hallucinated_ids: set[str] = set()
        if not final_answer:
            raise RuntimeError("DraftFinalAnswer returned no content; aborting run.")
        # Final safeguard: if the model still failed to emit the required Answer/Reasoning
        # structure, wrap it to keep outputs consistent.
        if not _is_structured(final_answer):
            final_answer = f"Answer: {final_answer}\nReasoning: Kein Reasoning angegeben."

        final_answer, numeric_hallucinated = self._replace_numeric_citations(final_answer, deduped_nodes)
        final_answer = self._normalise_citation_brackets(final_answer)
        hallucinated_ids.update(numeric_hallucinated)
        citations_metadata = self._ensure_citation_entries(final_answer, citations_metadata)
        final_answer = self._normalise_citation_ids(final_answer, citations_metadata, hallucinated_ids)

        self._emit_event(
            "summary_ready",
            final_answer=final_answer,
            missing_information=self._missing_information,
            required_information=self._required_information,
            context_token_total=self._estimate_context_tokens(deduped_nodes),
        )

        hallucination_note = ""
        if hallucinated_ids:
            hallucination_note = (
                "\n\nHinweis: Einige Quellen wurden nicht in den abgerufenen Dokumenten gefunden "
                "und könnten halluziniert sein."
            )
            final_answer += hallucination_note

        final_message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=final_answer,
            additional_kwargs={
                "context_nodes": deduped_nodes,
                "citations": citations_metadata,
                "actions": executed_actions,
                "hallucinated_citations": sorted(hallucinated_ids),
                "subqueries": [
                    record.get("arguments", {}).get("query")
                    for record in executed_actions
                    if isinstance(record.get("arguments"), dict) and record.get("arguments", {}).get("query")
                ],
                "analysis": self._analysis_text,
                "required_information": self._required_information,
                "missing_information": self._missing_information,
                "final_answer": final_answer,
                "context_tokens_used": self._estimate_context_tokens(deduped_nodes),
                "response_level": self._response_level,
                "response_requirements": self._response_requirements,
            },
        )
        self._history.append(final_message)
        self._default_filters = {}
        self._emit_event(
            "run_completed",
            answer=final_message.content,
            metadata=final_message.additional_kwargs,
        )
        self._validate_citations(final_message.content, citations_metadata)
        return final_message

    # ------------------------------------------------------------------
    # Signature helpers
    # ------------------------------------------------------------------
    def _run_signature(
        self,
        signature_cls: Type,
        inputs: Dict[str, Any],
        *,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        import time as _time

        prompt = self._build_signature_prompt(signature_cls, inputs)
        self._emit_event("signature_started", signature=signature_cls.__name__, inputs=inputs)
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                completion = self._complete(prompt)
                last_exc = None
                break
            except Exception as exc:  # pragma: no cover - defensive
                last_exc = exc
                logger.warning(
                    "Signature %s attempt %d/%d failed: %s",
                    signature_cls.__name__, attempt + 1, max_retries, exc,
                )
                if attempt < max_retries - 1:
                    _time.sleep(2 ** attempt)
                    continue
                logger.exception("Signature %s failed after %d attempts", signature_cls.__name__, max_retries)
                error_payload = {
                    "signature": signature_cls.__name__,
                    "error": str(exc),
                    "error_class": exc.__class__.__name__,
                }
                self._emit_event("signature_failed", **error_payload)
                self._emit_event("llm_error", **error_payload)
                raise
        parsed = self._parse_signature_output(completion, signature_cls)
        logger.debug("Signature %s output: %s", signature_cls.__name__, parsed)
        self._emit_event(
            "signature_completed",
            signature=signature_cls.__name__,
            outputs=parsed,
        )
        return parsed

    def _build_signature_prompt(self, signature_cls: Type, inputs: Dict[str, Any]) -> str:
        instructions = getattr(signature_cls, "instructions", "")
        input_blocks: List[str] = []
        for name, field in signature_cls.input_fields.items():  # type: ignore[attr-defined]
            value = inputs.get(name, "")
            desc = field.json_schema_extra.get("desc", "")
            prefix = field.json_schema_extra.get("prefix", f"{name}:")
            formatted_value = textwrap.indent(str(value or ""), "    ").rstrip()
            block_lines = [f"{prefix} ({desc})", formatted_value or "    (leer)"]
            input_blocks.append("\n".join(block_lines))

        output_lines = []
        for name, field in signature_cls.output_fields.items():  # type: ignore[attr-defined]
            desc = field.json_schema_extra.get("desc", "")
            output_lines.append(f'- "{name}": {desc}')

        prompt_parts = [
            instructions,
            "",
            "Eingaben:",
            "\n\n".join(input_blocks) if input_blocks else "(keine Eingaben)",
            "",
            "Erzeuge eine JSON-Antwort mit genau diesen Feldern:",
            "\n".join(output_lines),
            "",
            "JSON:",
        ]
        return "\n".join(prompt_parts)

    def _parse_signature_output(self, completion: str, signature_cls: Type) -> Dict[str, Any]:
        candidate = self._extract_json_object(completion)
        parsed_dict = candidate if isinstance(candidate, dict) else {}
        outputs: Dict[str, Any] = {}
        for name in signature_cls.output_fields.keys():  # type: ignore[attr-defined]
            outputs[name] = parsed_dict.get(name)
        if (
            signature_cls.__name__ == "DraftFinalAnswer"
            and (not outputs.get("final_answer"))
        ):
            outputs["final_answer"] = completion.strip()
        return outputs

    def _extract_json_object(self, text: str) -> Any:
        if not text:
            return {}
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        candidate = fenced.group(1) if fenced else text
        candidate = candidate.strip()
        if not candidate:
            return {}
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            collapsed = self._collapse_string_concatenation(candidate)
            if collapsed != candidate:
                try:
                    return json.loads(collapsed)
                except json.JSONDecodeError:
                    candidate = collapsed
            match = re.search(r"\{.*\}", candidate, re.DOTALL)
            if match:
                fragment = self._collapse_string_concatenation(match.group(0))
                try:
                    return json.loads(fragment)
                except json.JSONDecodeError:
                    logger.debug("Failed to parse JSON fragment: %s", candidate)
        return {}

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------
    def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
        arguments = dict(arguments or {})
        action_record: Dict[str, Any] = {
            "tool": tool_name or "",
            "arguments": arguments,
        }

        if not tool_name:
            action_record["result"] = "Kein Werkzeugname angegeben."
            action_record["is_error"] = True
            return [], action_record, action_record["result"]

        try:
            tool = self._get_tool(tool_name)
        except ValueError as exc:
            error_text = str(exc)
            action_record["result"] = error_text
            action_record["is_error"] = True
            self._append_assistant_message(
                f"Werkzeug '{tool_name}' steht nicht zur Verfügung: {error_text}",
                tool=tool_name,
                error=True,
            )
            return [], action_record, error_text

        if getattr(self, "_default_filters", None) and tool_name == "retrieve_reports":
            for key, value in self._default_filters.items():
                if value is not None and key not in arguments:
                    arguments[key] = value

        # Fix C: Sanitize answer-schema tokens from retrieval queries
        if tool_name in ("retrieve_reports", "retrieve_lab_values") and "query" in arguments:
            original_query = arguments["query"]
            sanitized = self._sanitize_query(original_query)
            if sanitized != original_query:
                logger.info("Sanitized query: '%s' -> '%s'", original_query, sanitized)
                arguments["query"] = sanitized
                action_record["arguments"] = arguments

        canonical_query_key = ""
        adjustment_note = None
        if tool_name == "retrieve_reports":
            canonical_query_key = self._canonical_query_key(arguments.get("query"))
            arguments, adjustment_note = self._auto_adjust_retrieve_arguments(arguments, canonical_query_key)
            if adjustment_note:
                self._append_assistant_message(
                    adjustment_note,
                    tool=tool_name,
                    auto_adjustment=True,
                )

        if tool_name == "retrieve_reports" and isinstance(arguments.get("report_type"), list):
            report_list = [
                rtype for rtype in arguments["report_type"] if isinstance(rtype, str) and rtype.strip()
            ]
            if report_list:
                arguments["report_type"] = report_list
            else:
                arguments.pop("report_type", None)

        # Check cache for duplicate tool calls (after all argument adjustments)
        cache_key = (tool_name, json.dumps(arguments, sort_keys=True))
        if cache_key in self._tool_result_cache:
            cached_nodes, cached_action, cached_summary = self._tool_result_cache[cache_key]
            cache_hit_msg = f"Tool {tool_name} with these arguments was already called in this run. Returning cached result with {len(cached_nodes)} nodes."
            logger.info("Cache hit: %s", cache_hit_msg)
            self._append_assistant_message(cache_hit_msg, tool=tool_name, arguments=arguments, cache_hit=True)
            # Return cached results directly
            return cached_nodes, cached_action, cached_summary
        
        # Check negative cache for retrieve_reports to avoid re-querying unsuccessful patterns
        if tool_name == "retrieve_reports":
            query_key = f"{arguments.get('query', '')}|{arguments.get('report_type', 'all')}"
            if query_key in self._negative_query_cache:
                negative_msg = f"Query '{arguments.get('query', '')}' previously returned no results. Consider broadening search or trying different terms."
                logger.info("Negative cache hit: %s", negative_msg)
                self._append_assistant_message(negative_msg, tool=tool_name, arguments=arguments, negative_cache_hit=True)
                # Return empty results immediately
                return [], {}, "No results (negative cache hit)"

        invocation_text = f"Calling tool {tool_name} with arguments {arguments}."
        self._append_assistant_message(invocation_text, tool=tool_name, arguments=arguments)

        output = self._safe_tool_call(tool, arguments)
        payload = getattr(output, "raw_output", None)
        if not isinstance(payload, dict):
            payload = self._parse_json_response(getattr(output, "content", None))

        tool_message_content = self._safe_json_dumps(payload) or (output.content or "")
        self._append_tool_message(
            tool.metadata.name,
            payload or {},
            tool_message_content,
            arguments=arguments,
        )

        nodes: List[Dict[str, Any]] = []
        if isinstance(payload, dict):
            nodes = payload.get("context_nodes") or []
            if not isinstance(nodes, list):
                nodes = []

        summary_text = ""
        if nodes:
            summary_text = self._summarise_nodes(nodes)
            action_record["result_count"] = len(nodes)
        else:
            summary_text = self._summarise_tool_response(output)
            action_record["result"] = summary_text
        if (
            tool_name == "retrieve_reports"
            and isinstance(arguments, dict)
            and isinstance(arguments.get("query"), str)
        ):
            query = arguments["query"].strip()
            lower_q = query.lower()
            generic_tokens = {"arztbrief", "berichte", "bericht", "labor", "labore", "report", "reports"}
            if (len(query.split()) <= 6 and any(tok in lower_q for tok in generic_tokens)) or lower_q in {"arztbrief", "berichte"}:
                warning = (
                    "The previous retrieve_reports query was generic. Rewrite it with concrete keywords from the task "
                    "(e.g., disease name, remission criteria, prior therapy) before calling the tool again."
                )
                summary_text = f"{warning}\nPrevious query: {query}"
                action_record["result"] = summary_text
        action_record["is_error"] = getattr(output, "is_error", False)

        if tool_name == "retrieve_reports":
            key = canonical_query_key or self._canonical_query_key(arguments.get("query"))
            query_key = f"{arguments.get('query', '')}|{arguments.get('report_type', 'all')}"
            if nodes:
                self._query_failure_counts[key] = 0
                self._consecutive_retrieve_failures = 0
                # Remove from negative cache if it was there
                self._negative_query_cache.discard(query_key)
            else:
                self._query_failure_counts[key] = self._query_failure_counts.get(key, 0) + 1
                self._consecutive_retrieve_failures += 1
                # Add to negative cache after first failure
                self._negative_query_cache.add(query_key)
        else:
            self._consecutive_retrieve_failures = 0

        # Store result in cache for this run (after all processing complete)
        cache_key = (tool_name, json.dumps(arguments, sort_keys=True))
        self._tool_result_cache[cache_key] = (nodes, action_record, summary_text)

        return nodes, action_record, summary_text

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    def _infer_time_scope(self, report_date: Optional[str]) -> str:
        if not report_date:
            return "all"
        text = str(report_date).strip().lower()
        if not text:
            return "all"
        if ".." in text or " to " in text or " bis " in text:
            return "range"
        if text in {"latest", "recent", "current", "now"}:
            return "latest"
        return "date"

    def _format_intro_message(
        self,
        question: str,
        patient_context: str,
        answer_schema: str,
        report_type: Optional[str],
        report_date: Optional[str],
        reference_date: Optional[str],
        time_scope: str,
        top_k: int,
    ) -> str:
        reference_line = ""
        if reference_date:
            formatted_ref = self._format_reference_date(reference_date)
            reference_line = (
                f"Heutiges Datum (entspricht dem letzten dokumentierten Bericht): {formatted_ref}"
            )
        lines = [
            "Aufgabe:",
            question,
            *(["", "Antwortschema:", answer_schema] if answer_schema else []),
            "",
            "Patientenkontext:",
            patient_context or "No additional patient data provided.",
            "",
            (
                "Standardfilter: "
                f"report_type={report_type or '-'}, time_scope={time_scope}, "
                f"report_date={report_date or '-'}, top_k={top_k}"
            ),
            reference_line,
        ]
        return "\n".join(line for line in lines if line)

    def _derive_time_scope_fields(self, report_date: Optional[str], time_scope: str) -> Dict[str, Any]:
        if time_scope not in {"date", "range"} or not report_date:
            return {}
        if time_scope == "date":
            normalized = self._normalize_date_token(report_date)
            return {"date_exact": normalized} if normalized else {}
        start_token, end_token = self._split_date_range(report_date)
        start_norm = self._normalize_date_token(start_token)
        end_norm = self._normalize_date_token(end_token)
        if start_norm and end_norm:
            if end_norm < start_norm:
                start_norm, end_norm = end_norm, start_norm
            return {"date_start": start_norm, "date_end": end_norm}
        return {}

    def _default_retrieve_arguments(self) -> Dict[str, Any]:
        keys = ("report_type", "report_date", "time_scope", "date_exact", "date_start", "date_end")
        arguments: Dict[str, Any] = {}
        for key in keys:
            value = self._default_filters.get(key)
            if value not in (None, "", [], {}):
                arguments[key] = value
        return arguments

    _SCHEMA_NOISE_PATTERNS = [
        re.compile(r"Status\s*[=:]\s*(Dokumentiert|Nie verabreicht|Nicht dokumentiert|Unklar)\s*\|?\|?", re.IGNORECASE),
        re.compile(r"Answer:\s*", re.IGNORECASE),
        re.compile(r"Reasoning:\s*", re.IGNORECASE),
        re.compile(r"\bDD\.MM\.YYYY\b", re.IGNORECASE),
        re.compile(r"\|\s*Dokumentiert\b", re.IGNORECASE),
        re.compile(r"\|\s*Berechnet\b", re.IGNORECASE),
        re.compile(r"Status:\s*nicht berechenbar", re.IGNORECASE),
        re.compile(r"\bNie verabreicht\b", re.IGNORECASE),
        re.compile(r"\bNicht dokumentiert\b", re.IGNORECASE),
    ]

    @staticmethod
    def _sanitize_query(query: str) -> str:
        """Remove answer-schema tokens that pollute retrieval queries."""
        cleaned = query
        for pat in DSPyAgentBase._SCHEMA_NOISE_PATTERNS:
            cleaned = pat.sub(" ", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        # Don't return an empty/trivial query
        if len(cleaned) < 3:
            return query
        return cleaned

    @staticmethod
    def _canonical_query_key(text: Optional[str]) -> str:
        if not text:
            return ""
        if isinstance(text, (list, tuple, set)):
            text = " ".join([str(item) for item in text])
        if not isinstance(text, str):
            text = str(text)
        tokens = re.findall(r"[A-Za-zÄÖÜäöüß0-9]+", text.lower())
        keywords = [tok for tok in tokens if len(tok) > 2]
        if not keywords:
            keywords = tokens
        return " ".join(keywords[:20])

    def _auto_adjust_retrieve_arguments(
        self,
        arguments: Dict[str, Any],
        query_key: str,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        fail_count = self._query_failure_counts.get(query_key, 0)
        if fail_count <= 0:
            return arguments, None
        updated = dict(arguments)
        adjustments: List[str] = []
        original_top_k = updated.get("top_k")
        try:
            current_top_k = int(original_top_k)
        except (TypeError, ValueError):
            current_top_k = int(self._default_filters.get("top_k") or 5)
        if fail_count >= 1:
            boosted = min(self._auto_top_k_cap, max(current_top_k, 5) * 2)
            if boosted > current_top_k:
                updated["top_k"] = boosted
                adjustments.append(f"top_k→{boosted}")
        if fail_count >= 2:
            narrowed = self._keyword_subset_query(updated.get("query"))
            if narrowed and narrowed != updated.get("query"):
                updated["query"] = narrowed
                adjustments.append("keywords_only")
        if fail_count >= 3:
            if updated.get("time_scope") not in (None, "", "all"):
                updated["time_scope"] = "all"
                updated.pop("date_exact", None)
                updated.pop("date_start", None)
                updated.pop("date_end", None)
                adjustments.append("time_scope=all")
        note = None
        if adjustments:
            note = (
                "Automatic retrieve_reports adjustment after repeated empty results: "
                + ", ".join(adjustments)
            )
        return updated, note

    @staticmethod
    def _keyword_subset_query(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        tokens = re.findall(r"[A-Za-zÄÖÜäöüß0-9]+", text)
        if not tokens:
            return None
        keywords = [tok for tok in tokens if len(tok) > 3]
        if len(keywords) < 3:
            keywords = tokens
        seen = []
        for token in keywords:
            lowered = token.lower()
            if lowered not in seen:
                seen.append(lowered)
        selection = seen[:8]
        if not selection:
            return None
        return " ".join(selection)

    @staticmethod
    def _normalize_date_token(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        text = value.strip()
        if not text:
            return None
        patterns = ("%Y-%m-%d", "%d.%m.%Y", "%d-%m-%Y")
        for pattern in patterns:
            try:
                return datetime.strptime(text, pattern).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return None

    @staticmethod
    def _split_date_range(value: str) -> Tuple[Optional[str], Optional[str]]:
        separators = ["..", " to ", " bis "]
        for separator in separators:
            if separator in value:
                left, right = value.split(separator, 1)
                return left.strip(), right.strip()
        return None, None

    @staticmethod
    def _format_reference_date(reference_date: str) -> str:
        normalized = DSPyAgentBase._normalize_date_token(reference_date)
        if normalized:
            year, month, day = normalized.split("-")
            return f"{day}.{month}.{year}"
        return reference_date

    def _format_action_summary(
        self,
        action: str,
        tool_name: str,
        arguments: Any,
        rationale: str,
        plan_chunk: str,
    ) -> str:
        action_text = action or "-"
        tool_text = tool_name or "-"
        arguments_text = self._normalize_text(arguments) or "-"
        rationale_text = rationale or "-"
        plan_text = plan_chunk or "(empty)"
        return "\n".join(
            [
                f"Plan chunk: {plan_text}",
                f"Action proposal: {action_text}",
                f"Tool: {tool_text}",
                f"Arguments: {arguments_text}",
                f"Rationale: {rationale_text}",
            ]
        )

    def _format_context_snippets(self, nodes: List[Dict[str, Any]], limit: int = 5, max_chars: int | None = 220) -> str:
        if not nodes:
            return ""
        snippets: List[str] = []
        for node in nodes[:limit]:
            # Prefer full text when available; fall back to snippet.
            snippet = node.get("text") or node.get("snippet")
            if not snippet:
                continue
            snippet = re.sub(r"\s+", " ", snippet).strip()
            if max_chars and max_chars > 0 and len(snippet) > max_chars:
                snippet = snippet[: max_chars - 3].rstrip() + "..."
            label = node.get("section_name") or node.get("report_type") or node.get("report_id") or "Section"
            citation_id = node.get("citation_id") or "citation"
            snippets.append(f"[{citation_id}] ({label}) {snippet}")
        return "\n".join(snippets)

    def _summarise_filters_from_action(self, action_record: Dict[str, Any]) -> str:
        arguments = action_record.get("arguments")
        if not isinstance(arguments, dict):
            return ""
        if not arguments:
            return ""
        formatted = self._format_tool_filters(arguments)
        if not formatted:
            return ""
        return f"{action_record.get('tool', '')}: {json.dumps(formatted, ensure_ascii=False)}"

    def _build_execution_context(
        self,
        *,
        collected_nodes: List[Dict[str, Any]],
        missing_information: List[str],
        last_action_record: Dict[str, Any] | None,
        previous_summaries: List[str],
    ) -> str:
        context_nodes = self._assign_citation_ids(self._deduplicate_context(collected_nodes))
        evidence_snippets = self._format_context_snippets(
            context_nodes,
            limit=len(context_nodes),
            max_chars=None,
        )
        evidence_block = evidence_snippets or "(none)"
        missing_block = "\n".join(f"- {item}" for item in missing_information) if missing_information else "(none)"
        last_call = "(none)"
        if last_action_record:
            tool_name = last_action_record.get("tool") or ""
            args = last_action_record.get("arguments") or {}
            if isinstance(args, dict) and args:
                args_text = json.dumps(args, ensure_ascii=False)
            else:
                args_text = "{}"
            last_call = f"{tool_name} {args_text}".strip()
        notes = "\n".join(previous_summaries[-3:]) if previous_summaries else ""
        notes_block = notes or "(none)"
        sections = [
            "EvidenceFound:",
            evidence_block,
            "",
            "EvidenceMissing:",
            missing_block,
            "",
            "LastToolCall:",
            last_call,
            "",
            "Notes:",
            notes_block,
        ]
        return "\n".join(sections)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _available_tools_summary(self) -> str:
        summaries: List[str] = []
        for tool in self.tools:
            desc = getattr(tool.metadata, "description", "")
            summaries.append(f"{tool.metadata.name}: {desc}")
        return "\n".join(summaries) if summaries else "No tools registered."

    def _available_tool_names(self) -> List[str]:
        return [tool.metadata.name for tool in self.tools]

    def _allowed_tools_payload(self, full: bool) -> str:
        if not self.tools:
            return "[]"
        if full:
            data = [
                {
                    "name": tool.metadata.name,
                    "description": getattr(tool.metadata, "description", ""),
                }
                for tool in self.tools
            ]
        else:
            data = self._available_tool_names()
        try:
            return json.dumps(data, ensure_ascii=False)
        except TypeError:
            return "[]"

    def _skill_prompt_text(self) -> str:
        if not self._skill_prompt_cache:
            self._skill_prompt_cache = build_skill_prompt()
        return self._skill_prompt_cache

    def _skill_entry(self, skill_id: str) -> Optional[Dict[str, Any]]:
        if not self._skill_lookup_cache:
            self._skill_lookup_cache = {entry["id"]: entry for entry in get_skill_catalog()}
        return self._skill_lookup_cache.get(skill_id)

    def _localize_skill_ids(self, answer_language: str) -> None:
        """Replace base skill IDs with language-specific variants when available.

        E.g. 'style.base' → 'style.base.en' when answer_language='English',
        falling back to the original ID if no variant is registered.
        """
        lang_suffix_map = {"English": ".en", "Français": ".fr", "Spanish": ".es", "Italian": ".it", "Albanian": ".sq", "Turkish": ".tr"}
        suffix = lang_suffix_map.get(answer_language)
        if not suffix:
            return
        self._active_skills = [
            (sid + suffix if self._skill_entry(sid + suffix) else sid)
            for sid in self._active_skills
        ]

    def _collect_style_response_requirements(self) -> Dict[str, Any]:
        aggregated = {
            "templates": [],
            "must_include": [],
            "additional_requirements": [],
            "notes": [],
            "response_level_hint": None,
        }
        for skill_id in self._active_skills:
            entry = self._skill_entry(skill_id)
            if not entry or entry.get("category") != "style":
                continue
            structure = entry.get("structure")
            if isinstance(structure, dict):
                template = structure.get("template")
                if isinstance(template, list):
                    aggregated["templates"].extend(template)
                elif isinstance(template, str):
                    aggregated["templates"].append(template)
                must_include = structure.get("must_include")
                if isinstance(must_include, list):
                    aggregated["must_include"].extend(must_include)
                elif isinstance(must_include, str):
                    aggregated["must_include"].append(must_include)
                level_note = structure.get("level_hint")
                if isinstance(level_note, str):
                    aggregated["notes"].append(level_note)
            requirements = entry.get("requirements")
            if isinstance(requirements, list):
                aggregated["additional_requirements"].extend(requirements)
            elif isinstance(requirements, str):
                aggregated["additional_requirements"].append(requirements)
            hint = entry.get("response_level_hint")
            if hint and not aggregated["response_level_hint"]:
                aggregated["response_level_hint"] = str(hint)
        return aggregated

    def _merge_style_requirements(
        self,
        base: Dict[str, Any],
        style_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        def ensure_list(value: Any) -> list:
            if value is None:
                return []
            if isinstance(value, list):
                return value
            return [value]

        merged: Dict[str, Any] = {}
        if isinstance(base, dict):
            merged.update(base)
        templates = style_data.get("templates") or []
        if templates:
            existing = ensure_list(merged.get("style_templates"))
            merged["style_templates"] = existing
            for template in templates:
                if template not in existing:
                    existing.append(template)
        must_include = style_data.get("must_include") or []
        if must_include:
            existing = ensure_list(merged.get("must_include"))
            merged["must_include"] = existing
            for item in must_include:
                if item not in existing:
                    existing.append(item)
        additional = style_data.get("additional_requirements") or []
        if additional:
            existing = ensure_list(merged.get("style_requirements"))
            merged["style_requirements"] = existing
            for item in additional:
                if item not in existing:
                    existing.append(item)
        notes = style_data.get("notes") or []
        if notes:
            existing = ensure_list(merged.get("notes"))
            merged["notes"] = existing
            for note in notes:
                if note not in existing:
                    existing.append(note)
        return merged

    def _infer_patient_id(self) -> Optional[str]:
        for tool in self.tools:
            patient_id = getattr(tool, "patient_id", None)
            if patient_id:
                return str(patient_id)
        return None

    def _infer_db_path(self) -> Optional[str]:
        for tool in self.tools:
            db_path = getattr(tool, "db_path", None)
            if db_path:
                return str(db_path)
        return None

    def _stringify_plan(self, plan: Any) -> str:
        if plan is None:
            return ""
        if isinstance(plan, str):
            return plan
        if isinstance(plan, (list, dict)):
            try:
                return json.dumps(plan, ensure_ascii=False, indent=2)
            except TypeError:
                return str(plan)
        return str(plan)

    def _normalize_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            parts = [self._normalize_text(item) for item in value if item is not None]
            return "\n".join(part for part in parts if part)
        if isinstance(value, dict):
            try:
                return json.dumps(value, ensure_ascii=False)
            except TypeError:
                return str(value)
        return str(value)

    def _heuristic_style_suggestions(self, question: str) -> List[str]:
        """Lightweight fallback mapping from question text to style skills."""
        q = (question or "").lower()
        hints: List[str] = []
        if any(tok in q for tok in ["aktuell", "derzeit", "momentan", "stichtag"]):
            hints.append("style.question_types.current_status")
        if any(tok in q for tok in ["jemals", "bisher", "schon einmal", "ever"]):
            hints.append("style.question_types.ever_status")
        if any(tok in q for tok in ["geeignet", "eligibility", "einschluss", "ausschluss", "kriterien", "hct-ci", "score"]):
            hints.append("style.question_types.eligibility")
        if any(tok in q for tok in ["response", "remission", "progression", "beste response", "best response"]):
            hints.append("style.question_types.response_assessment")
        if any(tok in q for tok in ["trend", "verlauf", "anstieg", "abfall"]):
            hints.append("style.question_types.lab_trend")
        if any(tok in q for tok in ["crp", "hb", "hämoglobin", "albumin", "ldh", "kreatinin", "calcium"]):
            hints.append("style.question_types.lab_snapshot")
        if any(tok in q for tok in ["therapie", "regime", "linie", "line", "behandlung", "exposition"]):
            hints.append("style.question_types.therapy_exposure")
        return hints

    def _collapse_string_concatenation(self, text: str) -> str:
        pattern = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"\s*\+\s*"([^"\\]*(?:\\.[^"\\]*)*)"')
        previous = None
        while previous != text:
            previous = text
            text = pattern.sub(lambda m: '"' + m.group(1) + m.group(2) + '"', text)
        text = text.replace('"""', '"')
        return text

    def _parse_json_value(self, value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return value
        if not value:
            return None
        if isinstance(value, str):
            stripped = self._strip_json_fence(value.strip())
            if not stripped:
                return None
            stripped = self._collapse_string_concatenation(stripped)
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return None
        return None

    def _parse_json_list(self, value: Any) -> List[str]:
        parsed = self._parse_json_value(value)
        if isinstance(parsed, list):
            result: List[str] = []
            for item in parsed:
                text = self._normalize_text(item).strip()
                if text:
                    result.append(text)
            return result
        text = self._normalize_text(value).strip()
        return [text] if text else []

    def _normalize_skill_ids(self, skill_ids: List[str]) -> List[str]:
        """Coerce model-returned skill names to bare IDs (no hints/percentages)."""
        normalized: List[str] = []
        seen: set[str] = set()
        for raw in skill_ids:
            text = str(raw or "").strip()
            if not text:
                continue
            if text.lower().startswith("skill "):
                text = text[6:].strip()
            text = text.lstrip("-•").strip()  # tolerate bullet prefixes
            # Drop any parenthetical/metrics or trailing commentary.
            text = re.split(r"[\s\[({|:,]", text, maxsplit=1)[0].strip()
            if text and text not in seen:
                normalized.append(text)
                seen.add(text)
        return normalized

    def _select_policy_skills(self, active_skills: List[str]) -> List[str]:
        """Deterministically attach policy skills based on selected styles/workflows."""
        policy_defaults = [
            "policy.temporal_authority",
            "policy.plan_vs_administered",
            "policy.contradiction_resolution",
        ]
        style_types = {s.split(".", 2)[2] for s in active_skills if s.startswith("style.question_types.")}
        workflow_types = {s.split(".", 1)[0] for s in active_skills if s.startswith("workflows.")}
        therapy_trigger = bool(
            style_types
            & {
                "current_status",
                "ever_status",
                "therapy_exposure",
                "response_assessment",
                "eligibility",
                "comparison",
                "temporal_localization",
            }
        )
        therapy_trigger = therapy_trigger or any("therapy" in s for s in workflow_types)
        if therapy_trigger:
            return policy_defaults
        return policy_defaults  # keep policy attached broadly for determinism

    def _infer_question_type_from_skills(self) -> str:
        """Infer question_type from selected style skills for policy routing."""
        for skill_id in self._active_skills:
            if skill_id.startswith("style.question_types."):
                return skill_id.rsplit(".", 1)[-1]
        return "*"

    def _parse_plan_object(self, plan_text: str) -> Dict[str, Any]:
        parsed = self._parse_json_value(plan_text)
        if isinstance(parsed, dict):
            return parsed
        return {}

    def _split_plan(self, plan_text: str) -> List[Dict[str, Any]]:
        if not plan_text:
            return []
        sections: List[str] = []
        current: List[str] = []
        step_pattern = re.compile(r"^\s*(?:\*{0,2}\s*)?(?:schritt|step)\s+\d+", re.IGNORECASE)
        ordered_pattern = re.compile(r"^\s*\d+[\).\s-]")
        for line in plan_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if current and (ordered_pattern.match(stripped) or step_pattern.match(stripped)):
                sections.append(" ".join(current))
                current = [stripped]
            else:
                current.append(stripped)
        if current:
            sections.append(" ".join(current))
        return [
            {"step_number": idx + 1, "objective": section, "tool_name": "", "arguments": {}}
            for idx, section in enumerate(sections)
        ]

    def _extract_plan_steps(self, plan: Any) -> List[Dict[str, Any]]:
        if plan is None:
            return []
        if isinstance(plan, str):
            plan = plan.strip()
            if not plan:
                return []
            parsed = self._try_parse_json(plan)
            if parsed is None:
                return self._split_plan(plan)
            plan = parsed

        if isinstance(plan, dict):
            steps_data = plan.get("steps")
            if isinstance(steps_data, list):
                return [self._normalize_plan_step(entry, idx + 1) for idx, entry in enumerate(steps_data)]
            return []
        if isinstance(plan, list):
            return [self._normalize_plan_step(entry, idx + 1) for idx, entry in enumerate(plan)]
        return []

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log_context_metrics(self, nodes: List[Dict[str, Any]]) -> None:
        if not nodes:
            logger.debug("Context metrics: nodes=0, chars=0, approx_tokens=0")
            return

        snippets: List[str] = []
        for node in nodes:
            snippet = node.get("snippet") or node.get("text") or ""
            snippet = re.sub(r"\s+", " ", snippet).strip()
            snippets.append(snippet)

        total_chars = sum(len(snippet) for snippet in snippets)
        approx_tokens = total_chars // 4
        logger.debug(
            "Context metrics: nodes=%d, chars=%d, approx_tokens=%d",
            len(nodes),
            total_chars,
            approx_tokens,
        )

    def _normalize_plan_step(self, entry: Any, index: int) -> Dict[str, Any]:
        if isinstance(entry, dict):
            step = dict(entry)
        elif isinstance(entry, str):
            step = {"objective": entry}
        else:
            step = {"objective": str(entry)}

        step_number = step.get("step_number") or step.get("id") or index
        step["step_number"] = step_number

        tool_name = step.get("tool_name") or step.get("tool")
        if tool_name:
            step["tool_name"] = self._normalize_text(tool_name).strip()
        else:
            step.setdefault("tool_name", "")

        arguments = step.get("arguments")
        if isinstance(arguments, str):
            parsed_args = self._parse_json_value(arguments)
            if isinstance(parsed_args, dict):
                step["arguments"] = parsed_args
            else:
                step["arguments"] = {}
        elif isinstance(arguments, dict):
            step["arguments"] = arguments
        else:
            step["arguments"] = {}

        step.setdefault("objective", step.get("description", ""))
        step.setdefault("evidence_required", step.get("evidence_required", []))
        step.setdefault("stop_if", step.get("stop_if") or "")

        return step

    def _try_parse_json(self, text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            stripped = self._strip_json_fence(text)
            if stripped and stripped != text:
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    return None
        return None

    def _parse_arguments(self, raw_arguments: Any) -> Dict[str, Any]:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if raw_arguments is None:
            return {}
        if isinstance(raw_arguments, str):
            text = raw_arguments.strip()
            if not text:
                return {}
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                stripped = self._strip_json_fence(text)
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    logger.debug("Argument parsing failed: %s", raw_arguments)
        return {}

    @staticmethod
    def _strip_json_fence(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            stripped = stripped[3:-3].strip()
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()
        return stripped

    def _deduplicate_context(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: Dict[Any, Dict[str, Any]] = {}
        for node in nodes:
            key = node.get("section_id") or (node.get("report_id"), node.get("section_name"))
            if key and key not in seen:
                seen[key] = node
        return list(seen.values())

    def _assign_citation_ids(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        assigned: List[Dict[str, Any]] = []
        for idx, node in enumerate(nodes, start=1):
            existing = node.get("citation_id")
            if existing:
                node.setdefault("citation_alias", existing)
            node["citation_id"] = f"ctx:{idx:03d}"
            assigned.append(node)
        return assigned

    def _build_citations_metadata(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        citations: List[Dict[str, Any]] = []
        for node in nodes:
            citation_id = node.get("citation_id")
            if not citation_id:
                continue
            label = node.get("section_name") or node.get("report_type") or node.get("test") or "Source"
            snippet = node.get("snippet") or node.get("text") or ""
            snippet = re.sub(r"\s+", " ", snippet).strip()
            aliases = []
            alias_value = node.get("citation_alias")
            if alias_value:
                aliases.append(str(alias_value))
            date_value = node.get("report_date") or node.get("date")
            if date_value:
                aliases.append(f"report:{date_value}")
            citations.append(
                {
                    "id": citation_id,
                    "label": label,
                    "type": node.get("report_type") or ("lab" if node.get("test") else "context"),
                    "date": node.get("report_date") or node.get("date"),
                    "snippet": snippet,
                    "aliases": aliases,
                }
            )
        return citations

    def _ensure_citation_entries(self, text: str, citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not text:
            return citations
        valid_ids = {entry.get("id") for entry in citations if entry.get("id")}
        augmented = list(citations)
        for cid in re.findall(r"\[([^\[\]]+)\]", text):
            if not cid:
                continue
            if cid not in valid_ids:
                augmented.append(
                    {
                        "id": cid,
                        "label": cid,
                        "type": "external",
                        "date": None,
                        "snippet": "Quelle nicht im Kontext verfügbar.",
                    }
                )
                valid_ids.add(cid)
        return augmented

    def _normalise_citation_ids(
        self,
        text: str,
        citations: List[Dict[str, Any]],
        hallucinated: Set[str],
    ) -> str:
        if not text:
            return text
        # First, wrap bare citation tokens (e.g., report:..., lab:...) in brackets if missing.
        def _wrap_token(match: re.Match[str]) -> str:
            token = match.group(0)
            # skip already wrapped ones
            return f"[{token}]"

        text = re.sub(r"(?<!\[)(?:report|lab|ctx):[^\s,\];]+", _wrap_token, text)
        valid_ids = {entry.get("id") for entry in citations if entry.get("id")}
        alias_map: Dict[str, str] = {}
        for cid in valid_ids:
            if cid and cid.startswith("report:"):
                parts = cid.split(":")
                if len(parts) >= 3:
                    alias = ":".join(parts[:2])
                    alias_map.setdefault(alias, cid)
                    date_part = parts[-1]
                    if date_part:
                        alias_map.setdefault(f"report:{date_part}", cid)
        for entry in citations:
            canonical = entry.get("id")
            if not canonical:
                continue
            for alias in entry.get("aliases") or []:
                alias_map.setdefault(alias, canonical)
        pattern = re.compile(r"\[([^\[\]]+)\]")

        def _replace(match: re.Match[str]) -> str:
            cid = match.group(1)
            if cid in valid_ids:
                return f"[{cid}]"
            canonical = alias_map.get(cid)
            if canonical:
                return f"[{canonical}]"
            logger.warning("Final answer cited unknown source: %s", cid)
            hallucinated.add(cid)
            return "[unknown]"

        return pattern.sub(_replace, text)

    def _replace_numeric_citations(self, text: str, nodes: List[Dict[str, Any]]) -> Tuple[str, Set[str]]:
        if not text:
            return text, set()
        numeric_map: Dict[str, str] = {}
        alias_map: Dict[str, str] = {}
        assign_numeric = False
        for idx, node in enumerate(nodes, start=1):
            cid = node.get("citation_id")
            if cid:
                numeric_map[str(idx)] = cid
                alias_map[cid] = f"ctx:{idx:03d}"
                if cid.startswith("ctx:"):
                    assign_numeric = True
        if assign_numeric:
            for idx, node in enumerate(nodes, start=1):
                node["citation_id"] = f"ctx:{idx:03d}"
        if not numeric_map:
            return text, set()
        hallucinated: Set[str] = set()

        def _convert(match: re.Match[str]) -> str:
            num = match.group(1)
            canonical = numeric_map.get(num)
            if canonical:
                return f"[{canonical}]"
            hallucinated.add(num)
            return "[unknown]"

        new_text = re.sub(r"\[(\d+)\]", _convert, text)
        return new_text, hallucinated

    @staticmethod
    def _normalise_citation_brackets(text: str) -> str:
        if not text:
            return text
        normalized = text.replace("【", "[").replace("】", "]").replace("〖", "[").replace("〗", "]")
        pieces: List[str] = []
        idx = 0
        length = len(normalized)
        while idx < length:
            if normalized.startswith("[", idx):
                end = DSPyAgentBase._find_matching_bracket(normalized, idx)
                if end == -1:
                    pieces.append(normalized[idx:])
                    break
                block = normalized[idx : end + 1]
                ids = re.findall(r"ctx:\w+", block)
                if len(ids) > 1:
                    pieces.append("".join(f"[{cid}]" for cid in ids))
                else:
                    pieces.append(block)
                idx = end + 1
            else:
                pieces.append(normalized[idx])
                idx += 1
        return "".join(pieces)

    @staticmethod
    def _find_matching_bracket(text: str, start_index: int) -> int:
        """Return the index of the matching closing bracket for text[start_index] or -1."""
        depth = 0
        for offset in range(start_index, len(text)):
            char = text[offset]
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    return offset
        return -1

    def _validate_citations(self, text: str, citations: List[Dict[str, Any]]) -> None:
        if not text or not citations:
            return
        valid_ids = {entry.get("id") for entry in citations if entry.get("id")}
        if not valid_ids:
            return
        found = set(re.findall(r"\[([^\[\]]+)\]", text))
        invalid = [cid for cid in found if cid not in valid_ids]
        if invalid:
            logger.warning("Summary cited unknown sources: %s", invalid)

    def _format_tool_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        for key, value in filters.items():
            if value in (None, "", [], {}, ()):
                continue
            cleaned[key] = value
        return cleaned

    def _parse_json_response(self, payload: Any) -> Dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if payload is None:
            return {}
        text = payload if isinstance(payload, str) else str(payload)
        text = text.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            stripped = self._strip_json_fence(text)
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                logger.debug("Failed to parse tool JSON payload: %s", text)
        return {}

    def _summarise_nodes(self, nodes: List[Dict[str, Any]], limit: int = 3, max_chars: int = 180) -> str:
        if not nodes:
            return "No relevant report excerpts were found."
        summaries: List[str] = []
        for node in nodes[:limit]:
            section = node.get("section_name") or node.get("report_type") or "Abschnitt"
            snippet = node.get("snippet") or node.get("text") or ""
            snippet = re.sub(r"\s+", " ", snippet).strip()
            if len(snippet) > max_chars:
                snippet = snippet[: max_chars - 3].rstrip() + "..."
            summaries.append(f"{section}: {snippet}")
        return "\n".join(summaries)

    def _summarise_tool_response(self, output: ToolOutput, max_chars: int = 200) -> str:
        if getattr(output, "is_error", False):
            return str(output.content or "Werkzeug meldete einen Fehler.")
        payload = getattr(output, "raw_output", None)
        text = ""
        if isinstance(payload, dict):
            summary = payload.get("summary")
            if isinstance(summary, str):
                text = summary
            else:
                response = payload.get("response")
                if isinstance(response, str):
                    text = response
                else:
                    text = payload.get("content") or ""
            if not text:
                warnings = payload.get("warnings")
                if isinstance(warnings, list) and warnings:
                    text = "; ".join(str(item).strip() for item in warnings if item)
                else:
                    resolution = payload.get("resolution")
                    if isinstance(resolution, dict):
                        markers = resolution.get("markers")
                        if isinstance(markers, list) and markers:
                            parts: List[str] = []
                            for marker in markers[:5]:
                                name = marker.get("marker") if isinstance(marker, dict) else ""
                                status = marker.get("status") if isinstance(marker, dict) else ""
                                if name or status:
                                    parts.append(f"{name or 'marker'}: {status or 'status unknown'}")
                            if parts:
                                text = "Resolution: " + "; ".join(parts)
                        elif resolution.get("status"):
                            reason = resolution.get("reason") or ""
                            text = f"Resolution: {resolution.get('status')}" + (f" ({reason})" if reason else "")
        else:
            text = output.content or ""
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return "Werkzeug lieferte keine textuelle Antwort."
        if len(text) > max_chars:
            return text[: max_chars - 3].rstrip() + "..."
        return text

    def _safe_json_dumps(self, payload: Any) -> str:
        if payload in (None, "", []):
            return ""
        try:
            return json.dumps(payload, ensure_ascii=False)
        except TypeError:
            return ""

    def _estimate_context_tokens(self, nodes: List[Dict[str, Any]]) -> int:
        if not nodes:
            return 0
        total_chars = 0
        for node in nodes:
            for key in ("snippet", "text", "content"):
                value = node.get(key)
                if isinstance(value, str):
                    total_chars += len(value)
        # Rough heuristic: 4 chars ≈ 1 token
        return max(0, total_chars // 4)

    # ------------------------------------------------------------------
    # Tool plumbing
    # ------------------------------------------------------------------
    def _get_tool(self, name: str) -> BaseTool:
        for tool in self.tools:
            if tool.metadata.name == name:
                return tool
        raise ValueError(f"Tool '{name}' ist nicht registriert.")

    def _safe_tool_call(self, tool: BaseTool, arguments: Dict[str, Any]) -> ToolOutput:
        try:
            return tool(**arguments)
        except Exception as exc:  # pylint: disable=broad-except
            # Log errors at DEBUG to avoid noisy stderr during batch runs.
            logger.debug("Tool %s failed: %s", tool.metadata.name, exc, exc_info=exc)
            return ToolOutput(
                tool_name=tool.metadata.name,
                content=f"Tool error: {exc}",
                raw_input={"kwargs": arguments},
                raw_output={"error": str(exc)},
                is_error=True,
            )

    def _invoke_tool_for_chat(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        try:
            tool = self._get_tool(tool_name)
        except ValueError as exc:
            error_text = str(exc)
            payload = {"error": error_text}
            content = self._safe_json_dumps(payload) or error_text
            self._append_tool_message(tool_name, payload, content, arguments=arguments)
            return content

        normalized_arguments = self._normalize_tool_arguments(tool, arguments)
        output = self._safe_tool_call(tool, normalized_arguments)
        payload = getattr(output, "raw_output", None)
        if not isinstance(payload, dict):
            payload = self._parse_json_response(getattr(output, "content", None))
        payload = payload or {}
        if getattr(output, "is_error", False):
            tool_message_content = output.content or self._safe_json_dumps(payload) or ""
        else:
            tool_message_content = self._safe_json_dumps(payload) if payload else (output.content or "")
        self._append_tool_message(tool.metadata.name, payload, tool_message_content, arguments=normalized_arguments)
        return tool_message_content

    def _append_user_message(self, content: str) -> None:
        self._history.append(ChatMessage(role=MessageRole.USER, content=content))

    def _append_assistant_message(self, content: str, **kwargs: Any) -> None:
        self._history.append(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                additional_kwargs=kwargs if kwargs else {},
            )
        )

    def _append_tool_message(
        self,
        name: str,
        response: Dict[str, Any],
        content: str,
        *,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> None:
        additional = {
            "name": name,
            "response": response,
        }
        if arguments is not None:
            additional["arguments"] = arguments
        self._history.append(
            ChatMessage(
                role=MessageRole.TOOL,
                content=content,
                additional_kwargs=additional,
            )
        )

    # ------------------------------------------------------------------
    # Tool argument normalization helpers
    # ------------------------------------------------------------------
    LAB_DELIMITER_PATTERN = re.compile(r"[;,]+")

    def _normalize_tool_arguments(self, tool: BaseTool, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if tool.metadata.name != "retrieve_lab_values":
            return arguments
        normalized = dict(arguments)
        lab_query = normalized.get("lab_query")
        if isinstance(lab_query, str):
            split_query = self._split_marker_argument(lab_query)
            if len(split_query) > 1:
                normalized["lab_query"] = "; ".join(split_query)
        return normalized

    def _split_marker_argument(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in self.LAB_DELIMITER_PATTERN.split(value) if item.strip()]
        parts: List[str] = []
        if isinstance(value, (list, tuple, set)):
            iterable = value
        else:
            iterable = [value]
        for item in iterable:
            if item is None:
                continue
            if isinstance(item, str):
                parts.extend([chunk.strip() for chunk in self.LAB_DELIMITER_PATTERN.split(item) if chunk.strip()])
            else:
                parts.append(str(item).strip())
        return [p for p in parts if p]

    # ------------------------------------------------------------------
    # LLM bridge
    # ------------------------------------------------------------------
    @abstractmethod
    def _complete(self, prompt: str) -> str:
        """Return a completion for the given prompt."""
