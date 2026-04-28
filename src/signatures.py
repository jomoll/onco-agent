"""DSPy signature definitions for the lightweight agents."""

from __future__ import annotations

from typing import ClassVar

import dspy


CANONICAL_TOOL_SPEC = (
    "Tools (canonical spec):\n"
    "- retrieve_reports(query, time_scope, report_type?, patient_id?, date_exact?, date_start?, date_end?, top_k?) "
    "time_scope must be one of 'all', 'latest', 'date', or 'range'; use date_exact for 'date' and date_start/date_end for 'range'. "
    "Queries must be in German and use clinical keywords (disease, therapy, timeframe).\n"
    "- retrieve_lab_values(lab_query, time_scope, patient_id?, date_exact?, date_start?, date_end?, window_policy?, "
    "window_days?, max_results_per_lab?) time_scope is required and must be one of 'all', 'latest', 'date', or 'range'; "
    "use date_exact for 'date' and date_start/date_end for 'range' (use window_policy + window_days for a date window). "
    "lab_query must be a JSON array of marker strings in German (e.g., [\"IgG\", \"IgA\", \"FLC kappa\"])."
)


class AssessAndSelectSkills(dspy.Signature):
    """Assess the question and select the most relevant skills."""

    question = dspy.InputField(desc="Original clinical question from the user (German or English).")
    patient_context = dspy.InputField(desc="Known patient context. Leave empty if none was supplied.")
    answer_schema = dspy.InputField(
        desc="Expected answer schema when explicitly provided by the caller. Empty string if not available."
    )
    default_filters = dspy.InputField(
        desc=(
            "Default filters (report_type, top_k, time_scope, etc.) that define the baseline scope. "
            "You may adjust them intentionally when evidence is missing, but never do so silently."
        )
    )
    allowed_tools = dspy.InputField(
        desc=(
            "List of allowed tools with argument hints. Use only these canonical names.\n"
            + CANONICAL_TOOL_SPEC
        )
    )
    skill_summaries = dspy.InputField(
        desc=(
            "Bullet list describing available skills/instruction packs. "
            "Pick only the relevant IDs for this conversation."
        )
    )
    lab_key_catalog = dspy.InputField(
        desc="JSON array of canonical lab markers available for the patient; copy these names verbatim when planning lab queries."
    )

    analysis = dspy.OutputField(
        desc="Short (≤3 sentences) medical analysis highlighting key hypotheses and clinical intent."
    )
    required_information = dspy.OutputField(
        desc="JSON array of required information points the agent must gather. Format: [\"...\"]"
    )
    missing_information = dspy.OutputField(
        desc="JSON array of currently missing information that might block the answer. Format: [\"...\"]"
    )
    selected_skills = dspy.OutputField(
        desc="JSON array of skill IDs (from skill_summaries) to activate for this run."
    )
    response_level = dspy.OutputField(
        desc=(
            "Question difficulty level inferred from clinician templates: "
            "\"1\" (status/ever/never), \"2\" (dose/response/timeline), or \"3\" (eligibility/next-step). Include rationale."
        )
    )
    response_requirements = dspy.OutputField(
        desc=(
            "JSON object describing the expected answer format: keys such as \"format\", \"must_include\", "
            "\"citations_required\", and domain-specific notes. Language defaults to German."
        )
    )


class BuildToolPlan(dspy.Signature):
    """Build a deterministic tool plan using the selected skills."""

    question = dspy.InputField(desc="Original clinical question from the user (German or English).")
    patient_context = dspy.InputField(desc="Known patient context. Leave empty if none was supplied.")
    answer_schema = dspy.InputField(
        desc="Expected answer schema when explicitly provided by the caller. Empty string if not available."
    )
    default_filters = dspy.InputField(
        desc=(
            "Default filters (report_type, top_k, time_scope, etc.) that define the baseline scope. "
            "You may adjust them intentionally when evidence is missing, but never do so silently."
        )
    )
    allowed_tools = dspy.InputField(
        desc=(
            "List of allowed tools with argument hints. Use only these canonical names.\n"
            + CANONICAL_TOOL_SPEC
        )
    )
    skills_context = dspy.InputField(
        desc="Instruction snippets from selected skills. Use these to plan steps and stop conditions."
    )
    lab_key_catalog = dspy.InputField(
        desc="JSON array of canonical lab markers available for the patient; copy these names verbatim when planning lab queries."
    )

    tool_plan = dspy.OutputField(
        desc=(
            "Deterministic plan as JSON. Structure: {\"steps\": ["
            "{\"step_number\": 1, \"objective\": \"...\", \"tool_name\": \"...\", "
            "\"arguments\": { ... }, \"evidence_required\": [\"...\"], "
            "\"stop_if\": \"condition\"}], \"global_stop_conditions\": [\"...\"]}. "
            "Each step must correspond to a concrete tool invocation addressing part of the question."
        )
    )


class ToolExecutionStep(dspy.Signature):
    """Select the next tool step or finish early."""

    question = dspy.InputField(desc="Original user question.")
    patient_context = dspy.InputField(desc="Patient context, if available.")
    answer_schema = dspy.InputField(
        desc="Expected answer schema when explicitly provided by the caller. Empty string if not available."
    )
    skills_context = dspy.InputField(
        desc="Instruction snippets from all active skills. Follow these operational guidelines."
    )
    lab_key_catalog = dspy.InputField(
        desc="JSON array of canonical lab keys available for the patient. Use these names exactly when calling retrieve_lab_values."
    )
    plan_chunk = dspy.InputField(desc="JSON object for the next planned step.")
    previous_results = dspy.InputField(desc="Summary of prior tool outputs/evidence.")
    evidence_required = dspy.InputField(
        desc="JSON array of evidence requirements for the current step. Format: [\"...\"]"
    )
    required_information = dspy.InputField(
        desc="JSON array of required information points the agent must gather. Format: [\"...\"]"
    )
    missing_information = dspy.InputField(
        desc="JSON array of currently missing information that might block the answer. Format: [\"...\"]"
    )
    allowed_tools = dspy.InputField(
        desc="Allowed tools. On error, correct parameters and retry; do not skip.\n" + CANONICAL_TOOL_SPEC
    )
    global_stop_conditions = dspy.InputField(desc="JSON list of stop conditions that allow early termination.")

    action = dspy.OutputField(
        desc=(
            "'call_tool', 'skip', or 'finish'. "
            "IMPORTANT: After EACH tool execution, evaluate if you have sufficient evidence to answer the question. "
            "If current context contains the required information, choose 'finish' immediately. "
            "Only continue if critical evidence is definitively missing—avoid redundant queries that add only marginal value. "
            "Finish when: (1) evidence_required items are satisfied, (2) stop_conditions are met, or (3) you have enough context to answer."
        )
    )
    tool_name = dspy.OutputField(desc="Tool name when action is 'call_tool'.")
    arguments = dspy.OutputField(
        desc="JSON string containing the arguments for the tool call. "
        "IMPORTANT: The 'query' field must contain only clinical search terms (drug names, "
        "lab names, dates, clinical concepts). Never include answer format tokens "
        "(Status=, Answer:, DD.MM.YYYY template, Dokumentiert, ||, etc.) in queries."
    )
    rationale = dspy.OutputField(desc="Brief explanation of the action taken.")



class DraftFinalAnswer(dspy.Signature):
    """Write a concise, citation-backed answer in the language specified by answer_language."""

    instructions: ClassVar[str] = (
        "Determine the expected answer style from answer_schema/response_level/response_requirements/style_context, then draft the final answer accordingly. "
        "Produce the final answer in the language specified by `answer_language`. This overrides any style-skill language instructions. "
        "Translate all user-facing terms into the target language, including Ja/Nein/Unklar and status labels (Dokumentiert, Nie verabreicht, Nicht dokumentiert). "
        "Keep only structural separators (||, |, ;, =), dates (DD.MM.YYYY), drug names, and clinical codes (CR/VGPR/PR/SD/PD) unchanged. "
        "Use the provided templates/requirements; if style_templates are present, follow them exactly (two lines: 'Answer: ...' and 'Reasoning: ...'). "
        "Do not apply hardcoded level-specific formats; rely on the supplied style templates and requirements. "
        "The Answer line must contain only the schema option(s) (e.g., Ja/Nein/Unklar/Nicht dokumentiert or the exact choice text); do NOT add citations, evidence, or extra prose in the Answer line. "
        "Citations belong only in the Reasoning using provided [ctx:ID]s; do not add a separate Evidence list. "
        "Every factual statement must cite an available ID using [citation_id]; never invent IDs and cite every listed criterion. "
        "Each citation ID must be enclosed in its own bracket pair, e.g., [ctx:001][ctx:002]—never combine multiple IDs within a single bracket or nest brackets. "
        "The Answer line must commit to the single most likely schema option. Express all uncertainty, caveats, and missing-data notes in the Reasoning only. 'Unklar' is a last resort for genuine contradictions, not for incomplete evidence. "
        "If evidence is missing, state the gap and cite the search scope as a compact range, e.g. '[ctx:001]–[ctx:020]', not by listing every ID individually. "
        "If no evidence at all exists, answer \"Keine relevanten Informationen in den bereitgestellten Quellen.\" "
        "You MUST respect policy_result: if policy_result indicates abstain or conflict, reflect that faithfully in the answer. "
        "Return exactly {\"final_answer\": \"<answer>\"}."
    )

    question = dspy.InputField(desc="Question to be answered.")
    answer_schema = dspy.InputField(desc="Expected answer schema when explicitly provided by the caller. Empty string if not available.")
    answer_language = dspy.InputField(desc="Language for the final answer (e.g. 'Deutsch', 'English', 'Français').")
    patient_context = dspy.InputField(desc="Patient context, if available.")
    plan_overview = dspy.InputField(desc="Brief recap of executed steps and analysis.")
    applied_filters = dspy.InputField(desc="Description of applied filters/tools.")
    context_snippets = dspy.InputField(desc="Numbered evidence snippets with citation IDs.")
    outstanding_information = dspy.InputField(desc="JSON array of still-missing information.")
    citations = dspy.InputField(desc="JSON array of citation metadata (id, label, type, date). Use these IDs verbatim in citations.")
    response_level = dspy.InputField(desc="Level value from assessment (1/2/3).")
    response_requirements = dspy.InputField(desc="JSON object describing format/must-include items.")
    style_context = dspy.InputField(desc="Additional formatting instructions derived from style skills.")
    policy_result = dspy.InputField(desc="JSON object with policy resolution (claim/abstain/conflict) and supporting evidence ids.")
    policy_trace = dspy.InputField(desc="JSON object with policy trace/rules fired.")
    final_answer = dspy.OutputField(
        desc=(
            "MUST be exactly two lines:\n"
            "  Answer: <schema-aligned value>\n"
            "  Reasoning: <1-2 sentences with [ctx:ID] citations>\n"
            "Example: \"Answer: Ja\\nReasoning: Im Arztbrief vom 23.07.2021 wird eine Daratumumab-Therapie dokumentiert [ctx:001].\"\n"
            "The Answer line must contain ONLY the schema option(s) — no citations, no extra prose. "
            "Citations belong only in Reasoning using [ctx:ID] format. "
            "Follow response_level/requirements/style_context for the specific schema. "
            "Produce the answer in the language specified by answer_language. Translate all user-facing terms (Ja/Nein/Unklar, Dokumentiert, Nie verabreicht, Nicht dokumentiert, etc.). "
            "Keep only structural separators (||, |, ;, =), dates (DD.MM.YYYY), drug names, and clinical codes (CR/VGPR/PR/SD/PD) unchanged."
        )
    )
