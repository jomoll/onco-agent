"""Integration checks for the lightweight agents talking to live endpoints."""

from __future__ import annotations

import json
import logging
import re
import os
from pathlib import Path
from typing import Any, Dict

import pytest
from dotenv import load_dotenv
from llama_index.core.base.llms.types import MessageRole

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("src.agent_gemini").setLevel(logging.DEBUG)
logging.getLogger("src.agent_vllm").setLevel(logging.DEBUG)

try:  # pragma: no cover - optional heavy dependencies
    from src.agent_vllm import VLLMChatAgent, VLLMConfig
    from src.agent_gemini import GeminiChatAgent, GeminiConfig
except ImportError as exc:  # pragma: no cover
    pytest.skip(
        f"Integration tests require optional agent dependencies: {exc}",
        allow_module_level=True,
    )

from src.agent_tools import load_default_tools

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DB = PROJECT_ROOT / "src" / "database" / "synthetic.sqlite"

# Synthetic patient IDs in the test DB
PATIENT_BERGMANN = "patient_001"   # IgG kappa, ISS II, standard risk, ASCT + MRD-neg CR
PATIENT_HOFFMANN = "patient_002"   # IgG lambda, ISS I, standard risk, sCR/MRD-neg, maintenance
PATIENT_FISCHER  = "patient_003"   # IgA lambda, ISS III, t(4;14)+1q21+, refractory, BSC


@pytest.fixture(scope="module")
def tools_bergmann():
    return load_default_tools(patient_id=PATIENT_BERGMANN, db_path=SYNTHETIC_DB)


@pytest.fixture(scope="module")
def tools_hoffmann():
    return load_default_tools(patient_id=PATIENT_HOFFMANN, db_path=SYNTHETIC_DB)


@pytest.fixture(scope="module")
def tools_fischer():
    return load_default_tools(patient_id=PATIENT_FISCHER, db_path=SYNTHETIC_DB)


_CTX_BERGMANN = (
    "Patient: Thomas Bergmann, *1958. Diagnose: Multiples Myelom IgG kappa, ISS II (2021). "
    "Therapie: VRd-Induktion → ASCT → Lenalidomid-Erhaltung. MRD-negativer Status dokumentiert."
)
_CTX_HOFFMANN = (
    "Patientin: Klara Hoffmann, *1970. Diagnose: Multiples Myelom IgG lambda, ISS I, R-ISS I (02/2024). "
    "Standard-Zytogenetik. Therapie: VRd×6 → HD-Mel 200 + ASCT (08/2024) → sCR/MRD-negativ (10/2024) → "
    "Lenalidomid-Erhaltung (seit 11/2024)."
)
_CTX_FISCHER = (
    "Patient: Gerhard Fischer, *1951. Diagnose: Multiples Myelom IgA lambda, ISS III, R-ISS III (09/2016). "
    "Hochrisiko-Zytogenetik: t(4;14), 1q21+, später del(17p). Mehrfach vorbehandelt (VCD, ASCT, KRd, "
    "PACE, Dara-Pd, Teclistamab). Seit 08/2022 BSC/Palliativversorgung."
)


def _scenarios_bergmann() -> list[Dict[str, Any]]:
    return [
        {
            "label": "iss_stage",
            "question": "What is the ISS stage at first diagnosis and which lab values justify it?",
            "patient_context": _CTX_BERGMANN,
            "kwargs": {},
            "min_context": 1,
        },
        {
            "label": "therapy_lines",
            "question": "Which therapy lines has the patient received, and what response was achieved after each?",
            "patient_context": _CTX_BERGMANN,
            "kwargs": {},
            "min_context": 1,
        },
        {
            "label": "mrd_status",
            "question": "What is the current MRD status and when was it last assessed?",
            "patient_context": _CTX_BERGMANN,
            "kwargs": {},
            "min_context": 1,
        },
    ]


def _scenarios_hoffmann() -> list[Dict[str, Any]]:
    return [
        {
            "label": "mrd_maintenance",
            "question": "Is the patient MRD-negative, and what maintenance therapy is currently ongoing?",
            "patient_context": _CTX_HOFFMANN,
            "kwargs": {},
            "min_context": 1,
        },
        {
            "label": "stem_cell_harvest",
            "question": "How many CD34+ cells per kg were collected during stem cell mobilization?",
            "patient_context": _CTX_HOFFMANN,
            "kwargs": {},
            "min_context": 1,
        },
    ]


def _scenarios_fischer() -> list[Dict[str, Any]]:
    return [
        {
            "label": "cytogenetics",
            "question": "What high-risk cytogenetic abnormalities were detected at diagnosis and during follow-up?",
            "patient_context": _CTX_FISCHER,
            "kwargs": {},
            "min_context": 1,
        },
        {
            "label": "therapy_lines_refractory",
            "question": "How many therapy lines has the patient received and why was the last active therapy discontinued?",
            "patient_context": _CTX_FISCHER,
            "kwargs": {},
            "min_context": 1,
        },
    ]


def _print_chat_history(agent, label: str) -> None:
    history = getattr(agent, "history", None)
    if history is None:
        history = getattr(agent, "_chat_history", [])
    print(f"\n[Integration:{label}] Conversation transcript:")
    for idx, message in enumerate(history, start=1):
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        additional = getattr(message, "additional_kwargs", None)
        if role == MessageRole.TOOL:
            display = _summarise_tool_message(content, additional)
            print(f"  [{idx}] role={role} content={display!r}")
        else:
            print(f"  [{idx}] role={role} content={content!r}")


def _summarise_tool_message(content: str | None, additional: dict | None) -> str:
    summary = ""
    try:
        parsed = json.loads(content or "{}")
    except Exception:
        parsed = {}
    response_text = ""
    if isinstance(parsed, dict):
        response = parsed.get("summary") or parsed.get("response")
        if isinstance(response, str):
            response_text = response
        context_nodes = parsed.get("context_nodes")
        if isinstance(context_nodes, list):
            summary += f" [{len(context_nodes)} nodes]"

    response_text = re.sub(r"\s+", " ", response_text).strip()
    if response_text:
        summary = response_text[:120] + ("..." if len(response_text) > 120 else "") + summary
    elif content:
        summary = re.sub(r"\s+", " ", content).strip()
        summary = summary[:120] + ("..." if len(summary) > 120 else "")

    if additional and isinstance(additional, dict):
        name = additional.get("name")
        args = additional.get("arguments")
        snippet = f"{name or 'tool'}"
        if isinstance(args, dict):
            filters = {k: v for k, v in args.items() if v not in (None, "") and k != "top_k"}
            if filters:
                snippet += f" {filters}"
        summary = f"{snippet}: {summary}"

    return summary


def _make_vllm_agent(tools):
    base_url = os.getenv("VLLM_BASE_URL")
    if not base_url:
        pytest.skip("Set VLLM_BASE_URL to run live vLLM integration test.")
    api_key = os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("VLLM_MODEL", "gpt-oss-120b")
    completion_kwargs = {}
    temperature = os.getenv("VLLM_TEMPERATURE")
    if temperature:
        completion_kwargs["temperature"] = float(temperature)
    config = VLLMConfig(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        completion_kwargs=completion_kwargs,
    )
    return VLLMChatAgent(config=config, tools=tools)


@pytest.mark.integration
def test_vllm_bergmann(tools_bergmann):
    agent = _make_vllm_agent(tools_bergmann)
    for scenario in _scenarios_bergmann():
        agent.reset()
        reply = agent.answer_with_rag(
            scenario["question"],
            patient_context=scenario["patient_context"],
            **scenario["kwargs"],
        )
        context_nodes = reply.additional_kwargs.get("context_nodes", [])
        assert len(context_nodes) >= scenario["min_context"], (
            f"[{scenario['label']}] Expected at least {scenario['min_context']} context nodes"
        )
        assert reply.content and reply.content.strip(), f"[{scenario['label']}] Answer should not be empty"
        _print_chat_history(agent, f"Bergmann/{scenario['label']}")


@pytest.mark.integration
def test_vllm_hoffmann(tools_hoffmann):
    agent = _make_vllm_agent(tools_hoffmann)
    for scenario in _scenarios_hoffmann():
        agent.reset()
        reply = agent.answer_with_rag(
            scenario["question"],
            patient_context=scenario["patient_context"],
            **scenario["kwargs"],
        )
        context_nodes = reply.additional_kwargs.get("context_nodes", [])
        assert len(context_nodes) >= scenario["min_context"], (
            f"[{scenario['label']}] Expected at least {scenario['min_context']} context nodes"
        )
        assert reply.content and reply.content.strip(), f"[{scenario['label']}] Answer should not be empty"
        _print_chat_history(agent, f"Hoffmann/{scenario['label']}")


@pytest.mark.integration
def test_vllm_fischer(tools_fischer):
    agent = _make_vllm_agent(tools_fischer)
    for scenario in _scenarios_fischer():
        agent.reset()
        reply = agent.answer_with_rag(
            scenario["question"],
            patient_context=scenario["patient_context"],
            **scenario["kwargs"],
        )
        context_nodes = reply.additional_kwargs.get("context_nodes", [])
        assert len(context_nodes) >= scenario["min_context"], (
            f"[{scenario['label']}] Expected at least {scenario['min_context']} context nodes"
        )
        assert reply.content and reply.content.strip(), f"[{scenario['label']}] Answer should not be empty"
        _print_chat_history(agent, f"Fischer/{scenario['label']}")


@pytest.mark.integration
def test_gemini_agent_live(tools_bergmann):
    project_id = os.getenv("GEMINI_PROJECT_ID")
    if not project_id:
        pytest.skip("Set GEMINI_PROJECT_ID to run live Gemini integration test.")

    location = os.getenv("GEMINI_LOCATION", "us-central1")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
    config = GeminiConfig(project_id=project_id, location=location, model=model_name)
    agent = GeminiChatAgent(config=config, tools=tools_bergmann)

    scenario = _scenarios_bergmann()[0]
    agent.reset()
    reply = agent.answer_with_rag(
        scenario["question"],
        patient_context=scenario["patient_context"],
        **scenario["kwargs"],
    )
    context_nodes = reply.additional_kwargs.get("context_nodes", [])
    assert len(context_nodes) >= scenario["min_context"], "Expected context nodes from RAG tool"
    assert reply.content and reply.content.strip(), "Answer should not be empty"
    _print_chat_history(agent, f"Gemini/Bergmann/{scenario['label']}")
