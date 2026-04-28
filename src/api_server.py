from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from datetime import date, datetime
from typing import Any, AsyncIterator, Dict, List, Optional
import sqlite3
import sys

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from pathlib import Path
# Prefer values from the repo .env (override=True) so changes are picked up on restart
# even when a shell still has older env exports lingering.
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)

logger = logging.getLogger(__name__)
from src.agent_gemini import GeminiChatAgent, GeminiConfig
from src.agent_vllm import VLLMChatAgent, VLLMConfig
from src.agent_tools import ReportsRAGTool, load_default_tools

# Ensure the project root is importable when the file is executed directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)
USER_ACTIVITY_DIR = PROJECT_ROOT / "logs"


class AgentRunStore:
    def __init__(self, max_runs: int = 50) -> None:
        self._max_runs = max_runs
        self._runs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    def _prune_locked(self) -> None:
        while len(self._runs) > self._max_runs:
            self._runs.popitem(last=False)

    def start_run(self, run_id: str) -> None:
        with self._lock:
            self._runs[run_id] = {
                "events": [],
                "created_at": time.time(),
            }
            self._prune_locked()

    def add_event(self, run_id: str, event: Dict[str, Any]) -> None:
        with self._lock:
            if run_id in self._runs:
                self._runs[run_id]["events"].append(event)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return None
            return {
                "run_id": run_id,
                "events": list(run["events"]),
                "created_at": run["created_at"],
            }

    def list_runs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {"run_id": run_id, "created_at": data["created_at"], "event_count": len(data["events"])}
                for run_id, data in reversed(self._runs.items())
            ]


run_store = AgentRunStore()


def _normalise_event(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "type": event.get("type", "unknown"),
        "timestamp": float(event.get("timestamp") or time.time()),
        "payload": event.get("payload", {}),
    }


def _prune_heavy_fields(obj: Any) -> Any:
    """
    Remove bulky fields from agent events before persisting them to the detailed log.
    Keeps debugging essentials (ids, types, timestamps, small params) while dropping
    evidence nodes, verbose skills, and long tool descriptions.
    """
    keys_to_strip = {
        "nodes",
        "skills_context",
        "skill_style_context",
        "skill_workflow_context",
        "skill_summaries",
        "tool_descriptions",
        "lab_key_catalog",
        "plan_text",
        "plan_steps",
        "allowed_tools",  # drop entire block; can be very long strings or dicts
    }
    if isinstance(obj, dict):
        pruned: Dict[str, Any] = {}
        for key, value in obj.items():
            if key in keys_to_strip:
                continue
            if key == "allowed_tools" and isinstance(value, list):
                # Keep only tool names to avoid long description blobs.
                pruned[key] = [
                    item.get("name") if isinstance(item, dict) else item for item in value
                ]
                continue
            if key in {"skills", "policy_skills"} and isinstance(value, list):
                pruned[key] = [
                    item.get("id") if isinstance(item, dict) else item for item in value
                ]
                continue
            if key == "inputs" and isinstance(value, dict):
                pruned_inputs = {
                    k: v for k, v in value.items() if k not in keys_to_strip
                }
                pruned[key] = _prune_heavy_fields(pruned_inputs)
            else:
                pruned[key] = _prune_heavy_fields(value)
        return pruned
    if isinstance(obj, list):
        return [_prune_heavy_fields(item) for item in obj]
    return obj


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    patient_id: Optional[str] = None
    history: Optional[List[ChatMessage]] = None
    model: Optional[str] = None


class ChatReply(BaseModel):
    content: str
    reply_metadata: dict = Field(default_factory=dict, alias="metadata")
    run_id: Optional[str] = None
    events: Optional[List[Dict[str, Any]]] = None


class PatientSummary(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    details: Optional[str] = None
    latest_report_date: Optional[str] = None
    earliest_report_date: Optional[str] = None
    approx_tokens: Optional[int] = None
    report_counts: Optional[Dict[str, int]] = None
    dob: Optional[str] = None
    key_lab_recency: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None


def _format_history_for_prompt(history: Optional[List[ChatMessage]], limit: int = 8) -> str:
    if not history:
        return ""
    snippets: List[str] = []
    recent = history[-limit:]
    role_map = {
        "user": "User",
        "assistant": "Assistant",
        "system": "System",
    }
    for entry in recent:
        role = role_map.get((entry.role or "").strip().lower(), "User")
        content = (entry.content or "").strip()
        if not content:
            continue
        snippets.append(f"{role}: {content}")
    return "\n".join(snippets)


_REFERENCE_DATE_PLACEHOLDER = "{datum_des_letzten_verfügbaren_Berichts}"


def _format_reference_date_for_question(reference_date: Optional[str]) -> Optional[str]:
    if not reference_date:
        return None
    text = str(reference_date).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d-%m-%Y"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.strftime("%d.%m.%Y")
        except ValueError:
            continue
    return text


def _inject_reference_date(question: str, reference_date: Optional[str]) -> str:
    if not question:
        return ""
    if _REFERENCE_DATE_PLACEHOLDER not in question:
        return question
    formatted = _format_reference_date_for_question(reference_date)
    if not formatted:
        return question
    return question.replace(_REFERENCE_DATE_PLACEHOLDER, formatted)


def _normalize_report_type_label(raw: str) -> str:
    mapping = {
        "doctor_letter": "Doctor letter",
        "arztbrief": "Doctor letter",
        "tumorboard beschluss": "Tumor board",
        "beschluss": "Tumor board",
        "tumor_board": "Tumor board",
        "cytology": "Cytology",
        "flow": "Flow cytometry",
        "flow cytometry": "Flow cytometry",
        "pathology": "Pathology",
        "pathology report": "Pathology",
        "path": "Pathology",
        "radiology": "Radiology",
        "radiology report": "Radiology",
        "rad": "Radiology",
        "cardiology": "Cardiology",
        "consult": "Consult",
        "history": "History",
        "labor": "Lab reports (accumulated)",
    }
    key = (raw or "").strip().lower()
    return mapping.get(key, raw or "Unknown")


def _log_user_activity(
    *,
    question: str,
    patient_id: Optional[str],
    model_name: str,
    metadata: Optional[Dict[str, Any]],
    answer: str,
) -> None:
    try:
        USER_ACTIVITY_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow()
        log_file = USER_ACTIVITY_DIR / f"user_activity_{timestamp.strftime('%Y-%m-%d')}.log"
        timestamp_iso = timestamp.isoformat()
        lines = [
            f"Timestamp: {timestamp_iso} UTC",
            f"Patient ID: {patient_id or '-'}",
            f"Model: {model_name}",
            f"Question: {question.strip()}",
        ]
        tool_actions = (metadata or {}).get("actions") or []
        if tool_actions:
            lines.append("Tools used:")
            for action in tool_actions:
                tool_name = action.get("tool") or "-"
                result_count = action.get("result_count")
                arguments = action.get("arguments") or {}
                entry = f"  - {tool_name}"
                if isinstance(result_count, int):
                    entry += f" (results: {result_count})"
                if arguments:
                    entry += f" | params: {json.dumps(arguments, ensure_ascii=False)}"
                lines.append(entry)
        nodes = (metadata or {}).get("context_nodes") or []
        if nodes:
            lines.append("Retrieved context:")
            for node in nodes:
                report = node.get("report_type") or node.get("report_name") or "-"
                section = node.get("section_name") or "-"
                report_id = node.get("report_id") or "-"
                report_date = node.get("report_date") or node.get("date") or "-"
                lines.append(f"  - {report} / {section} (report_id={report_id}, date={report_date})")
        final_answer = (metadata or {}).get("final_answer") or answer
        lines.append(f"Final answer: {final_answer.strip() if final_answer else '-'}")
        lines.append("-" * 60)
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except Exception:  # pragma: no cover
        logger.exception("Failed to log user activity.")


def _log_agent_monitor_event(run_id: str, event: Dict[str, Any]) -> None:
    try:
        USER_ACTIVITY_DIR.mkdir(parents=True, exist_ok=True)
        log_file = USER_ACTIVITY_DIR / "agent_detailed_log.jsonl"
        payload = _prune_heavy_fields({"run_id": run_id, **event})
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:  # pragma: no cover
        logger.exception("Failed to log agent monitor event.")


def _parse_dob(dob_str: str) -> Optional[date]:
    """Parse DD.MM.YYYY birth date."""
    try:
        parts = dob_str.strip().split(".")
        if len(parts) == 3:
            return date(int(parts[2]), int(parts[1]), int(parts[0]))
    except (ValueError, IndexError, AttributeError):
        pass
    return None


def _parse_date_ymd(date_str: str) -> Optional[date]:
    """Parse YYYY-MM-DD (or ISO datetime) to a date object."""
    if not date_str:
        return None
    try:
        return date.fromisoformat(date_str[:10])
    except ValueError:
        return None


def _age_at(dob: date, ref: date) -> int:
    years = ref.year - dob.year
    if (ref.month, ref.day) < (dob.month, dob.day):
        years -= 1
    return years


def _load_patients_from_db(db_path: str | Path | None = None) -> List[PatientSummary]:
    conn: Optional[sqlite3.Connection] = None
    resolved = str(db_path) if db_path else str(ReportsRAGTool.DB_PATH)
    try:
        conn = sqlite3.connect(resolved)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT patient_id, fullname, dob
            FROM patients
            ORDER BY patient_id
            """
        )
        patient_rows = cursor.fetchall()
        cursor.execute(
            """
            SELECT patient_id, report_id, report_date
            FROM reports
            """
        )
        report_rows = cursor.fetchall()
        cursor.execute(
            """
            SELECT patient_id, report_type, COUNT(*) as count
            FROM reports
            GROUP BY patient_id, report_type
            """
        )
        count_rows = cursor.fetchall()
        cursor.execute(
            """
            SELECT patient_id,
                   SUM(LENGTH(COALESCE(section_content, ''))) as report_chars
            FROM report_sections
            GROUP BY patient_id
            """
        )
        report_char_rows = cursor.fetchall()
        cursor.execute(
            """
            SELECT patient_id,
                   SUM(LENGTH(COALESCE(canonical_key, '')) + LENGTH(COALESCE(value, '')))
                   as lab_chars
            FROM lab_values
            GROUP BY patient_id
            """
        )
        lab_char_rows = cursor.fetchall()
        cursor.execute(
            """
            SELECT lv.patient_id, lv.canonical_key,
                   COUNT(*) AS n, MAX(lv.assessment_dt) AS last_dt
            FROM lab_values lv
            LEFT JOIN (
                SELECT patient_id, MAX(report_date) AS stichtag
                FROM reports
                GROUP BY patient_id
            ) s ON s.patient_id = lv.patient_id
            WHERE lv.canonical_key IN ('Beta2-Mikroglobulin', 'Albumin', 'LDH')
              AND (s.stichtag IS NULL OR DATE(lv.assessment_dt) <= s.stichtag)
            GROUP BY lv.patient_id, lv.canonical_key
            """
        )
        key_lab_rows = cursor.fetchall()
        summary_rows: List[tuple[Any, Any]] = []
        try:
            cursor.execute(
                """
                SELECT patient_id, payload
                FROM patient_summaries
                """
            )
            summary_rows = cursor.fetchall()
        except sqlite3.OperationalError:
            logger.warning("patient_summaries table missing; continuing without summaries.")
    except Exception as exc:
        logger.exception("Failed to load patient list from SQLite database.")
        raise RuntimeError("Patient database is unavailable; check the SQLite configuration.") from exc
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    latest_map: Dict[str, str] = {}
    earliest_map: Dict[str, str] = {}
    for patient_id, report_id, report_date in report_rows:
        if not patient_id:
            continue
        candidate = report_date or ReportsRAGTool._derive_report_date(report_id or "")
        normalized = ReportsRAGTool._normalize_report_date(candidate)
        if not normalized:
            continue
        key = str(patient_id)
        current_latest = latest_map.get(key)
        if current_latest is None or normalized > current_latest:
            latest_map[key] = normalized
        current_earliest = earliest_map.get(key)
        if current_earliest is None or normalized < current_earliest:
            earliest_map[key] = normalized
    approx_tokens_map: Dict[str, int] = {}
    for patient_id, chars in report_char_rows:
        if patient_id:
            approx_tokens_map[str(patient_id)] = approx_tokens_map.get(str(patient_id), 0) + (chars or 0)
    for patient_id, chars in lab_char_rows:
        if patient_id:
            approx_tokens_map[str(patient_id)] = approx_tokens_map.get(str(patient_id), 0) + (chars or 0)
    # Convert char counts to approximate token counts (4 chars ≈ 1 token)
    approx_tokens_map = {pid: max(1, chars // 4) for pid, chars in approx_tokens_map.items()}
    summary_map: Dict[str, Dict[str, Any]] = {}
    for patient_id, payload in summary_rows:
        if not patient_id or not payload:
            continue
        try:
            summary_map[str(patient_id)] = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            summary_map[str(patient_id)] = {"raw": payload}
    key_lab_map: Dict[str, Dict[str, Any]] = {}
    for patient_id, canonical_key, n, last_dt in key_lab_rows:
        if not patient_id or not canonical_key:
            continue
        key_lab_map.setdefault(str(patient_id), {})[canonical_key] = {
            "count": n or 0,
            "last_dt": last_dt,
        }
    report_counts: Dict[str, Dict[str, int]] = {}
    for patient_id, report_type, count in count_rows:
        if not patient_id or not report_type:
            continue
        bucket = report_counts.setdefault(str(patient_id), {})
        label = _normalize_report_type_label(str(report_type))
        bucket[label] = bucket.get(label, 0) + (count or 0)
    patients: List[PatientSummary] = []
    for patient_id, fullname, dob in patient_rows:
        if not patient_id:
            continue
        patient = PatientSummary(
            id=str(patient_id),
            name=(fullname or "").strip() or str(patient_id),
            latest_report_date=latest_map.get(str(patient_id)),
            earliest_report_date=earliest_map.get(str(patient_id)),
            approx_tokens=approx_tokens_map.get(str(patient_id)),
            report_counts=report_counts.get(str(patient_id)),
            dob=dob or None,
            key_lab_recency=key_lab_map.get(str(patient_id)),
            summary=summary_map.get(str(patient_id)),
        )
        patients.append(patient)
    return patients

def _agent_backend() -> str:
    return (os.getenv("CLINICAL_AGENT_BACKEND") or "vllm").strip().lower()


def _allowed_gemini_models() -> List[str]:
    raw = os.getenv("GEMINI_ALLOWED_MODELS")
    if raw:
        models = [item.strip() for item in raw.split(",") if item.strip()]
        if models:
            return models
    default_model = os.getenv("GEMINI_MODEL")
    return [default_model] if default_model else []


def _allowed_vllm_models() -> List[str]:
    raw = os.getenv("VLLM_ALLOWED_MODELS")
    if raw:
        models = [item.strip() for item in raw.split(",") if item.strip()]
        if models:
            return models
    default_model = os.getenv("VLLM_MODEL")
    return [default_model] if default_model else []


_cached_vllm_model: Optional[str] = None
_cached_gemini_model: Optional[str] = None


def _resolve_vllm_model(requested: Optional[str]) -> str:
    global _cached_vllm_model
    if _cached_vllm_model is not None and not requested:
        return _cached_vllm_model

    allowed = _allowed_vllm_models()
    default_model = os.getenv("VLLM_MODEL") or (allowed[0] if allowed else "gpt-oss-120b")
    enforce_env = (os.getenv("VLLM_ENFORCE_ENV_MODEL") or "").strip().lower() in {"1", "true", "yes", "on"}

    # If enforcement is on, ignore the client-provided model and stick to the env default.
    model_name = default_model if enforce_env else (requested or default_model)

    if allowed and model_name not in allowed:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not allowed for vLLM backend.")

    if _cached_vllm_model is None:
        print(f"INFO:     Resolved vLLM model: {model_name}")
        _cached_vllm_model = model_name
    return model_name


def _resolve_gemini_model(requested: Optional[str]) -> str:
    global _cached_gemini_model
    if _cached_gemini_model is not None and not requested:
        return _cached_gemini_model

    allowed = _allowed_gemini_models()
    default_model = os.getenv("GEMINI_MODEL") or (allowed[0] if allowed else "gemini-3.1-pro-preview")
    model_name = requested or default_model
    if allowed and model_name not in allowed:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not allowed.")

    if _cached_gemini_model is None:
        print(f"INFO:     Resolved Gemini model: {model_name}")
        _cached_gemini_model = model_name
    return model_name


def _lookup_latest_report_date(patient_id: str, db_path: str | Path | None = None) -> Optional[str]:
    resolved = str(db_path) if db_path else str(ReportsRAGTool.DB_PATH)
    conn = sqlite3.connect(resolved)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT report_id, report_date
            FROM reports
            WHERE patient_id = ?
            """,
            (patient_id,),
        )
        rows = cursor.fetchall()
        latest_iso: Optional[str] = None
        for report_id, report_date in rows:
            candidate = report_date or ReportsRAGTool._derive_report_date(report_id or "")
            normalized = ReportsRAGTool._normalize_report_date(candidate)
            if not normalized:
                continue
            if latest_iso is None or normalized > latest_iso:
                latest_iso = normalized
        return latest_iso
    except Exception:  # pragma: no cover - defensive logging
        logger.warning("Failed to derive latest report date for patient %s", patient_id, exc_info=True)
        return None
    finally:
        conn.close()


def _current_patients(db_path: str | Path | None = None) -> List[PatientSummary]:
    patients = _load_patients_from_db(db_path=db_path)
    if not patients:
        raise RuntimeError("Patient database returned no entries.")
    return patients


def _find_patient_entry(patient_id: Optional[str], db_path: str | Path | None = None) -> Optional[PatientSummary]:
    if not patient_id:
        return None
    return next((entry for entry in _current_patients(db_path=db_path) if entry.id == patient_id), None)


def build_patient_context(patient: Optional[PatientSummary]) -> str:
    if not patient:
        return ""
    stichtag = _parse_date_ymd(patient.latest_report_date or "")
    # Age at Stichtag
    age_line = ""
    if patient.dob and stichtag:
        dob_date = _parse_dob(patient.dob)
        if dob_date:
            age_line = f"Alter am Stichtag: {_age_at(dob_date, stichtag)} Jahre"
    lines = [
        f"Patienten-ID: {patient.id}",
        f"Name: {patient.name}" if patient.name else "",
        age_line,
        f"Details: {patient.details}" if patient.details else "",
        (
            f"Erster Bericht: {patient.earliest_report_date}"
            if patient.earliest_report_date
            else ""
        ),
        (
            f"Letzter Bericht (Stichtag): {patient.latest_report_date}"
            if patient.latest_report_date
            else ""
        ),
        (
            f"Ungefähre Datenmenge: ~{patient.approx_tokens // 1000}k Tokens"
            if patient.approx_tokens
            else ""
        ),
    ]
    if patient.report_counts:
        counts_str = ", ".join(
            f"{rtype}: {n}"
            for rtype, n in sorted(patient.report_counts.items(), key=lambda x: -x[1])
        )
        lines.append(f"Berichte nach Typ: {counts_str}")
    # Key lab recency for scoring (β2M, Albumin, LDH)
    if patient.key_lab_recency:
        lab_parts = []
        for key, label in [("Beta2-Mikroglobulin", "β2M"), ("Albumin", "Albumin"), ("LDH", "LDH")]:
            info = patient.key_lab_recency.get(key)
            if info:
                last_date = _parse_date_ymd(info.get("last_dt") or "")
                if last_date and stichtag:
                    days = (stichtag - last_date).days
                    lab_parts.append(
                        f"{label} (n={info['count']}, zuletzt {last_date.strftime('%Y-%m')}, {days}d vor Stichtag)"
                    )
                else:
                    lab_parts.append(f"{label} (n={info['count']})")
            else:
                lab_parts.append(f"{label}: nicht vorhanden")
        lines.append(f"Scoring-Laborwerte: {'; '.join(lab_parts)}")
    return "\n".join([line for line in lines if line])


def _create_vllm_agent(model_name: Optional[str], *, patient_id: str, max_tool_rounds: int = 6, db_path: str | Path | None = None, base_url: Optional[str] = None) -> VLLMChatAgent:
    base_url = base_url or os.getenv("VLLM_BASE_URL") or "http://10.32.16.43:4000"
    api_key = os.getenv("OPENAI_API_KEY") or "sk-JFFBxryuvDtgt_WoIwYdmw"
    model = _resolve_vllm_model(model_name)
    completion_kwargs: Dict[str, Any] = {}
    temp = os.getenv("VLLM_TEMPERATURE")
    if temp:
        completion_kwargs["temperature"] = float(temp)
    max_completion_tokens = os.getenv("VLLM_MAX_COMPLETION_TOKENS") or os.getenv("MAX_TOKENS_ANSWER")
    if max_completion_tokens:
        completion_kwargs["max_tokens"] = int(max_completion_tokens)
    use_hybrid = os.getenv("USE_HYBRID_RETRIEVAL", "").lower() in ("1", "true", "yes")
    tools = load_default_tools(patient_id=patient_id, db_path=db_path, use_hybrid=use_hybrid)
    config = VLLMConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        completion_kwargs=completion_kwargs,
        service_tier=os.getenv("VLLM_SERVICE_TIER") or None,
    )
    return VLLMChatAgent(config=config, tools=tools, max_tool_rounds=max_tool_rounds)


def _create_gemini_agent(model_name: Optional[str], *, patient_id: str, db_path: str | Path | None = None) -> GeminiChatAgent:
    project_id = os.getenv("GEMINI_PROJECT_ID")
    if not project_id:
        raise RuntimeError("GEMINI_PROJECT_ID is required for the Gemini backend.")
    location = os.getenv("GEMINI_LOCATION", "us-central1")
    model = _resolve_gemini_model(model_name)
    use_hybrid = os.getenv("USE_HYBRID_RETRIEVAL", "").lower() in ("1", "true", "yes")
    tools = load_default_tools(patient_id=patient_id, db_path=db_path, use_hybrid=use_hybrid)
    config = GeminiConfig(project_id=project_id, location=location, model=model)
    return GeminiChatAgent(config=config, tools=tools)


def create_agent(model_name: Optional[str] = None, *, patient_id: Optional[str] = None, max_tool_rounds: int = 6, db_path: str | Path | None = None, base_url: Optional[str] = None) -> Any:
    normalized_patient = (patient_id or "").strip()
    if not normalized_patient:
        raise ValueError("patient_id is required to create an agent.")
    backend = _agent_backend()
    if backend == "gemini":
        return _create_gemini_agent(model_name, patient_id=normalized_patient, db_path=db_path)
    return _create_vllm_agent(model_name, patient_id=normalized_patient, max_tool_rounds=max_tool_rounds, db_path=db_path, base_url=base_url)


def create_app() -> FastAPI:
    fastapi_app = FastAPI(title="Clinical RAG Assistant API", version="0.1.0")

    @fastapi_app.on_event("startup")
    async def log_context_mode() -> None:
        raw = os.getenv("AGENT_FULL_CONTEXT_FOR_STEPS", "")
        enabled = raw.strip().lower() in {"1", "true", "yes", "on"}
        mode_label = "full_context" if enabled else "summaries"
        # Print ensures visibility even if app loggers are filtered by Uvicorn.
        print(f"INFO:     Tool execution context mode: {mode_label}")
        logger.info("INFO:     execution context mode: %s", mode_label)

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://10.184.8.240:5173",
            "http://10.184.13.69:5173",
            "http://192.168.2.106:5173",
            "http://10.32.16.225:5173",
            "http://10.184.6.57:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @fastapi_app.get("/patients", response_model=List[PatientSummary])
    async def list_patients() -> List[PatientSummary]:
        try:
            return _current_patients()
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @fastapi_app.get("/models")
    async def list_models() -> Dict[str, Any]:
        backend = _agent_backend()
        if backend == "gemini":
            models = _allowed_gemini_models()
            default_model = os.getenv("GEMINI_MODEL") or (models[0] if models else "")
            capacity_tokens = int(os.getenv("GEMINI_CONTEXT_TOKENS") or "260000")
        else:
            models = _allowed_vllm_models()
            default_model = os.getenv("VLLM_MODEL") or (models[0] if models else "")
            capacity_tokens = int(os.getenv("VLLM_CONTEXT_TOKENS") or "65000")
        return {"models": models, "default": default_model, "backend": backend, "context_tokens": capacity_tokens}

    @fastapi_app.post("/chat", response_model=ChatReply)
    async def chat(request: ChatRequest) -> ChatReply:
        logger.info("Received chat request for patient %s", request.patient_id)
        if not request.patient_id or not request.patient_id.strip():
            raise HTTPException(status_code=400, detail="patient_id is required.")
        run_id = str(uuid.uuid4())
        run_store.start_run(run_id)
        events: List[Dict[str, Any]] = []
        backend = _agent_backend()
        model_name = (
            _resolve_gemini_model(request.model) if backend == "gemini" else _resolve_vllm_model(request.model)
        )

        patient_entry = _find_patient_entry(request.patient_id)
        if not patient_entry:
            raise HTTPException(status_code=404, detail=f"Patient '{request.patient_id}' not found.")
        patient_context = build_patient_context(patient_entry)
        patient_id_for_tools = patient_entry.id
        agent = create_agent(model_name, patient_id=patient_id_for_tools)
        resolved_question = _inject_reference_date(
            request.question,
            patient_entry.latest_report_date if patient_entry else None,
        )

        def capture(event: Dict[str, Any]) -> None:
            record = _normalise_event(event)
            events.append(record)
            run_store.add_event(run_id, record)
            _log_agent_monitor_event(run_id, record)

        try:
            for tool in getattr(agent, "tools", []):
                if hasattr(tool, "patient_id"):
                    tool.patient_id = patient_id_for_tools
            conversation_history = _format_history_for_prompt(request.history)
            reply = agent.answer_with_rag(
                resolved_question,
                conversation_history=conversation_history,
                patient_context=patient_context,
                reference_date=patient_entry.latest_report_date if patient_entry else None,
                event_handler=capture,
            )
            metadata = dict(reply.additional_kwargs or {})
            metadata["model"] = model_name
            _log_user_activity(
                question=resolved_question,
                patient_id=request.patient_id,
                model_name=model_name,
                metadata=metadata,
                answer=reply.content or "",
            )
            return ChatReply(
                content=reply.content or "",
                metadata=metadata,
                run_id=run_id,
                events=events,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Agent call failed: %s", exc)
            error_event = {
                "type": "run_failed",
                "payload": {"message": str(exc), "error_class": exc.__class__.__name__},
                "timestamp": time.time(),
            }
            capture(error_event)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @fastapi_app.post("/chat/stream")
    async def chat_stream(request: Request, chat_request: ChatRequest) -> StreamingResponse:
        logger.info("Streaming chat request for patient %s", chat_request.patient_id)
        if not chat_request.patient_id or not chat_request.patient_id.strip():
            raise HTTPException(status_code=400, detail="patient_id is required.")
        run_id = str(uuid.uuid4())
        run_store.start_run(run_id)
        backend = _agent_backend()
        model_name = (
            _resolve_gemini_model(chat_request.model)
            if backend == "gemini"
            else _resolve_vllm_model(chat_request.model)
        )

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
        patient_entry = _find_patient_entry(chat_request.patient_id)
        if not patient_entry:
            raise HTTPException(status_code=404, detail=f"Patient '{chat_request.patient_id}' not found.")
        patient_context = build_patient_context(patient_entry)
        conversation_history = _format_history_for_prompt(chat_request.history)
        patient_id_for_tools = patient_entry.id
        resolved_question = _inject_reference_date(
            chat_request.question,
            patient_entry.latest_report_date if patient_entry else None,
        )

        def capture(event: Dict[str, Any]) -> None:
            record = _normalise_event(event)
            run_store.add_event(run_id, record)
            _log_agent_monitor_event(run_id, record)
            loop.call_soon_threadsafe(queue.put_nowait, record)

        def worker() -> None:
            agent = create_agent(model_name, patient_id=patient_id_for_tools)
            for tool in getattr(agent, "tools", []):
                if hasattr(tool, "patient_id"):
                    tool.patient_id = patient_id_for_tools
            try:
                agent.answer_with_rag(
                    resolved_question,
                    conversation_history=conversation_history,
                    patient_context=patient_context,
                    reference_date=patient_entry.latest_report_date if patient_entry else None,
                    event_handler=capture,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Agent call failed during stream: %s", exc)
                capture(
                    {
                        "type": "run_failed",
                        "payload": {"message": str(exc), "error_class": exc.__class__.__name__},
                        "timestamp": time.time(),
                    }
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, worker)

        async def event_generator() -> AsyncIterator[str]:
            try:
                while True:
                    if await request.is_disconnected():
                        logger.info("Client disconnected from stream for run %s", run_id)
                        break
                    event = await queue.get()
                    if event is None:
                        break
                    if event.get("type") == "run_completed":
                        payload_data = event.get("payload") or {}
                        metadata = payload_data.get("metadata")
                        answer = payload_data.get("answer") or ""
                        _log_user_activity(
                            question=resolved_question,
                            patient_id=chat_request.patient_id,
                            model_name=model_name,
                            metadata=metadata if isinstance(metadata, dict) else {},
                            answer=answer,
                        )
                    payload = {"run_id": run_id, **event}
                    yield json.dumps(payload, ensure_ascii=False) + "\n"
            finally:
                # Drain remaining events to avoid pending tasks
                while True:
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                logger.info("Stream for run %s completed", run_id)

        return StreamingResponse(event_generator(), media_type="application/x-ndjson")

    @fastapi_app.get("/monitor/runs/{run_id}")
    async def get_run(run_id: str) -> Dict[str, Any]:
        run = run_store.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @fastapi_app.get("/monitor/runs")
    async def list_runs() -> Dict[str, Any]:
        return {"runs": run_store.list_runs()}

    @fastapi_app.get("/reports/{report_id}")
    async def get_report(report_id: str) -> Dict[str, Any]:
        conn = sqlite3.connect(str(ReportsRAGTool.DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT report_id, patient_id, report_type, report_date, content, filename
            FROM reports
            WHERE report_id = ?
            """,
            (report_id,),
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            raise HTTPException(status_code=404, detail="Report not found")
        return {
            "report_id": row["report_id"],
            "patient_id": row["patient_id"],
            "report_type": row["report_type"],
            "report_date": row["report_date"],
            "content": row["content"],
            "filename": row["filename"],
        }

    return fastapi_app


app = create_app()
