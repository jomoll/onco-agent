#!/usr/bin/env python3
"""
Run the clinical agent on one or more synthetic patient cases.

For each patient and each question, the agent retrieves relevant report sections
and lab values from the SQLite database, applies domain skills and the policy
engine, and produces a structured answer with inline citations.

Single patient, interactive question
-------------------------------------
  python scripts/run_agent.py --db src/database/synthetic.sqlite \\
    --patient-id patient_001 \\
    --question "What is the patient's current ISS stage?"

Batch: all patients, all questions from a JSON file
----------------------------------------------------
  python scripts/run_agent.py --db src/database/synthetic.sqlite \\
    --questions-file questions.json \\
    --output answers.json

Batch: specific patients from a text file (one ID per line)
------------------------------------------------------------
  python scripts/run_agent.py --db src/database/synthetic.sqlite \\
    --patient-ids-file patient_ids.txt \\
    --questions-file questions.json \\
    --output answers.json

Questions file format
---------------------
A JSON file with a list of question objects:

  [
    {
      "id": "q01",
      "question": "What is the ISS stage at diagnosis?",
      "answer_schema": "score_value_plus_date_and_source"   // optional
    },
    ...
  ]

Output format
-------------
A JSON file (--output) containing:

  {
    "patients": [
      {
        "patient_id": "patient_001",
        "questions": [
          {
            "id": "q01",
            "question": "...",
            "answer": "Answer: ISS II | 15.10.2023 | Berechnet\\nReasoning: ...",
            "elapsed_seconds": 4.2,
            "tool_calls": 3,
            "skills": ["schema.iss_stage", "policy.evidence_ranking"],
            "error": null
          }
        ]
      }
    ]
  }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

# Suppress noisy dependency warnings
try:
    from pydantic._internal._generate_schema import UnsupportedFieldAttributeWarning  # type: ignore
    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
except Exception:
    warnings.filterwarnings("ignore", message="UnsupportedFieldAttributeWarning")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env", override=False)
except Exception:
    pass

try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None

logging.getLogger("src.agent_tools").setLevel(logging.ERROR)
logging.getLogger("src.agent_base").setLevel(logging.ERROR)
logging.getLogger("src.api_server").setLevel(logging.ERROR)

from src.api_server import (  # type: ignore
    build_patient_context,
    create_agent,
    _find_patient_entry,
    _log_agent_monitor_event,
    _normalise_event,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    # Also accept {"questions": [...]}
    return data.get("questions", data.get("questions_rewritten", []))


def _load_patient_ids(path: Path) -> List[str]:
    ids: List[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            pid = line.strip()
            if pid and not pid.startswith("#"):
                ids.append(pid)
    return ids


def _list_all_patients(db_path: str) -> List[str]:
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT patient_id FROM patients ORDER BY patient_id").fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception as exc:
        print(f"Error reading patient list from database: {exc}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Core: answer questions for one patient
# ---------------------------------------------------------------------------

def answer_patient(
    patient_id: str,
    questions: List[Dict[str, Any]],
    *,
    db_path: Optional[str] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    answer_language: str = "English",
    max_tool_rounds: int = 8,
    verbose: bool = False,
) -> Dict[str, Any]:
    patient_entry = _find_patient_entry(patient_id, db_path=db_path)
    if patient_entry is None:
        return {
            "patient_id": patient_id,
            "error": f"Patient '{patient_id}' not found in database.",
            "questions": [],
        }

    patient_context = build_patient_context(patient_entry)
    reference_date  = getattr(patient_entry, "latest_report_date", None)

    agent = create_agent(
        model_name=model_name,
        patient_id=patient_id,
        max_tool_rounds=max_tool_rounds,
        db_path=db_path,
        base_url=base_url,
    )

    answered: List[Dict[str, Any]] = []

    iterator = questions
    if _tqdm and not verbose:
        iterator = _tqdm(questions, desc=f"  Questions ({patient_id})", unit="q", leave=False)

    for item in iterator:
        agent.reset()
        run_id = str(uuid.uuid4())
        skills_used: Optional[List[str]] = None

        def event_handler(event: Dict[str, Any]) -> None:
            record = _normalise_event(event)
            payload = record.get("payload", {}) or {}
            if record.get("type") == "plan_ready":
                maybe_skills = payload.get("skills")
                if isinstance(maybe_skills, list):
                    nonlocal skills_used
                    skills_used = [str(s) for s in maybe_skills if s]
            if verbose:
                _log_agent_monitor_event(run_id, record)

        t0 = time.perf_counter()
        error_key: Optional[str] = None
        answer_text = ""

        for attempt in range(2):
            try:
                question_text = f"{item.get('question', '')}\n\n[cache_buster:{uuid.uuid4()}]"
                reply = agent.answer_with_rag(
                    question_text,
                    patient_context=patient_context,
                    answer_schema=item.get("answer_schema"),
                    reference_date=reference_date,
                    event_handler=event_handler,
                    answer_language=answer_language,
                )
                answer_text = reply.content or ""
                error_key = None
                break
            except Exception as exc:
                error_key = exc.__class__.__name__
                if attempt == 0:
                    time.sleep(2)
                    continue
                answer_text = f"ERROR: {exc}"

        elapsed = time.perf_counter() - t0

        answered.append({
            "id":              item.get("id"),
            "question":        item.get("question"),
            "answer":          answer_text,
            "elapsed_seconds": round(elapsed, 2),
            "skills":          skills_used or [],
            "error":           error_key,
        })

        if verbose:
            print(f"\n  Q: {item.get('question')}")
            print(f"  A: {answer_text}")
            print(f"  Time: {elapsed:.1f}s  |  Skills: {skills_used or []}")

    return {"patient_id": patient_id, "questions": answered}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the clinical agent on synthetic patient cases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input / scope
    parser.add_argument("--patient-id",       help="Single patient ID to run.")
    parser.add_argument("--patient-ids",      nargs="+", help="Explicit list of patient IDs.")
    parser.add_argument("--patient-ids-file", help="Text file with one patient ID per line.")
    parser.add_argument("--all-patients",     action="store_true", help="Run all patients in the database.")
    parser.add_argument("--question",         help="A single question to ask (interactive mode).")
    parser.add_argument("--questions-file",   help="JSON file with a list of question objects.")

    # Backend
    parser.add_argument("--db",        default=os.environ.get("DB_PATH", "src/database/synthetic.sqlite"),
                        help="Path to the SQLite database (default: src/database/synthetic.sqlite).")
    parser.add_argument("--model",     default=None, help="Model name (overrides VLLM_MODEL / .env).")
    parser.add_argument("--base-url",  default=None, help="LLM base URL (overrides VLLM_BASE_URL / .env).")
    parser.add_argument("--language",  default=os.environ.get("ANSWER_LANGUAGE", "English"),
                        help="Language for answers (default: English).")
    parser.add_argument("--max-rounds", type=int, default=int(os.environ.get("MAX_TOOL_ROUNDS", "8")),
                        help="Maximum tool-use rounds per question (default: 8).")

    # Output
    parser.add_argument("--output",  default=None, help="Path to write results JSON.")
    parser.add_argument("--verbose", action="store_true", help="Print answers to stdout and log agent events.")

    args = parser.parse_args()

    # --- Resolve patient IDs ---
    if args.patient_id:
        patient_ids = [args.patient_id.strip()]
    elif args.patient_ids:
        patient_ids = [p.strip() for p in args.patient_ids]
    elif args.patient_ids_file:
        patient_ids = _load_patient_ids(Path(args.patient_ids_file))
    elif args.all_patients:
        patient_ids = _list_all_patients(args.db)
        if not patient_ids:
            print("No patients found in the database.", file=sys.stderr)
            sys.exit(1)
    else:
        parser.error("Specify at least one of --patient-id, --patient-ids, --patient-ids-file, or --all-patients.")

    # --- Resolve questions ---
    if args.question:
        questions = [{"id": "q01", "question": args.question}]
    elif args.questions_file:
        questions = _load_questions(Path(args.questions_file))
    else:
        parser.error("Specify either --question or --questions-file.")

    if not questions:
        print("No questions found.", file=sys.stderr)
        sys.exit(1)

    # --- Run ---
    all_results: List[Dict[str, Any]] = []

    outer_iter = patient_ids
    if _tqdm and len(patient_ids) > 1:
        outer_iter = _tqdm(patient_ids, desc="Patients", unit="patient")

    for pid in outer_iter:
        if args.verbose or len(patient_ids) > 1:
            print(f"\nPatient: {pid}")

        result = answer_patient(
            pid,
            questions,
            db_path=args.db,
            model_name=args.model,
            base_url=args.base_url,
            answer_language=args.language,
            max_tool_rounds=args.max_rounds,
            verbose=args.verbose,
        )
        all_results.append(result)

    # --- Output ---
    payload = {"patients": all_results} if len(all_results) != 1 else all_results[0]

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\nResults written to {out_path}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
