#!/usr/bin/env python3
"""
Build a SQLite database from JSON-formatted synthetic patient cases.

Each patient is described by a single JSON file. The script creates (or updates)
a database with the schema expected by the agent:

  patients         — one row per patient
  reports          — one row per clinical document
  report_sections  — one row per section within a document  ← agent reads from here
  lab_values       — one row per lab measurement            ← agent reads from here

Usage
-----
  python scripts/build_database.py cases/ --db src/database/synthetic.sqlite
  python scripts/build_database.py cases/patient_001.json --db src/database/synthetic.sqlite

Input format (one JSON file per patient)
-----------------------------------------
{
  "patient_id":    "patient_001",          # required, must be unique
  "firstname":     "Max",                  # optional
  "lastname":      "Mustermann",           # optional
  "date_of_birth": "1960-03-15",           # optional, YYYY-MM-DD

  "reports": [
    {
      "report_type": "doctor_letter",      # required — see VALID_REPORT_TYPES below
      "report_date": "2023-06-15",         # required, YYYY-MM-DD
      "title":       "Discharge Letter",   # optional, used as report_id base
      "sections": [
        {
          "name":    "Diagnosis",          # section heading (displayed in citations)
          "content": "Patient presents with ..."
        },
        {
          "name":    "Treatment",
          "content": "Started VRd protocol ..."
        }
      ]
    }
  ],

  "lab_values": [
    {
      "name":            "Hemoglobin",     # lab test name — used as canonical_key
      "date":            "2023-06-15",     # YYYY-MM-DD
      "time":            "08:30",          # optional, HH:MM
      "value":           "12.5",           # string (may contain < > +)
      "unit":            "g/dL",           # optional
      "reference_range": "12.0-16.0"       # optional
    }
  ]
}

Valid report_type values
------------------------
  doctor_letter  — discharge summaries, outpatient letters
  consult        — specialist consultation notes
  radiology      — imaging reports (CT, MRI, PET, X-ray, ultrasound)
  pathology      — histopathology, biopsy reports
  tumor_board    — multidisciplinary tumor board decisions
  cardiology     — echocardiography, cardiology assessments
  cytology       — cytology reports
  flow           — flow cytometry reports
  history        — patient history, anamnesis
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VALID_REPORT_TYPES = {
    "doctor_letter",
    "consult",
    "radiology",
    "pathology",
    "tumor_board",
    "cardiology",
    "cytology",
    "flow",
    "history",
}

UMLAUT_MAP = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.executescript("""
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS patients (
        patient_id  TEXT PRIMARY KEY,
        firstname   TEXT,
        lastname    TEXT,
        fullname    TEXT,
        dob         TEXT,
        created_at  TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS reports (
        report_id   TEXT PRIMARY KEY,
        patient_id  TEXT NOT NULL,
        filename    TEXT,
        report_type TEXT,
        report_date TEXT,
        created_at  TEXT,
        source_path TEXT,
        sha256      TEXT,
        content     TEXT,
        FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
    );

    CREATE TABLE IF NOT EXISTS report_sections (
        section_id      TEXT PRIMARY KEY,
        report_id       TEXT NOT NULL,
        patient_id      TEXT NOT NULL,
        section_name    TEXT NOT NULL,
        section_content TEXT NOT NULL,
        section_order   INTEGER NOT NULL,
        word_count      INTEGER,
        filename        TEXT,
        report_type     TEXT,
        report_date     TEXT,
        created_at      TEXT,
        source_path     TEXT,
        FOREIGN KEY(report_id)  REFERENCES reports(report_id),
        FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
    );

    CREATE TABLE IF NOT EXISTS lab_values (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id      TEXT NOT NULL,
        canonical_key   TEXT NOT NULL,
        mapped_name     TEXT,
        date            TEXT,
        time            TEXT,
        assessment_dt   TEXT,
        value           TEXT,
        value_num       REAL,
        unit            TEXT,
        ref_range       TEXT,
        source_test_id  TEXT,
        FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
    );

    CREATE INDEX IF NOT EXISTS idx_sections_patient  ON report_sections(patient_id);
    CREATE INDEX IF NOT EXISTS idx_sections_report   ON report_sections(report_id);
    CREATE INDEX IF NOT EXISTS idx_sections_type     ON report_sections(report_type);
    CREATE INDEX IF NOT EXISTS idx_sections_date     ON report_sections(report_date);
    CREATE INDEX IF NOT EXISTS idx_lab_patient       ON lab_values(patient_id);
    CREATE INDEX IF NOT EXISTS idx_lab_key           ON lab_values(canonical_key);
    CREATE INDEX IF NOT EXISTS idx_lab_date          ON lab_values(date);
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_key(name: str) -> str:
    """Normalise a lab test name to a stable lookup key."""
    name = (name or "").lower().translate(UMLAUT_MAP)
    cleaned = []
    for ch in name:
        if ch.isalnum():
            cleaned.append(ch)
        else:
            cleaned.append(" ")
    return " ".join("".join(cleaned).split())


def _to_numeric(text: str) -> Optional[float]:
    if not text:
        return None
    t = text.strip().lstrip("<>").strip().replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def _safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", text).strip("_")[:80]


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Insertion helpers
# ---------------------------------------------------------------------------

def _upsert_patient(conn: sqlite3.Connection, patient_id: str, data: Dict[str, Any]) -> None:
    firstname = data.get("firstname", "")
    lastname  = data.get("lastname", "")
    fullname  = f"{firstname} {lastname}".strip()
    dob       = data.get("date_of_birth", "")
    conn.execute(
        """
        INSERT OR REPLACE INTO patients(patient_id, firstname, lastname, fullname, dob)
        VALUES (?, ?, ?, ?, ?)
        """,
        (patient_id, firstname, lastname, fullname, dob),
    )


def _insert_report(
    conn: sqlite3.Connection,
    patient_id: str,
    report_id: str,
    report_type: str,
    report_date: str,
    title: str,
    full_content: str,
) -> str:
    sha = _sha256(full_content)
    dup = conn.execute(
        "SELECT report_id FROM reports WHERE patient_id=? AND sha256=?",
        (patient_id, sha),
    ).fetchone()
    if dup:
        return dup[0]

    # Resolve conflicts by appending a counter
    base_id = report_id
    counter = 1
    while conn.execute("SELECT 1 FROM reports WHERE report_id=?", (report_id,)).fetchone():
        report_id = f"{base_id}_{counter}"
        counter += 1

    conn.execute(
        """
        INSERT INTO reports(report_id, patient_id, filename, report_type, report_date,
                            created_at, source_path, sha256, content)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (report_id, patient_id, f"{title}.json", report_type, report_date, _now(), "", sha, full_content),
    )
    return report_id


def _insert_sections(
    conn: sqlite3.Connection,
    patient_id: str,
    report_id: str,
    report_type: str,
    report_date: str,
    sections: List[Dict[str, Any]],
) -> int:
    now = _now()
    inserted = 0
    for order, sec in enumerate(sections):
        name    = str(sec.get("name") or "Content").strip()
        content = str(sec.get("content") or "").strip()
        if not content:
            continue
        section_id = f"{report_id}::{_safe_id(name)}::{order}"
        word_count = len(content.split())
        conn.execute(
            """
            INSERT OR IGNORE INTO report_sections(
                section_id, report_id, patient_id, section_name, section_content,
                section_order, word_count, filename, report_type, report_date, created_at, source_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                section_id, report_id, patient_id, name, content,
                order, word_count, f"{report_id}.json", report_type, report_date, now, "",
            ),
        )
        inserted += 1
    return inserted


def _insert_lab_values(
    conn: sqlite3.Connection,
    patient_id: str,
    lab_values: List[Dict[str, Any]],
) -> int:
    inserted = 0
    for lab in lab_values:
        name  = str(lab.get("name") or "").strip()
        if not name:
            print(f"  [WARN] Lab value missing 'name', skipping: {lab}", file=sys.stderr)
            continue
        date_str   = str(lab.get("date") or "").strip()
        time_str   = str(lab.get("time") or "").strip()
        value      = str(lab.get("value") or "").strip()
        unit       = str(lab.get("unit") or "").strip()
        ref_range  = str(lab.get("reference_range") or "").strip()
        ckey       = _canonical_key(name)
        value_num  = _to_numeric(value)
        assess_dt  = f"{date_str}T{time_str}" if date_str and time_str else date_str or None
        conn.execute(
            """
            INSERT INTO lab_values(
                patient_id, canonical_key, mapped_name, date, time, assessment_dt,
                value, value_num, unit, ref_range, source_test_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id, ckey, name,
                date_str or None, time_str or None, assess_dt,
                value, value_num, unit or None, ref_range or None, None,
            ),
        )
        inserted += 1
    return inserted


# ---------------------------------------------------------------------------
# Per-patient processing
# ---------------------------------------------------------------------------

def process_patient(conn: sqlite3.Connection, data: Dict[str, Any]) -> Tuple[int, int]:
    """Insert one patient record into the database. Returns (sections, labs) counts."""
    patient_id = str(data.get("patient_id") or "").strip()
    if not patient_id:
        raise ValueError("Patient record is missing 'patient_id'.")

    _upsert_patient(conn, patient_id, data)

    total_sections = 0
    total_labs     = 0

    for report in data.get("reports", []):
        report_type = str(report.get("report_type") or "doctor_letter").strip().lower()
        if report_type not in VALID_REPORT_TYPES:
            print(
                f"  [WARN] Unknown report_type '{report_type}' for patient {patient_id}. "
                f"Valid types: {', '.join(sorted(VALID_REPORT_TYPES))}",
                file=sys.stderr,
            )
            report_type = "doctor_letter"

        report_date = str(report.get("report_date") or "").strip()
        title       = str(report.get("title") or f"{report_type}_{report_date}").strip()
        sections    = report.get("sections", [])

        if not sections:
            print(f"  [WARN] Report '{title}' for patient {patient_id} has no sections — skipping.", file=sys.stderr)
            continue

        full_content = "\n\n".join(
            f"## {sec.get('name', 'Content')}\n{sec.get('content', '')}" for sec in sections
        )
        base_report_id = _safe_id(f"{patient_id}_{report_date}_{title}")
        report_id = _insert_report(
            conn, patient_id, base_report_id, report_type, report_date, title, full_content
        )
        n = _insert_sections(conn, patient_id, report_id, report_type, report_date, sections)
        total_sections += n

    total_labs += _insert_labs(conn, patient_id, data.get("lab_values", []))
    return total_sections, total_labs


def _insert_labs(conn: sqlite3.Connection, patient_id: str, lab_values: List[Dict[str, Any]]) -> int:
    return _insert_lab_values(conn, patient_id, lab_values)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build(input_path: Path, db_path: Path, replace: bool = False) -> None:
    if replace and db_path.exists():
        db_path.unlink()
        print(f"Removed existing database: {db_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Opening database: {db_path}")
    conn = sqlite3.connect(db_path, timeout=30.0)
    init_db(conn)

    # Collect input files
    if input_path.is_file():
        json_files = [input_path]
    elif input_path.is_dir():
        json_files = sorted(input_path.rglob("*.json"))
    else:
        print(f"Error: {input_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    if not json_files:
        print(f"No JSON files found under {input_path}.", file=sys.stderr)
        sys.exit(1)

    total_patients  = 0
    total_sections  = 0
    total_labs      = 0
    errors          = 0

    for jf in json_files:
        print(f"Processing {jf.name} ...", end=" ")
        try:
            with jf.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"ERROR (JSON parse): {exc}", file=sys.stderr)
            errors += 1
            continue

        # Support both a single patient object and a list
        records = data if isinstance(data, list) else [data]
        for record in records:
            try:
                secs, labs = process_patient(conn, record)
                total_sections += secs
                total_labs     += labs
                total_patients += 1
                print(f"{record.get('patient_id')} — {secs} sections, {labs} labs")
            except Exception as exc:
                pid = record.get("patient_id", "<unknown>")
                print(f"ERROR processing {pid}: {exc}", file=sys.stderr)
                errors += 1

        conn.commit()

    # Summary
    n_patients = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    n_reports  = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
    n_sections = conn.execute("SELECT COUNT(*) FROM report_sections").fetchone()[0]
    n_labs     = conn.execute("SELECT COUNT(*) FROM lab_values").fetchone()[0]

    print(f"\n{'='*50}")
    print(f"Database: {db_path}")
    print(f"Patients:         {n_patients}")
    print(f"Reports:          {n_reports}")
    print(f"Report sections:  {n_sections}")
    print(f"Lab values:       {n_labs}")
    if errors:
        print(f"Errors:           {errors}", file=sys.stderr)
    print(f"{'='*50}")

    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a SQLite database from JSON-formatted synthetic patient cases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        help="Path to a single patient JSON file or a directory containing multiple JSON files.",
    )
    parser.add_argument(
        "--db",
        default="src/database/synthetic.sqlite",
        help="Output SQLite database path (default: src/database/synthetic.sqlite).",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete the existing database before building (full rebuild).",
    )
    args = parser.parse_args()
    build(Path(args.input), Path(args.db), replace=args.replace)


if __name__ == "__main__":
    main()
