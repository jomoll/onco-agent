from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lab_catalog_resolver import LabCatalogResolver, normalise_label


def _build_catalog(tmp_path: Path) -> Path:
    db_path = tmp_path / "catalog.sqlite"
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE lab_catalog(
                lab_id TEXT PRIMARY KEY,
                code TEXT,
                code_norm TEXT,
                display_name TEXT,
                n_variants INTEGER,
                variants_json TEXT,
                search_terms_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE lab_aliases(
                alias TEXT,
                alias_norm TEXT,
                lab_id TEXT,
                code TEXT
            )
            """
        )
        entries = [
            ("aaa111bbb111", "Kreatinin", normalise_label("Kreatinin"), "Kreatinin", ["Kreatinin"], ["kreatinin"]),
            ("ccc222ddd222", "Calcium ionisiert", normalise_label("Calcium ionisiert"), "Calcium ionisiert", ["Ca++ (ionisiert)", "Ca++"], ["calcium", "ionisiert", "ca++"]),
            ("eee333fff333", "Paraprotein IgG", normalise_label("Paraprotein IgG"), "Paraprotein IgG", ["IgG paraprotein"], ["paraprotein", "igg"]),
        ]
        for lab_id, code, code_norm, display, variants, search_terms in entries:
            conn.execute(
                "INSERT INTO lab_catalog VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    lab_id,
                    code,
                    code_norm,
                    display,
                    len(variants),
                    json.dumps(variants, ensure_ascii=False),
                    json.dumps(search_terms, ensure_ascii=False),
                ),
            )
        alias_rows = [
            ("Creatinine", normalise_label("Creatinine"), "aaa111bbb111", "Kreatinin"),
            ("Ca++ (ionisiert)", normalise_label("Ca++ (ionisiert)"), "ccc222ddd222", "Calcium ionisiert"),
            ("Paraproteine", normalise_label("Paraproteine"), "eee333fff333", "Paraprotein IgG"),
        ]
        for alias, alias_norm, lab_id, code in alias_rows:
            conn.execute(
                "INSERT INTO lab_aliases VALUES (?, ?, ?, ?)",
                (alias, alias_norm, lab_id, code),
            )
    conn.close()
    return db_path


def test_exact_match(tmp_path: Path) -> None:
    db_path = _build_catalog(tmp_path)
    resolver = LabCatalogResolver(db_path)
    result = resolver.resolve("Kreatinin")
    assert result.status == "confident"
    assert result.selected[0].lab_id == "aaa111bbb111"


def test_umlaut_normalization(tmp_path: Path) -> None:
    db_path = _build_catalog(tmp_path)
    resolver = LabCatalogResolver(db_path)
    result = resolver.resolve("Calcium ionisiert")
    assert result.status == "confident"
    assert result.selected[0].lab_id == "ccc222ddd222"


def test_alias_resolution(tmp_path: Path) -> None:
    db_path = _build_catalog(tmp_path)
    resolver = LabCatalogResolver(db_path)
    result = resolver.resolve("Paraproteine")
    assert result.status == "confident"
    assert result.selected[0].lab_id == "eee333fff333"


def test_trigram_similarity(tmp_path: Path) -> None:
    db_path = _build_catalog(tmp_path)
    resolver = LabCatalogResolver(db_path)
    result = resolver.resolve("Creatinin")
    assert result.status in {"confident", "ambiguous"}
    assert any(c.lab_id == "aaa111bbb111" for c in result.candidates)


def test_no_match(tmp_path: Path) -> None:
    db_path = _build_catalog(tmp_path)
    resolver = LabCatalogResolver(db_path)
    result = resolver.resolve("Unknown Test")
    assert result.status == "no_match"
