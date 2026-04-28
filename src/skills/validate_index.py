"""Validation helper to ensure skills_index.json matches files on disk."""

from __future__ import annotations

import json
import sys
from pathlib import Path


SKILLS_ROOT = Path(__file__).resolve().parent
INDEX_PATH = SKILLS_ROOT / "skills_index.json"


def main() -> int:
    if not INDEX_PATH.exists():
        print(f"Missing skills_index.json at {INDEX_PATH}", file=sys.stderr)
        return 1

    try:
        index_payload = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to parse {INDEX_PATH}: {exc}", file=sys.stderr)
        return 1

    missing: list[str] = []
    duplicates: list[str] = []
    seen: set[str] = set()

    for category, rel_paths in index_payload.items():
        if not isinstance(rel_paths, list):
            continue
        for rel_path in rel_paths:
            if rel_path in seen:
                duplicates.append(rel_path)
                continue
            seen.add(rel_path)
            target = SKILLS_ROOT / rel_path
            if not target.exists():
                missing.append(rel_path)

    # Detect unindexed YAML files on disk.
    yaml_on_disk = {
        str(path.relative_to(SKILLS_ROOT))
        for path in SKILLS_ROOT.rglob("*.yaml")
    }
    unindexed = sorted(yaml_on_disk - seen)

    if missing or duplicates or unindexed:
        if missing:
            print("Missing files referenced in index:", *missing, sep="\n  - ", file=sys.stderr)
        if duplicates:
            print("Duplicate entries in index:", *duplicates, sep="\n  - ", file=sys.stderr)
        if unindexed:
            print("YAML files not listed in index:", *unindexed, sep="\n  - ", file=sys.stderr)
        return 1

    print("skills_index.json is consistent with files on disk.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
