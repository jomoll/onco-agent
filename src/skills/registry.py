"""Utility helpers for loading skill metadata for the planning prompt."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


_SKILLS_ROOT = Path(__file__).resolve().parent
_SKILLS_INDEX = _SKILLS_ROOT / "skills_index.json"

# Keys to strip from skill payloads before injecting into LLM context
# These are only useful for skill evaluation/testing, not runtime
_STRIP_KEYS = {"micro_tests", "adversarial"}


def _strip_testing_keys(obj: Any) -> Any:
    """Recursively strip testing/evaluation keys from skill payload."""
    if isinstance(obj, dict):
        return {k: _strip_testing_keys(v) for k, v in obj.items() if k not in _STRIP_KEYS}
    if isinstance(obj, list):
        return [_strip_testing_keys(item) for item in obj]
    return obj


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Skill file missing: {path}")
    if yaml is None:
        logger.warning("PyYAML not available; skipping skill file %s", path.name)
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # pragma: no cover - defensive
        logger.warning("Failed to parse skill file %s", path, exc_info=True)
        return {}


def _build_skill_catalog() -> List[Dict[str, Any]]:
    catalog: List[Dict[str, Any]] = []
    if not _SKILLS_INDEX.exists():
        logger.debug("Skill index %s missing.", _SKILLS_INDEX)
        return catalog
    try:
        index_payload = json.loads(_SKILLS_INDEX.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover
        logger.warning("Failed to parse skills_index.json", exc_info=True)
        return catalog

    for category, relative_paths in index_payload.items():
        if not isinstance(relative_paths, list):
            continue
        for rel_path in relative_paths:
            file_path = _SKILLS_ROOT / rel_path
            data = _load_yaml(file_path)
            if not isinstance(data, dict):
                continue
            skill_block = data.get("skill") or {}
            if not isinstance(skill_block, dict):
                continue
            skill_id = str(skill_block.get("id") or file_path.stem)
            name = str(skill_block.get("name") or skill_id)
            description = str(skill_block.get("description") or "").strip()
            version = str(skill_block.get("version") or "1")
            category = str(skill_block.get("category") or "general")
            triggers = skill_block.get("triggers") or {}
            if not triggers:
                logger.warning("Skill %s lacks triggers metadata.", skill_id)
            if "requirements" not in data:
                logger.warning("Skill %s has no requirements block.", skill_id)
            entry = {
                "id": skill_id,
                "name": name,
                "description": description,
                "category": str(category),
                "version": version,
                "payload": data,
                "steps": data.get("steps") or [],
                "failure_modes": data.get("failure_modes") or [],
                "structure": data.get("structure") or {},
                "requirements": data.get("requirements") or [],
                "triggers": triggers,
            }
            if entry["category"] == "workflows" and not entry["steps"]:
                logger.warning("Workflow skill %s has no steps defined.", skill_id)
            if entry["category"] == "style" and not entry["structure"]:
                logger.warning("Style skill %s has no structure template.", skill_id)
            catalog.append(entry)
    return catalog


_SKILL_CATALOG: List[Dict[str, Any]] = _build_skill_catalog()


def get_skill_catalog() -> List[Dict[str, Any]]:
    """Return all parsed skills (already loaded at import)."""
    return list(_SKILL_CATALOG)


def build_skill_prompt() -> str:
    """Return a compact bullet list describing available skills, including workflow heft."""
    entries = _SKILL_CATALOG
    if not entries:
        return "No specialty skills are registered; rely on general reasoning."
    snippets: List[str] = []
    for entry in entries:
        category = entry.get("category") or "general"
        description = entry["description"] or "No description available."
        workflow_hint = ""
        if category in {"workflow", "workflows"}:
            steps = entry.get("steps") or []
            step_count = len(steps)
            step_names = [step.get("name") or f"Step {idx+1}" for idx, step in enumerate(steps)]
            synopsis = " → ".join(step_names[:3])  # brief synopsis of first steps
            heavy_flag = "heavy" if step_count >= 4 else "light"
            workflow_hint = f" | {step_count} steps ({heavy_flag}: {synopsis})"
        snippets.append(f"- Skill {entry['id']} ({category}{workflow_hint}): {description}")
    return "\n".join(snippets)


def render_skill_context(skill_ids: List[str], *, categories: set[str] | None = None) -> str:
    """Return a formatted instruction block for the selected skills."""
    if not skill_ids or not _SKILL_CATALOG:
        return ""
    catalog_index = {entry["id"]: entry for entry in _SKILL_CATALOG}
    sections: List[str] = []
    for skill_id in skill_ids:
        entry = catalog_index.get(skill_id)
        if not entry:
            continue
        if categories and entry["category"] not in categories:
            continue
        header = f"Skill {entry['id']} (v{entry.get('version','1')} | {entry['category']})"
        sections.append(header)
        if entry["description"]:
            sections.append(f"Purpose: {entry['description']}")
        steps = entry.get("steps") or []
        if isinstance(steps, list) and steps:
            sections.append("Steps:")
            for idx, step in enumerate(steps, start=1):
                name = step.get("name") or f"Step {idx}"
                detail = step.get("detail") or ""
                sections.append(f"  {idx}. {name}: {detail}")
        structure = entry.get("structure")
        if isinstance(structure, dict):
            template = structure.get("template")
            if isinstance(template, list) and template:
                sections.append("Template hints:")
                for item in template:
                    sections.append(f"  - {item}")
        requirements = entry.get("requirements")
        if isinstance(requirements, list) and requirements:
            sections.append("Requirements:")
            for requirement in requirements:
                sections.append(f"  - {requirement}")
        payload = entry.get("payload")
        if isinstance(payload, dict) and payload:
            # Strip testing/evaluation keys before injecting into context
            stripped = _strip_testing_keys(payload)
            sections.append("Full skill:")
            if yaml is not None:
                sections.append(yaml.safe_dump(stripped, sort_keys=False, allow_unicode=False).rstrip())
            else:
                sections.append(json.dumps(stripped, ensure_ascii=False, indent=2))
    return "\n".join(sections)
