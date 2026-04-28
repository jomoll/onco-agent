"""Deterministic checker for policy skills and their inline tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.skills.policy_engine import EvidenceItem, rank_evidence, derive_therapy_state, resolve_conflicts  # type: ignore
from src.skills.registry import get_skill_catalog  # type: ignore


def _load_policy_skills() -> Dict[str, Dict[str, Any]]:
    catalog = {}
    for entry in get_skill_catalog():
        if entry.get("category") == "policy":
            catalog[entry["id"]] = entry.get("payload") or {}
    return catalog


def _evidence_from_dicts(items: List[Dict[str, Any]]) -> List[EvidenceItem]:
    results = []
    for itm in items:
        results.append(EvidenceItem(**itm))
    return results


def run_tests() -> int:
    policy_skills = _load_policy_skills()
    failures: List[str] = []
    for skill_id, payload in policy_skills.items():
        tests = payload.get("tests") or []
        for test in tests:
            name = test.get("name", "unnamed")
            t_input = test.get("input") or {}
            expect = test.get("expect") or {}
            try:
                if skill_id == "policy.temporal_authority":
                    evid = _evidence_from_dicts(t_input.get("evidence_items") or [])
                    ranked, trace = rank_evidence(evid, t_input.get("question_type", "*"), skill_id)
                    top_id = ranked[0].evidence_id if ranked else None
                    if expect.get("top_ranked_evidence_id") and top_id != expect.get("top_ranked_evidence_id"):
                        failures.append(f"{skill_id}:{name}: expected top {expect.get('top_ranked_evidence_id')} got {top_id}")
                    if "fired_rules_contains" in expect:
                        fired = []
                        if trace.get("scores"):
                            for meta in trace["scores"].values():
                                fired.extend(meta.get("fired_rules") or [])
                        missing = [r for r in expect["fired_rules_contains"] if r not in fired]
                        if missing:
                            failures.append(f"{skill_id}:{name}: missing fired rules {missing}")
                elif skill_id == "policy.plan_vs_administered":
                    evid = _evidence_from_dicts(t_input.get("evidence_items") or [])
                    state, det, support, _trace = derive_therapy_state(
                        evid, t_input.get("target_concept")
                    )
                    if expect.get("therapy_state") and state != expect["therapy_state"]:
                        failures.append(f"{skill_id}:{name}: expected state {expect['therapy_state']} got {state}")
                    if expect.get("determinacy") and det != expect["determinacy"]:
                        failures.append(f"{skill_id}:{name}: expected determinacy {expect['determinacy']} got {det}")
                    if expect.get("support_ids") and support != expect["support_ids"]:
                        failures.append(f"{skill_id}:{name}: expected support {expect['support_ids']} got {support}")
                elif skill_id == "policy.contradiction_resolution":
                    claims = t_input.get("candidate_claims") or []
                    action, final_claim, _trace = resolve_conflicts(claims, t_input.get("question_type", "*"))
                    if expect.get("resolution_action") and action != expect["resolution_action"]:
                        failures.append(f"{skill_id}:{name}: expected action {expect['resolution_action']} got {action}")
                    if expect.get("supporting_evidence_ids"):
                        ids = []
                        if isinstance(final_claim, dict):
                            ids.extend(final_claim.get("supporting_evidence_ids") or [])
                            for claim in final_claim.get("claims", []):
                                ids.extend(claim.get("supporting_evidence_ids", []))
                        ids = list(dict.fromkeys(ids))
                        if ids != expect["supporting_evidence_ids"]:
                            failures.append(f"{skill_id}:{name}: expected support {expect['supporting_evidence_ids']} got {ids}")
            except Exception as exc:  # pragma: no cover - defensive
                failures.append(f"{skill_id}:{name}: exception {exc}")

    if failures:
        print("Policy tests failed:", file=sys.stderr)
        for fail in failures:
            print(f"  - {fail}", file=sys.stderr)
        return 1
    print("All policy tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(run_tests())
