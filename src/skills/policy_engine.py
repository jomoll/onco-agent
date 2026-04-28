"""Deterministic policy engine operating on normalized evidence items."""

from __future__ import annotations

import dataclasses
import datetime as dt
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .registry import get_skill_catalog


@dataclasses.dataclass
class EvidenceItem:
    evidence_id: str
    concept_type: str
    concept_id: str
    evidence_source_kind: str
    document_time: Optional[str]
    provenance: Dict[str, Any]
    event_time: Optional[str] = None
    report_type: Optional[str] = None
    source_institution: Optional[str] = None
    status: Optional[str] = None
    value: Any = None
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    local_code: Optional[str] = None
    specimen: Optional[str] = None
    method: Optional[str] = None
    evidence_type: Optional[str] = None


def _parse_dt(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return dt.datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _load_policy_skill(skill_id: str) -> Dict[str, Any]:
    catalog = {entry["id"]: entry for entry in get_skill_catalog()}
    entry = catalog.get(skill_id)
    if not entry:
        return {}
    return entry.get("payload") or {}


def extract_target_concepts(
    question: str,
    skill_id: str = "parsing.therapy_normalization",
) -> Tuple[List[str], Dict[str, Set[str]]]:
    """Scan *question* for therapy mentions using the normalization catalog.

    Returns ``(concept_ids, concept_search_terms)`` where *concept_search_terms*
    maps each found concept to the set of lowercase strings that should be used
    to locate it inside evidence-node text.
    """
    catalog = _load_policy_skill(skill_id)
    if not catalog:
        return [], {}

    abbreviations: Dict[str, str] = catalog.get("abbreviations") or {}
    class_mapping: Dict[str, Any] = catalog.get("class_mapping") or {}
    regimen_mapping: Dict[str, Any] = catalog.get("regimen_mapping") or {}

    # Collect the set of canonical drug names from class_mapping.
    known_drugs: Set[str] = set()
    for cls_info in class_mapping.values():
        for d in cls_info.get("drugs") or []:
            known_drugs.add(d)

    # Build a reverse-lookup: lowercase term → concept_id
    term_to_concept: Dict[str, str] = {}

    # Abbreviations that resolve to a known drug → concept is the drug
    for abbrev, expanded in abbreviations.items():
        if expanded in known_drugs:
            term_to_concept[abbrev.lower()] = expanded

    # Class aliases → concept is the class key
    for cls_key, cls_info in class_mapping.items():
        for alias in cls_info.get("aliases") or []:
            term_to_concept[alias.lower()] = cls_key

    # Drug names themselves → concept is the drug
    for drug in known_drugs:
        term_to_concept[drug.lower()] = drug

    # Regimen names → concept is the regimen
    for regimen_name in regimen_mapping:
        term_to_concept[regimen_name.lower()] = regimen_name

    # Scan question text longest-first (prefer specific matches)
    q_lower = question.lower()
    found_concepts: List[str] = []
    seen: Set[str] = set()

    for term in sorted(term_to_concept, key=len, reverse=True):
        concept = term_to_concept[term]
        if concept in seen:
            continue
        # Use word-boundary regex for short terms to avoid false positives
        if len(term) <= 4:
            if not re.search(r"\b" + re.escape(term) + r"\b", q_lower):
                continue
        else:
            if term not in q_lower:
                continue
        found_concepts.append(concept)
        seen.add(concept)

    if not found_concepts:
        return [], {}

    # Build concept_search_terms: for each concept, collect ALL terms that map
    # to it so we can find mentions in evidence text.
    # Also expand classes/regimens to include constituent drugs + their abbrevs.
    concept_search_terms: Dict[str, Set[str]] = {}

    # Helper: collect all abbreviations that resolve to a given drug name
    def _drug_terms(drug_name: str) -> Set[str]:
        terms: Set[str] = {drug_name.lower()}
        for ab, exp in abbreviations.items():
            if exp == drug_name:
                terms.add(ab.lower())
        return terms

    for concept in found_concepts:
        search: Set[str] = set()
        # Direct terms that map to this concept
        for t, c in term_to_concept.items():
            if c == concept:
                search.add(t)

        # Expand class → constituent drugs + their abbreviations
        if concept in class_mapping:
            for drug in class_mapping[concept].get("drugs") or []:
                search.update(_drug_terms(drug))

        # Expand regimen → constituent drugs + their abbreviations
        if concept in regimen_mapping:
            for drug in regimen_mapping[concept]:
                search.update(_drug_terms(drug))

        # If concept is a specific drug, include all its abbreviations
        if concept in known_drugs:
            search.update(_drug_terms(concept))

        concept_search_terms[concept] = search

    return found_concepts, concept_search_terms


def rank_evidence(
    evidence_items: List[EvidenceItem],
    question_type: str,
    policy_skill_id: str = "policy.temporal_authority",
    reference_date: Optional[str] = None,
) -> Tuple[List[EvidenceItem], Dict[str, Any]]:
    policy = _load_policy_skill(policy_skill_id)
    rules_block = policy.get("policy_rules") or {}
    ranking_cfg = rules_block.get("ranking") or {}
    report_type_priority = ranking_cfg.get("report_type_priority") or {}
    institution_priority = ranking_cfg.get("institution_priority") or {}
    recency_cfg = ranking_cfg.get("recency") or {}
    decay_days = recency_cfg.get("decay_days", 180) or 180
    primary_time_field = recency_cfg.get("primary_time_field") or "event_time"
    fallback_time_field = recency_cfg.get("fallback_time_field") or "document_time"
    additional_rules = rules_block.get("rules") or []

    # --- question_type_overrides: recency direction & decay toggles -----------
    qt_overrides = rules_block.get("question_type_overrides") or []
    sort_ascending = False
    disable_recency_decay = False
    for override in qt_overrides:
        when = override.get("when") or {}
        qtypes = when.get("question_types") or []
        if question_type not in qtypes:
            continue
        action = override.get("action") or {}
        if action.get("sort_order") == "ascending":
            sort_ascending = True
        if action.get("disable_recency_decay"):
            disable_recency_decay = True

    now = dt.datetime.utcnow()
    ref_dt = _parse_dt(reference_date) if reference_date else None
    trace: Dict[str, Any] = {"scores": {}, "overrides_applied": []}

    def _days_from_ref(ts: Optional[dt.datetime]) -> Optional[float]:
        if not ts or not ref_dt:
            return None
        return abs((ts - ref_dt).total_seconds()) / 86400.0

    def compute_score(item: EvidenceItem) -> float:
        score = 0.0
        if item.report_type:
            score += float(report_type_priority.get(item.report_type, 0))
        if item.source_institution:
            score += float(institution_priority.get(item.source_institution, 0))
        time_val = getattr(item, primary_time_field) or getattr(item, fallback_time_field)
        ts = _parse_dt(time_val)
        if ts and not disable_recency_decay:
            age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
            score += max(0.0, decay_days - age_days) / decay_days
        fired: List[str] = []
        for rule in additional_rules:
            when = rule.get("when") or {}
            qtypes = when.get("question_types") or []
            if qtypes and question_type not in qtypes and "*" not in qtypes:
                continue
            match = when.get("match") or {}
            matched = True
            for k, v in match.items():
                if getattr(item, k, None) != v:
                    matched = False
                    break
            if not matched:
                continue
            action = rule.get("action") or {}
            # Simple equality boost
            boost = action.get("boost_if") or {}
            if boost:
                target_field = list(boost.keys())[0]
                target_value = boost.get(target_field)
                if getattr(item, target_field, None) == target_value:
                    score += float(boost.get("add_score", 0))
                    fired.append(rule.get("rule_id", ""))

            # Reference-date window boost
            if "event_time_within_days_of_reference" in boost and ts and ref_dt:
                max_days = float(boost["event_time_within_days_of_reference"])
                if _days_from_ref(ts) is not None and _days_from_ref(ts) <= max_days:
                    score += float(boost.get("add_score", 0))
                    fired.append(rule.get("rule_id", ""))

            # Conditional penalty with newer evidence
            cond_pen = action.get("conditional_penalty") or {}
            if cond_pen and ref_dt:
                ref_recent_lab_days = cond_pen.get("reference_recent_lab_within_days")
                older_by = cond_pen.get("if_score_older_than_newer_by_days") or cond_pen.get("if_report_older_than_recent_lab_by_days")
                add_pen = cond_pen.get("add_score", 0)
                if ref_recent_lab_days is not None and older_by is not None:
                    # find newer evidence of same concept within window
                    for other in evidence_items:
                        if other is item:
                            continue
                        if other.concept_id != item.concept_id:
                            continue
                        ots = _parse_dt(getattr(other, primary_time_field) or getattr(other, fallback_time_field))
                        if not ots:
                            continue
                        if _days_from_ref(ots) is None or _days_from_ref(ots) > float(ref_recent_lab_days):
                            continue
                        if ts and (ts < ots) and ((ots - ts).total_seconds() / 86400.0) > float(older_by):
                            score += float(add_pen)
                            fired.append(rule.get("rule_id", ""))
                            break

            # Penalty if older than reference window
            pen_if = action.get("penalty_if") or {}
            if "event_time_older_than_days_from_reference" in pen_if and ts and ref_dt:
                max_days = float(pen_if["event_time_older_than_days_from_reference"])
                if _days_from_ref(ts) is not None and _days_from_ref(ts) > max_days:
                    score += float(pen_if.get("add_score", 0))
                    fired.append(rule.get("rule_id", ""))

            if action.get("apply_recency_decay") and ts:
                score -= max(0.0, (now - ts).total_seconds() / 86400.0) / decay_days
                fired.append(rule.get("rule_id", ""))
        trace["scores"][item.evidence_id] = {"score": score, "report_type": item.report_type, "fired_rules": fired}
        return score

    scored = [(compute_score(item), item) for item in evidence_items]
    if sort_ascending:
        # For first-occurrence / temporal_localization: prefer oldest evidence
        # Sort by event_time ascending, using score only as tiebreaker
        def _ts_key(tup: Tuple[float, EvidenceItem]) -> Tuple[float, float]:
            _score, _item = tup
            tv = getattr(_item, primary_time_field) or getattr(_item, fallback_time_field)
            parsed = _parse_dt(tv)
            epoch = parsed.timestamp() if parsed else float("inf")
            return (epoch, -_score)
        scored.sort(key=_ts_key)
        trace["overrides_applied"].append("sort_ascending")
    else:
        scored.sort(key=lambda tup: tup[0], reverse=True)
    ranked = [t[1] for t in scored]
    return ranked, trace


def _get_evidence_text(item: EvidenceItem) -> str:
    """Retrieve searchable text from an evidence item's provenance or value."""
    parts: List[str] = []
    if item.value and isinstance(item.value, str):
        parts.append(item.value)
    prov = item.provenance or {}
    for key in ("text", "snippet"):
        if prov.get(key):
            parts.append(str(prov[key]))
    return " ".join(parts)


def _apply_maintenance_rules(
    item: EvidenceItem,
    maintenance_rules: List[Dict[str, Any]],
) -> Optional[str]:
    """Check maintenance_therapy_rules against evidence text.

    Returns the overridden status string if a rule matches, else None.
    """
    text = _get_evidence_text(item).lower()
    if not text:
        return None
    for rule in maintenance_rules:
        keywords = rule.get("keywords") or []
        patterns = rule.get("patterns") or []
        action = rule.get("action") or {}
        override = action.get("override_status")
        if not override:
            continue
        # Keyword match (case-insensitive substring)
        for kw in keywords:
            if kw.lower() in text:
                return override
        # Regex pattern match
        for pat in patterns:
            try:
                if re.search(pat, text, re.IGNORECASE):
                    return override
            except re.error:
                continue
    return None


# Keywords indicating contraindication or cancellation
_CONTRAINDICATION_KEYWORDS = [
    "kontraindikation", "kontraindiziert", "nicht durchführbar",
    "abgelehnt", "nicht möglich", "nicht geeignet", "cancelled",
    "abgebrochen", "abgesetzt", "nicht toleriert", "unverträglich",
    "therapieabbruch",
]


def _check_contraindication(item: EvidenceItem) -> bool:
    """Return True if the evidence text indicates a contraindication or cancellation."""
    text = _get_evidence_text(item).lower()
    if not text:
        return False
    return any(kw in text for kw in _CONTRAINDICATION_KEYWORDS)


def _has_corroborating_administration(
    candidates: List[EvidenceItem],
    exclude_item: EvidenceItem,
) -> bool:
    """Check if any non-tumor-board evidence confirms administration (started/ongoing)."""
    for item in candidates:
        if item is exclude_item:
            continue
        if item.report_type == "tumor_board":
            continue
        status = (item.status or "").lower()
        if status in ("started", "ongoing"):
            return True
    return False


def derive_therapy_state(
    ranked_items: List[EvidenceItem],
    target_concept: Optional[str],
    policy_skill_id: str = "policy.plan_vs_administered",
) -> Tuple[str, str, List[str], Dict[str, Any]]:
    policy = _load_policy_skill(policy_skill_id)
    rules = policy.get("policy_rules") or {}
    status_priority = rules.get("status_priority") or {}
    recency_cfg = rules.get("recency") or {}
    report_type_bonus = rules.get("report_type_bonus") or {}
    decay_days = recency_cfg.get("decay_days", 365) or 365
    primary_time_field = recency_cfg.get("primary_time_field") or "event_time"
    fallback_time_field = recency_cfg.get("fallback_time_field") or "document_time"
    now = dt.datetime.utcnow()

    # Load maintenance therapy override rules
    maint_cfg = rules.get("maintenance_therapy_rules") or {}
    maint_rules = maint_cfg.get("rules") or []

    candidates = [
        item
        for item in ranked_items
        if item.concept_type == "therapy" and (target_concept is None or item.concept_id == target_concept)
    ]
    if not candidates:
        return "unknown", "low", [], {"reason": "no_matching_evidence"}

    def _effective_status(item: EvidenceItem) -> str:
        """Compute the effective status after all deterministic overrides."""
        status = (item.status or "unknown").lower()

        # 1. Contraindication check — hard override to "declined"
        if _check_contraindication(item):
            return "declined"

        # 2. Tumor board cap — never higher than "planned" on its own
        if item.report_type == "tumor_board" and status in ("started", "ongoing"):
            status = "planned"

        # 3. Maintenance therapy overrides
        if maint_rules:
            override = _apply_maintenance_rules(item, maint_rules)
            if override:
                status = override.lower()

        return status

    def score(item: EvidenceItem) -> float:
        effective = _effective_status(item)
        score_val = float(status_priority.get(effective, 0))
        if item.report_type:
            score_val += float(report_type_bonus.get(item.report_type, 0))
        tval = getattr(item, primary_time_field) or getattr(item, fallback_time_field)
        ts = _parse_dt(tval)
        if ts:
            age = max(0.0, (now - ts).total_seconds() / 86400.0)
            score_val += max(0.0, decay_days - age) / decay_days
        return score_val

    ranked_candidates = sorted(candidates, key=score, reverse=True)
    top = ranked_candidates[0]
    top_score = score(top)
    effective_top_status = _effective_status(top)

    # Tumor board corroboration: if the top item is tumor_board with "planned"
    # status, check if any non-tumor-board evidence confirms administration.
    # If corroborated, upgrade to "administered".
    tb_corroborated = False
    if top.report_type == "tumor_board" and effective_top_status == "planned":
        if _has_corroborating_administration(candidates, top):
            effective_top_status = "started"
            tb_corroborated = True

    # determinacy
    if effective_top_status in {"started", "stopped", "ongoing"}:
        determinacy = "high"
        state = "administered"
    elif effective_top_status == "declined":
        determinacy = "medium"
        state = "declined"
    elif effective_top_status == "planned":
        determinacy = "medium"
        state = "planned_only"
    else:
        determinacy = "low"
        state = "unknown"
    trace = {
        "top_score": top_score,
        "top_status": top.status,
        "effective_status": effective_top_status,
        "contraindication_detected": _check_contraindication(top),
        "tumor_board_corroborated": tb_corroborated,
        "maintenance_override": _apply_maintenance_rules(top, maint_rules) if maint_rules else None,
        "used_ids": [top.evidence_id],
    }
    return state, determinacy, [top.evidence_id], trace


def resolve_conflicts(
    candidate_claims: List[Dict[str, Any]],
    question_type: str,
    policy_skill_id: str = "policy.contradiction_resolution",
) -> Tuple[str, Any, Dict[str, Any]]:
    policy = _load_policy_skill(policy_skill_id)
    rules = policy.get("policy_rules") or {}
    resolution_map = rules.get("resolution_by_question_type") or {}
    # Support both old key (strict_abstain) and new key (consensus_or_abstain)
    consensus_or_abstain = set(
        resolution_map.get("consensus_or_abstain", [])
        or resolution_map.get("strict_abstain", [])
    )
    select_highest = set(resolution_map.get("select_highest", []))
    determinacy_priority = rules.get("determinacy_priority") or {}

    if not candidate_claims:
        return "abstain", {"reason": "no_claims"}, {"applied_rule": "no_claims"}
    if len(candidate_claims) == 1:
        claim = candidate_claims[0]
        return "select", claim, {"applied_rule": "single_claim"}

    values = {c.get("value") for c in candidate_claims}
    if question_type in consensus_or_abstain:
        if len(values) > 1:
            # Genuine disagreement on direction → abstain
            ids = []
            for c in candidate_claims:
                ids.extend(c.get("supporting_evidence_ids", []))
            return "abstain", {"reason": "conflict", "supporting_evidence_ids": ids}, {"applied_rule": "consensus_conflict"}
        else:
            # All claims agree on the same value → select highest determinacy
            def _det(claim: Dict[str, Any]) -> int:
                return int(determinacy_priority.get(str(claim.get("determinacy") or "low"), 0))
            best = sorted(candidate_claims, key=_det, reverse=True)[0]
            return "select", best, {"applied_rule": "consensus_select"}

    def det_score(claim: Dict[str, Any]) -> int:
        det = str(claim.get("determinacy") or "low")
        return int(determinacy_priority.get(det, 0))

    sorted_claims = sorted(candidate_claims, key=det_score, reverse=True)
    if question_type in select_highest:
        return "select", sorted_claims[0], {"applied_rule": "select_highest"}

    # default: present conflict
    support_ids: List[str] = []
    for claim in candidate_claims:
        support_ids.extend(claim.get("supporting_evidence_ids", []))
    return "present_conflict", {"claims": candidate_claims, "supporting_evidence_ids": support_ids}, {
        "applied_rule": "present_conflict"
    }


def apply(
    selected_policy_skills: List[str],
    evidence_items: List[EvidenceItem],
    question_type: str,
    target_concepts: Optional[List[str]] = None,
    reference_date: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply ranking, state derivation, and conflict resolution."""
    trace: Dict[str, Any] = {"skills": selected_policy_skills}
    ranked, trace_rank = rank_evidence(evidence_items, question_type, reference_date=reference_date)
    trace.update({"rank": trace_rank, "ranked_ids": [item.evidence_id for item in ranked]})

    candidate_claims: List[Dict[str, Any]] = []
    target_concepts = target_concepts or []
    if target_concepts:
        for concept in target_concepts:
            state, det, support_ids, tstate = derive_therapy_state(ranked, concept)
            candidate_claims.append(
                {
                    "claim_id": f"{concept}:{state}",
                    "concept_id": concept,
                    "value": state,
                    "determinacy": det,
                    "supporting_evidence_ids": support_ids,
                }
            )
            trace.setdefault("therapy_state", {})[concept] = tstate
    policy_result: Dict[str, Any]
    if candidate_claims:
        action, final_claim, trace_res = resolve_conflicts(candidate_claims, question_type)
        trace["resolution"] = trace_res
        policy_result = {
            "resolution_action": action,
            "final_claim": final_claim,
        }
    else:
        policy_result = {
            "resolution_action": "select",
            "final_claim": None,
        }
    return policy_result, trace


def normalise_nodes_to_evidence(
    nodes: List[Dict[str, Any]],
    target_concepts: Optional[List[str]] = None,
    concept_search_terms: Optional[Dict[str, Set[str]]] = None,
) -> List[EvidenceItem]:
    """Convert context nodes into EvidenceItem objects.

    When *concept_search_terms* is provided, each node's text is scanned for
    therapy-concept mentions so that ``concept_type`` and ``concept_id`` are
    set to ``"therapy"`` / the matching concept rather than the default
    ``"other"``.
    """
    results: List[EvidenceItem] = []
    for node in nodes:
        evidence_id = node.get("citation_id") or node.get("section_id") or node.get("assessment_id") or ""
        if not evidence_id:
            continue
        report_type = node.get("report_type")
        concept_type = "other"
        concept_id = node.get("section_name") or "unknown"

        if concept_search_terms:
            text_to_search = (
                (node.get("snippet") or "") + " " + (node.get("text") or "")
            ).lower()
            for tc, search_terms in concept_search_terms.items():
                if any(term in text_to_search for term in search_terms):
                    concept_type = "therapy"
                    concept_id = tc
                    break
        evidence_source_kind = "report" if report_type else ("lab_tool" if node.get("lab_id") else "other")
        # Heuristic evidence_type
        evidence_type = node.get("evidence_type")
        section_lower = ((node.get("section_name") or "") + " " + (node.get("snippet") or "") + " " + (node.get("text") or "")).lower()
        if not evidence_type:
            if any(tok in section_lower for tok in ["r-iss", "riss", "r2-iss", "iss ", "ipss-r", "hct-ci", "ecog"]):
                evidence_type = "score"
            elif evidence_source_kind == "lab_tool" or (report_type and report_type in ["Albumin", "LDH", "Beta2-Mikroglobulin", "β2-Mikroglobulin", "Beta-2-Mikroglobulin"]):
                evidence_type = "lab"
        document_time = node.get("report_date") or node.get("date")
        event_time = node.get("event_date") or None
        status = node.get("status")
        source_institution = node.get("source_institution")
        provenance = {
            "report_id": node.get("report_id"),
            "section_id": node.get("section_id"),
            "assessment_id": node.get("assessment_id"),
            "span": node.get("span"),
            "text": node.get("text"),
            "snippet": node.get("snippet"),
        }
        value = node.get("value")
        unit = node.get("unit")
        reference_range = node.get("reference_range")
        local_code = node.get("local_code")
        specimen = node.get("specimen")
        method = node.get("method")

        results.append(
            EvidenceItem(
                evidence_id=str(evidence_id),
                concept_type=str(concept_type),
                concept_id=str(concept_id),
                evidence_source_kind=str(evidence_source_kind),
                evidence_type=str(evidence_type) if evidence_type else None,
                document_time=str(document_time) if document_time else None,
                event_time=str(event_time) if event_time else None,
                report_type=str(report_type) if report_type else None,
                source_institution=str(source_institution) if source_institution else None,
                status=str(status) if status else None,
                provenance={k: v for k, v in provenance.items() if v is not None},
                value=value,
                unit=unit,
                reference_range=reference_range,
                local_code=local_code,
                specimen=specimen,
                method=method,
            )
        )
    return results
