"""Lab catalog resolver that maps free-text queries to canonical lab IDs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import sqlite3
LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[lab_catalog] %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

UMLAUT_MAP = str.maketrans({
    "ä": "ae",
    "ö": "oe",
    "ü": "ue",
    "ß": "ss",
})


def normalise_label(text: str | None) -> str:
    """Return a normalized, ASCII-friendly version of a lab label."""
    if not text:
        return ""
    lowered = str(text).strip().lower().translate(UMLAUT_MAP)
    chunks: List[str] = []
    for char in lowered:
        if char.isalnum() or char.isspace():
            chunks.append(char)
        else:
            chunks.append(" ")
    collapsed = "".join(chunks)
    return " ".join(collapsed.split())


def collapsed_form(text: str | None) -> str:
    return normalise_label(text).replace(" ", "")


def trigram_set(value: str) -> Set[str]:
    norm = normalise_label(value)
    if not norm:
        return set()
    if len(norm) < 3:
        return {norm}
    return {norm[i : i + 3] for i in range(len(norm) - 2)}


def trigram_similarity(a: str, b: str) -> float:
    set_a = trigram_set(a)
    set_b = trigram_set(b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union else 0.0


@dataclass
class LabCandidate:
    lab_id: str
    code: str
    code_norm: str
    display_name: str
    score: float
    match_features: Dict[str, float] = field(default_factory=dict)
    example_aliases: Sequence[str] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, object]:
        return {
            "lab_id": self.lab_id,
            "code": self.code,
            "code_norm": self.code_norm,
            "display_name": self.display_name,
            "score": round(self.score, 4),
            "example_aliases": list(self.example_aliases),
        }


@dataclass
class ResolutionResult:
    status: str
    reason: str
    selected: List[LabCandidate]
    candidates: List[LabCandidate]
    debug: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "selected": [cand.to_dict() for cand in self.selected],
            "candidates": [cand.to_dict() for cand in self.candidates],
            "debug": self.debug,
        }


@dataclass
class LabCatalogEntry:
    lab_id: str
    code: str
    code_norm: str
    display_name: str
    variants: Sequence[str]
    search_terms: Sequence[str]
    aliases: Sequence[str]
    alias_norms: Sequence[str]


class LabCatalogResolver:
    """Resolve free-text lab queries to canonical lab IDs."""

    EXACT_MATCH_WEIGHT = 1.2
    ALIAS_EXACT_WEIGHT = 1.0
    PREFIX_WEIGHT = 0.8
    SUBSTRING_WEIGHT = 0.6
    TOKEN_INTERSECT_WEIGHT = 0.4
    TRIGRAM_WEIGHT = 0.5
    MIN_SCORE = 0.4
    DOMINANT_THRESHOLD = 1.2
    DOMINANT_GAP = 0.25
    AMBIGUOUS_BAND = 0.85

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Lab catalog database not found: {self.db_path}")
        self.entries: Dict[str, LabCatalogEntry] = {}
        self.alias_index: Dict[str, Set[str]] = {}
        self._load_catalog()

    def _load_catalog(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute("SELECT lab_id, code, code_norm, display_name, variants_json, search_terms_json FROM lab_catalog")
            catalog_rows = cur.fetchall()
            alias_rows = conn.execute("SELECT alias, alias_norm, lab_id, code FROM lab_aliases").fetchall()
        finally:
            conn.close()
        alias_map: Dict[str, List[Tuple[str, str]]] = {}
        for row in alias_rows:
            lab_id = row["lab_id"]
            alias_map.setdefault(lab_id, []).append((row["alias"], row["alias_norm"]))
            if row["alias_norm"]:
                self.alias_index.setdefault(row["alias_norm"], set()).add(lab_id)
        for row in catalog_rows:
            lab_id = row["lab_id"]
            variants = json.loads(row["variants_json"] or "[]")
            search_terms = json.loads(row["search_terms_json"] or "[]")
            aliases = [alias for alias, _ in alias_map.get(lab_id, [])] or list(variants)
            alias_norms = [norm for _, norm in alias_map.get(lab_id, []) if norm] or [normalise_label(item) for item in aliases]
            entry = LabCatalogEntry(
                lab_id=lab_id,
                code=row["code"],
                code_norm=row["code_norm"],
                display_name=row["display_name"] or row["code"],
                variants=variants,
                search_terms=search_terms,
                aliases=aliases,
                alias_norms=alias_norms,
            )
            self.entries[lab_id] = entry

    def aliases_for_lab_id(self, lab_id: str) -> Sequence[str]:
        entry = self.entries.get(lab_id)
        return entry.aliases if entry else []

    def resolve(self, query: str, *, top_k: int = 25, hard_cap: int = 10) -> ResolutionResult:
        query_norm = normalise_label(query)
        query_collapsed = query_norm.replace(" ", "")
        tokens = set(query_norm.split())
        if not query_norm:
            LOGGER.warning("lab-query dropped; empty after normalization. Raw='%s'", query)
            return ResolutionResult(
                status="no_match",
                reason="Query was empty after normalization.",
                selected=[],
                candidates=[],
                debug={},
            )
        LOGGER.debug(
            "Resolving lab query: raw='%s' norm='%s' collapsed='%s' tokens=%s",
            query,
            query_norm,
            query_collapsed,
            sorted(tokens),
        )
        candidates = self._score_candidates(query_norm, query_collapsed, tokens)
        if not candidates:
            LOGGER.info("No lab catalog matches for query '%s' (norm='%s')", query, query_norm)
            return ResolutionResult(
                status="no_match",
                reason="No catalog entries matched the query.",
                selected=[],
                candidates=[],
                debug={},
            )
        candidates.sort(key=lambda cand: cand.score, reverse=True)
        candidates = candidates[:max(top_k, hard_cap)]
        top_score = candidates[0].score
        if top_score < self.MIN_SCORE:
            LOGGER.info(
                "Top score %.3f below threshold for query '%s'; returning no_match", top_score, query_norm
            )
            return ResolutionResult(
                status="no_match",
                reason="Top match score below threshold.",
                selected=[],
                candidates=candidates[: hard_cap],
                debug={"top_score": top_score},
            )
        selected: List[LabCandidate]
        status: str
        reason: str
        if len(candidates) == 1 or (
            top_score >= self.DOMINANT_THRESHOLD
            and (len(candidates) == 1 or top_score - candidates[1].score >= self.DOMINANT_GAP)
        ):
            status = "confident"
            reason = "Top candidate dominated the scoring thresholds."
            selected = [candidates[0]]
        else:
            band_threshold = top_score * self.AMBIGUOUS_BAND
            selection_pool = [cand for cand in candidates if cand.score >= band_threshold]
            status = "ambiguous"
            reason = "Multiple candidates scored similarly; manual disambiguation recommended."
            deduped: Dict[str, LabCandidate] = {}
            for cand in selection_pool:
                key = cand.code_norm
                if key not in deduped:
                    deduped[key] = cand
            selected = list(deduped.values())[:hard_cap]
        LOGGER.debug(
            "Resolver outcome for query '%s': status=%s top_score=%.3f selected=%s",
            query_norm,
            status,
            top_score,
            [cand.code for cand in selected],
        )
        return ResolutionResult(
            status=status,
            reason=reason,
            selected=selected,
            candidates=candidates[:hard_cap],
            debug={"top_score": top_score},
        )

    def _score_candidates(
        self,
        query_norm: str,
        query_collapsed: str,
        tokens: Set[str],
    ) -> List[LabCandidate]:
        candidates: List[LabCandidate] = []
        for entry in self.entries.values():
            features: Dict[str, float] = {}
            score = 0.0
            if query_norm == entry.code_norm:
                features["code_exact"] = self.EXACT_MATCH_WEIGHT
                score += self.EXACT_MATCH_WEIGHT
            if query_norm in entry.alias_norms:
                features["alias_exact"] = self.ALIAS_EXACT_WEIGHT
                score += self.ALIAS_EXACT_WEIGHT
            if len(query_norm) >= 3:
                if entry.code_norm.startswith(query_norm):
                    features["code_prefix"] = self.PREFIX_WEIGHT
                    score += self.PREFIX_WEIGHT
                elif any(alias.startswith(query_norm) for alias in entry.alias_norms):
                    features["alias_prefix"] = self.PREFIX_WEIGHT * 0.9
                    score += self.PREFIX_WEIGHT * 0.9
            if query_collapsed:
                collapsed_code = entry.code_norm.replace(" ", "")
                if query_collapsed in collapsed_code or collapsed_code in query_collapsed:
                    features["substring"] = self.SUBSTRING_WEIGHT
                    score += self.SUBSTRING_WEIGHT
                elif any(query_collapsed in alias.replace(" ", "") for alias in entry.alias_norms):
                    features["alias_substring"] = self.SUBSTRING_WEIGHT * 0.9
                    score += self.SUBSTRING_WEIGHT * 0.9
            shared_tokens = tokens.intersection(entry.search_terms)
            if shared_tokens:
                contribution = self.TOKEN_INTERSECT_WEIGHT * min(1.0, len(shared_tokens) / max(1, len(tokens)))
                features["token_overlap"] = round(contribution, 4)
                score += contribution
            trigram_score = trigram_similarity(query_norm, entry.code_norm)
            if trigram_score > 0:
                contribution = self.TRIGRAM_WEIGHT * trigram_score
                features["trigram"] = round(contribution, 4)
                score += contribution
            if score <= 0:
                continue
            LOGGER.debug(
                "candidate '%s' (lab_id=%s) scored %.3f with features=%s",
                entry.code,
                entry.lab_id,
                score,
                features,
            )
            candidates.append(
                LabCandidate(
                    lab_id=entry.lab_id,
                    code=entry.code,
                    code_norm=entry.code_norm,
                    display_name=entry.display_name,
                    score=score,
                    match_features=features,
                    example_aliases=entry.variants[:3],
                )
            )
        return candidates
