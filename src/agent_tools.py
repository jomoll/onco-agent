"""
Lightweight tools that the local agents can call.

These mirror the developer tooling from the legacy project but keep the
implementations simple (and dependency-light) so they are safe to use in tests
or demos. The goal is to validate agent behaviour before wiring in the full
clinical RAG pipeline.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from rank_bm25 import BM25Okapi

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
except Exception:
    _SentenceTransformer = None  # type: ignore

from .report_type_synonyms import REPORT_TYPE_SYNONYMS
from .llm_output import strip_hidden_reasoning
from .toolkit import BaseTool, ToolMetadata, ToolOutput

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

__all__ = [
    "ReportsRAGTool",
    "FullContextTool",
    "LabQueryTool",
    "load_default_tools",
    "fetch_patient_lab_keys",
    "ISSScoreTool",
    "RISSScoreTool",
    "R2ISSScoreTool",
    "IPSSRScoreTool",
    "HCTCITool",
]


# ---------------------------------------------------------------------------
# Hybrid retrieval helpers (char-ngram BM25 + dense embeddings)
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    return (text or "").lower()


def _char_ngrams(text: str, n: int = 3) -> List[str]:
    cleaned = re.sub(r"[^0-9a-zA-ZäöüÄÖÜß]", " ", _normalize_text(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.replace(" ", "_")
    if len(cleaned) < n:
        return [cleaned] if cleaned else []
    return [cleaned[i : i + n] for i in range(len(cleaned) - n + 1)]


class _BM25Index:
    """Character n-gram BM25 index for hybrid retrieval."""

    def __init__(self, k1: float = 1.2, b: float = 0.75, ngram: int = 3) -> None:
        self.k1 = k1
        self.b = b
        self.ngram = ngram
        self.df: Counter[str] = Counter()
        self.doc_len: List[int] = []
        self.avgdl = 0.0
        self.N = 0
        self.postings: Dict[str, Dict[int, int]] = defaultdict(dict)

    def add_docs(self, docs: List[str]) -> None:
        self.N = len(docs)
        for doc_id, text in enumerate(docs):
            toks = _char_ngrams(text, self.ngram)
            tf = Counter(toks)
            self.doc_len.append(sum(tf.values()))
            for token, cnt in tf.items():
                self.df[token] += 1
                self.postings[token][doc_id] = cnt
        self.avgdl = (sum(self.doc_len) / max(1, self.N)) if self.N else 0.0

    def score(self, query: str, mask: "Optional[np.ndarray]" = None) -> "np.ndarray":
        toks = _char_ngrams(query, self.ngram)
        unique_q = set(toks)
        idxs = np.nonzero(mask)[0] if mask is not None else np.arange(self.N)
        scores = np.zeros(len(idxs), dtype=np.float32)
        for j, doc_id in enumerate(idxs):
            dl = self.doc_len[doc_id]
            denom_norm = self.k1 * (1 - self.b + self.b * (dl / self.avgdl)) if self.avgdl > 0 else self.k1
            s = 0.0
            for t in unique_q:
                tf = self.postings.get(t, {}).get(doc_id, 0)
                if tf == 0:
                    continue
                df_t = self.df.get(t, 0)
                idf = math.log((self.N - df_t + 0.5) / (df_t + 0.5) + 1.0) if self.N else 0.0
                num = tf * (self.k1 + 1.0)
                s += idf * (num / (tf + denom_norm))
            scores[j] = s
        return scores


def _minmax_normalize(scores: "np.ndarray") -> "np.ndarray":
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)


class ReportsRAGTool(BaseTool):
    """Lightweight RAG tool backed by the anonymised SQLite database."""

    DB_PATH = PROJECT_ROOT / "src/database/synthetic.sqlite"
    DEFAULT_PATIENT_ID = "patient_001"
    DEFAULT_TOP_K = 5
    REPORT_TYPES = [
        "doctor_letter",
        "consult",
        "radiology",
        "pathology",
        "tumor_board",
        "cardiology",
        "cytology",
        "flow",
        "history",
    ]
    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        top_k: int = DEFAULT_TOP_K,
        patient_id: str = DEFAULT_PATIENT_ID,
        use_hybrid: bool = False,
        hybrid_alpha: float = 0.5,
        embed_model: str = "models/distiluse-base-multilingual-cased-v2",
    ) -> None:
        self.db_path = Path(db_path) if db_path else self.DB_PATH
        self.top_k = top_k
        normalized_patient = str(patient_id).strip() if patient_id else ""
        self.patient_id = normalized_patient or self.DEFAULT_PATIENT_ID

        # Hybrid retrieval settings
        if use_hybrid and (_SentenceTransformer is None or np is None):
            logger.warning(
                "use_hybrid=True but sentence-transformers/numpy unavailable; falling back to BM25-only"
            )
            use_hybrid = False
        self.use_hybrid = use_hybrid
        self.hybrid_alpha = hybrid_alpha
        self._embed_model_name = embed_model

        # Lazy-loaded cache slots for hybrid retrieval
        self._embedder = None
        self._cached_embeddings = None
        self._cached_bm25_index: Optional[_BM25Index] = None
        self._cached_sections: Optional[List[Dict[str, Any]]] = None
        self._cache_key: Optional[Tuple] = None

        self._metadata = ToolMetadata(
            name="retrieve_reports",
            description=(
                "Retrieve clinical report sections for a patient using a BM25 keyword search "
                "over the anonymised SQLite database. Supports optional filtering by report_type "
                "and explicit temporal scoping via the time_scope argument ('all', 'latest', 'date', 'range'). "
                "Valid report_type values include: "
                "'doctor_letter', "
                "'consult', "
                "'radiology', "
                "'pathology', "
                "'tumor_board', "
                "'cardiology', "
                "'cytology', "
                "'flow', and "
                "'history'. "
                "Multiple report types can be provided as a list."
            ),
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    # -------------------- hybrid retrieval helpers --------------------

    def _ensure_embedder(self) -> Any:
        """Lazy-load the SentenceTransformer model on first hybrid call."""
        if self._embedder is None:
            t0 = time.monotonic()
            self._embedder = _SentenceTransformer(self._embed_model_name)
            logger.info(
                "Loaded embedding model %s in %.1fs",
                self._embed_model_name,
                time.monotonic() - t0,
            )
        return self._embedder

    def _build_cache(
        self,
        sections: List[Dict[str, Any]],
        cache_key: Tuple,
    ) -> None:
        """Pre-compute dense embeddings and char-ngram BM25 index for *all* sections."""
        texts = [s.get("section_content") or "" for s in sections]
        embedder = self._ensure_embedder()
        self._cached_embeddings = np.array(
            embedder.encode(texts, normalize_embeddings=True)
        )
        bm25_idx = _BM25Index(k1=1.2, b=0.75, ngram=3)
        bm25_idx.add_docs(texts)
        self._cached_bm25_index = bm25_idx
        self._cached_sections = sections
        self._cache_key = cache_key
        logger.debug(
            "Built hybrid cache for key=%s  (%d sections)", cache_key, len(sections)
        )

    # ------------------------------------------------------------------

    def __call__(  # type: ignore[override]
        self,
        query: str | None = None,
        report_type: Union[str, Sequence[str], None] = None,
        report_date: str | None = None,
        time_scope: str | None = None,
        date_exact: str | None = None,
        date_start: str | None = None,
        date_end: str | None = None,
        patient_id: str | None = None,
        input: Any | None = None,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> ToolOutput:
        scoped_kwargs = dict(kwargs)
        if time_scope is not None:
            scoped_kwargs["time_scope"] = time_scope
        if date_exact is not None:
            scoped_kwargs["date_exact"] = date_exact
        if date_start is not None:
            scoped_kwargs["date_start"] = date_start
        if date_end is not None:
            scoped_kwargs["date_end"] = date_end

        payload = self._coerce_payload(
            query=query,
            report_type=report_type,
            report_date=report_date,
            patient_id=patient_id,
            input_value=input,
            extra_kwargs=scoped_kwargs,
        )

        sections = self._fetch_sections(
            patient_id=payload["patient_id"],
            report_type=payload.get("report_type"),
        )

        if self.use_hybrid:
            rt = payload.get("report_type")
            cache_key = (
                payload["patient_id"],
                tuple(rt) if isinstance(rt, (list, tuple)) else (rt,),
            )
            if self._cache_key != cache_key:
                self._build_cache(sections, cache_key)

        sections = self._filter_sections_by_scope(
            sections,
            payload.get("time_scope"),
            payload.get("date_exact"),
            payload.get("date_start"),
            payload.get("date_end"),
        )

        matches = self._rank_sections(
            sections=sections,
            query=payload["query"],
            top_k=top_k or payload.get("top_k") or self.top_k,
        )

        if not matches:
            logger.warning(
                "No report sections found for patient %s (report_type=%s, scope=%s)",
                payload.get("patient_id"),
                payload.get("report_type"),
                payload.get("time_scope") or "all",
            )

        summary = self._format_response(matches, payload)
        result = {
            "status": "ok",
            "summary": summary,
            "context_nodes": matches,
            "data": {
                "query": payload["query"],
                "report_type": payload.get("report_type"),
                "report_date": payload.get("report_date"),
                "patient_id": payload["patient_id"],
                "top_k": top_k or payload.get("top_k") or self.top_k,
                "time_scope": payload.get("time_scope"),
                "date_exact": payload.get("date_exact"),
                "date_start": payload.get("date_start"),
                "date_end": payload.get("date_end"),
                "matches": matches,
            },
        }

        return ToolOutput(
            tool_name=self._metadata.name,
            content=summary,
            raw_input={"kwargs": payload},
            raw_output=result,
        )

    def _coerce_payload(
        self,
        *,
        query: str | None,
        report_type: Union[str, Sequence[str], None],
        report_date: str | None,
        patient_id: str | None,
        input_value: Any,
        extra_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized_patient = str(patient_id or self.patient_id or "").strip()
        if not normalized_patient:
            raise ValueError("ReportsRAGTool requires a patient_id.")

        payload: Dict[str, Any] = {
            "patient_id": normalized_patient,
            "report_type": report_type,
            "report_date": report_date,
        }

        for candidate in (input_value, extra_kwargs.get("input")):
            if not candidate:
                continue
            parsed = self._parse_json(candidate)
            if isinstance(parsed, dict):
                payload.update({k: v for k, v in parsed.items() if v not in (None, "")})

        payload.update(
            {
                k: v
                for k, v in extra_kwargs.items()
                if k
                in {
                    "query",
                    "report_type",
                    "report_date",
                    "patient_id",
                    "top_k",
                    "time_scope",
                    "date_exact",
                    "date_start",
                    "date_end",
                }
            }
        )

        if query and "query" not in payload:
            payload["query"] = query

        if "query" not in payload or not str(payload["query"]).strip():
            raise ValueError("missing a required argument: 'query'")

        # LLMs sometimes pass query as a list of keywords; join into a single string.
        raw_query = payload["query"]
        if isinstance(raw_query, (list, tuple)):
            payload["query"] = " ".join(str(t) for t in raw_query).strip()
        else:
            payload["query"] = str(raw_query).strip()
        payload["patient_id"] = str(payload.get("patient_id") or normalized_patient)

        if payload.get("report_type") not in (None, ""):
            normalized_types = self._normalize_report_types(payload["report_type"])
            if not normalized_types:
                payload["report_type"] = None
            elif len(normalized_types) == 1:
                payload["report_type"] = normalized_types[0]
            else:
                payload["report_type"] = normalized_types
        else:
            payload["report_type"] = None

        if payload.get("report_date") in (None, ""):
            payload["report_date"] = None
        else:
            payload["report_date"] = str(payload["report_date"]).strip()
        scope, norm_exact, norm_start, norm_end = self._parse_date_scope(
            scope_raw=payload.get("time_scope"),
            report_date_raw=payload.get("report_date"),
            date_exact_raw=payload.get("date_exact"),
            date_start_raw=payload.get("date_start"),
            date_end_raw=payload.get("date_end"),
        )
        payload["time_scope"] = scope
        payload["date_exact"] = norm_exact
        payload["date_start"] = norm_start
        payload["date_end"] = norm_end

        if payload.get("top_k") in (None, ""):
            payload.pop("top_k", None)

        return payload

    def _normalize_report_types(self, value: Union[str, Sequence[str]]) -> List[str]:
        collected: List[str] = []

        def _handle(item: Any) -> None:
            if item is None:
                return

            if isinstance(item, str):
                text = item.strip()
                if not text:
                    return

                lowered = text.lower()
                if lowered in {"all", "any", "*"}:
                    return

                parsed = self._try_parse_literal(text)
                if parsed is not None and not (isinstance(parsed, str) and parsed.strip() == text):
                    _handle(parsed)
                    return

                if "," in text:
                    parts = [part.strip() for part in text.split(",")]
                    for part in parts:
                        if part:
                            _handle(part)
                    return

                canonical, adjusted = self._resolve_report_type(text)
                if adjusted:
                    logger.debug("Normalised report_type '%s' -> '%s'", text, canonical)
                if canonical not in collected:
                    collected.append(canonical)
                return

            if isinstance(item, (list, tuple, set)):
                for sub in item:
                    _handle(sub)
                return

            parsed = self._try_parse_literal(str(item))
            if parsed is not None and parsed is not item:
                _handle(parsed)
                return

            raise ValueError(
                f"Unsupported report_type value: {item!r}. Expected a string or sequence of strings."
            )

        _handle(value)
        return collected

    def _resolve_report_type(self, raw: str) -> Tuple[str, bool]:
        lowered = raw.strip().lower()
        lookup = getattr(self, "_report_type_lookup", None)
        if lookup is None:
            lookup = {name.lower(): name for name in self.REPORT_TYPES}
            self._report_type_lookup = lookup
        if lowered in lookup:
            canonical = lookup[lowered]
            return canonical, canonical != raw

        compact_map = getattr(self, "_report_type_compact_lookup", None)
        if compact_map is None:
            compact_map = {self._compact_report_token(name): name for name in self.REPORT_TYPES}
            self._report_type_compact_lookup = compact_map
        compact = self._compact_report_token(lowered)
        if compact in compact_map:
            return compact_map[compact], True

        synonym_map = getattr(self, "_report_type_synonyms", None)
        if synonym_map is None:
            synonym_map = {key: value for key, value in REPORT_TYPE_SYNONYMS.items()}
            synonym_map.update({self._compact_report_token(key): value for key, value in REPORT_TYPE_SYNONYMS.items()})
            self._report_type_synonyms = synonym_map
        canonical = synonym_map.get(lowered) or synonym_map.get(compact)
        if canonical:
            return canonical, True

        raise ValueError(
            f"report_type '{raw}' is not supported. Known types: {', '.join(self.REPORT_TYPES)}"
        )

    @staticmethod
    def _compact_report_token(text: str) -> str:
        return re.sub(r"[\s_\-]", "", text or "")

    @staticmethod
    def _try_parse_literal(text: str) -> Any:
        try:
            return ast.literal_eval(text)
        except Exception:
            return None

    def _fetch_sections(
        self,
        *,
        patient_id: str,
        report_type: Union[str, Sequence[str], None],
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        conditions = ["rs.patient_id = ?"]
        params: List[Any] = [patient_id]

        if report_type:
            if isinstance(report_type, (list, tuple, set)):
                valid_types = [rtype for rtype in report_type if rtype]
                if valid_types:
                    placeholders = ", ".join(["?"] * len(valid_types))
                    conditions.append(f"rs.report_type IN ({placeholders})")
                    params.extend(valid_types)
            else:
                conditions.append("rs.report_type = ?")
                params.append(report_type)

        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT
                rs.section_id,
                rs.report_id,
                rs.section_name,
                rs.section_content,
                rs.report_type,
                rs.report_date,
                rs.patient_id
            FROM report_sections rs
            WHERE {where_clause}
            ORDER BY rs.report_date DESC, rs.section_order ASC
            LIMIT ?
        """
        params.append(limit)

        cursor.execute(sql, params)
        rows = [dict(row) for row in cursor.fetchall()]
        for section in rows:
            if not section.get("report_date"):
                derived = self._derive_report_date(section.get("report_id") or "")
                if derived:
                    section["report_date"] = derived
            normalized = self._normalize_report_date(section.get("report_date"))
            if normalized:
                section["_normalized_report_date"] = normalized
        conn.close()
        return rows

    def _filter_sections_by_scope(
        self,
        sections: List[Dict[str, Any]],
        scope: Optional[str],
        date_exact: Optional[str],
        date_start: Optional[str],
        date_end: Optional[str],
    ) -> List[Dict[str, Any]]:
        if not sections:
            return []

        normalized_scope = (scope or "all").lower()
        if normalized_scope == "all":
            return sections

        def _section_date(section: Dict[str, Any]) -> Optional[str]:
            value = section.get("_normalized_report_date")
            if value:
                return value
            normalized = self._normalize_report_date(section.get("report_date"))
            if normalized:
                section["_normalized_report_date"] = normalized
            return normalized

        if normalized_scope == "latest":
            dated_sections = [section for section in sections if _section_date(section)]
            if not dated_sections:
                return sections
            latest_value = max(section["_normalized_report_date"] for section in dated_sections if section.get("_normalized_report_date"))
            return [section for section in dated_sections if section.get("_normalized_report_date") == latest_value]

        if normalized_scope == "date" and date_exact:
            return [section for section in sections if _section_date(section) == date_exact]

        if normalized_scope == "range" and date_start and date_end:
            filtered: List[Dict[str, Any]] = []
            for section in sections:
                normalized = _section_date(section)
                if not normalized:
                    continue
                if date_start <= normalized <= date_end:
                    filtered.append(section)
            return filtered

        return sections

    def _parse_date_scope(
        self,
        *,
        scope_raw: Any,
        report_date_raw: Any,
        date_exact_raw: Any,
        date_start_raw: Any,
        date_end_raw: Any,
    ) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
        def _normalize_scope_token(value: Any) -> str:
            if value is None:
                return ""
            text = str(value).strip().lower()
            if not text:
                return ""
            mapping = {
                "all": {"all", "history", "entire", "complete", "any", "*"},
                "latest": {"latest", "recent", "current", "most_recent"},
                "date": {"date", "single", "exact", "on_date", "day"},
                "range": {"range", "window", "between", "timeframe", "interval"},
            }
            for normalized, aliases in mapping.items():
                if text in aliases:
                    return normalized
            if text in {"all", "latest", "date", "range"}:
                return text
            return ""

        def _stringify(value: Any) -> Optional[str]:
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        def _normalize_date(value: Any) -> Optional[str]:
            return self._normalize_report_date(value)

        def _infer_scope_from_literal(literal: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
            text = literal.strip()
            lowered = text.lower()
            if not text:
                return "", None, None, None
            if lowered in {"all", "*", "history"}:
                return "all", None, None, None
            if lowered in {"latest", "recent", "current", "now"}:
                return "latest", None, None, None

            if ".." in text:
                left, right = text.split("..", 1)
                start_norm = _normalize_date(left)
                end_norm = _normalize_date(right)
                if start_norm and end_norm:
                    if end_norm < start_norm:
                        start_norm, end_norm = end_norm, start_norm
                    return "range", None, start_norm, end_norm
                raise ValueError(
                    "report_date ranges must include valid start and end dates (YYYY-MM-DD or DD.MM.YYYY)."
                )

            for keyword in (" to ", " bis "):
                if keyword in lowered:
                    idx = lowered.index(keyword)
                    left = text[:idx]
                    right = text[idx + len(keyword) :]
                    start_norm = _normalize_date(left)
                    end_norm = _normalize_date(right)
                    if start_norm and end_norm:
                        if end_norm < start_norm:
                            start_norm, end_norm = end_norm, start_norm
                        return "range", None, start_norm, end_norm
                    raise ValueError(
                        "report_date ranges must include valid start and end dates (YYYY-MM-DD or DD.MM.YYYY)."
                    )

            normalized_single = _normalize_date(text)
            if normalized_single:
                return "date", normalized_single, None, None
            return "", None, None, None

        normalized_scope = _normalize_scope_token(scope_raw)
        literal = _stringify(report_date_raw)
        inferred_scope = ("", None, None, None)
        if not normalized_scope and literal:
            inferred_scope = _infer_scope_from_literal(literal)
            normalized_scope = inferred_scope[0]

        if not normalized_scope:
            raise ValueError(
                "missing a required argument: 'time_scope'. Choose one of 'all', 'latest', 'date', or 'range'. "
                "Provide date_exact for 'date' or date_start/date_end for 'range' using YYYY-MM-DD or DD.MM.YYYY."
            )

        if normalized_scope == "all":
            return "all", None, None, None
        if normalized_scope == "latest":
            return "latest", None, None, None

        normalized_exact = _normalize_date(date_exact_raw) or inferred_scope[1]
        normalized_start = _normalize_date(date_start_raw) or inferred_scope[2]
        normalized_end = _normalize_date(date_end_raw) or inferred_scope[3]

        if normalized_scope == "date":
            if not normalized_exact:
                if literal:
                    normalized_exact = _normalize_date(literal)
            if not normalized_exact:
                raise ValueError(
                    "time_scope='date' requires date_exact in YYYY-MM-DD or DD.MM.YYYY format (e.g., '2020-05-01')."
                )
            return "date", normalized_exact, None, None

        if normalized_scope == "range":
            if literal and not (normalized_start and normalized_end):
                if ".." in literal:
                    left, right = literal.split("..", 1)
                    normalized_start = normalized_start or _normalize_date(left)
                    normalized_end = normalized_end or _normalize_date(right)
                elif " to " in literal.lower():
                    left, right = literal.lower().split(" to ", 1)
                    normalized_start = normalized_start or _normalize_date(left)
                    normalized_end = normalized_end or _normalize_date(right)
            if not normalized_start or not normalized_end:
                raise ValueError(
                    "time_scope='range' requires both date_start and date_end using YYYY-MM-DD or DD.MM.YYYY."
                )
            if normalized_end < normalized_start:
                normalized_start, normalized_end = normalized_end, normalized_start
            return "range", None, normalized_start, normalized_end

        raise ValueError(f"Unsupported time_scope '{normalized_scope}'.")

    def _rank_sections_hybrid(
        self,
        *,
        sections: List[Dict[str, Any]],
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Rank using fused dense + char-ngram BM25 scores."""
        if not sections or self._cached_sections is None:
            return []

        # Map filtered sections to indices in the full cached set
        cached_id_to_idx = {
            s.get("section_id"): i for i, s in enumerate(self._cached_sections)
        }
        indices = []
        valid_sections = []
        for s in sections:
            idx = cached_id_to_idx.get(s.get("section_id"))
            if idx is not None:
                indices.append(idx)
                valid_sections.append(s)
        if not indices:
            return []

        idx_arr = np.array(indices)

        # Dense scores
        embedder = self._ensure_embedder()
        q_vec = embedder.encode([query], normalize_embeddings=True)
        dense_scores = np.dot(self._cached_embeddings[idx_arr], q_vec.T).squeeze()
        dense_norm = _minmax_normalize(dense_scores)

        # BM25 scores (char-ngram)
        mask = np.zeros(self._cached_bm25_index.N, dtype=bool)
        mask[idx_arr] = True
        bm25_scores = self._cached_bm25_index.score(query, mask=mask)
        bm25_norm = _minmax_normalize(bm25_scores)

        # Fuse
        fused = self.hybrid_alpha * bm25_norm + (1 - self.hybrid_alpha) * dense_norm
        order = np.argsort(-fused)

        k = max(1, int(top_k) if top_k else self.top_k)
        matches = []
        for rank, pos in enumerate(order[:k]):
            section = valid_sections[int(pos)]
            text = section.get("section_content") or ""
            snippet = self._build_snippet(text, query)
            normalized_date = section.get("_normalized_report_date") or self._normalize_report_date(
                section.get("report_date")
            )
            alias_id = None
            if section.get("report_id") and normalized_date:
                alias_id = f"report:{section['report_id']}:{normalized_date}"
            matches.append(
                {
                    "section_id": section.get("section_id"),
                    "report_id": section.get("report_id"),
                    "report_type": section.get("report_type"),
                    "report_date": section.get("report_date"),
                    "patient_id": section.get("patient_id"),
                    "section_name": section.get("section_name"),
                    "text": text,
                    "snippet": snippet,
                    "score": float(fused[pos]),
                    "normalized_report_date": normalized_date,
                    "citation_id": self._build_report_citation_id(section, rank),
                    "citation_alias": alias_id,
                }
            )
        return matches

    def _rank_sections(
        self,
        *,
        sections: List[Dict[str, Any]],
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not sections:
            return []

        # Dispatch to hybrid path when available
        if self.use_hybrid and self._cached_embeddings is not None:
            return self._rank_sections_hybrid(sections=sections, query=query, top_k=top_k)

        documents = [section.get("section_content") or "" for section in sections]
        tokenized_docs = [self._tokenize(text) for text in documents]
        if not any(tokenized_docs):
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(query_tokens)
        scored_sections = list(zip(sections, scores))
        scored_sections.sort(key=lambda item: item[1], reverse=True)

        k = max(1, int(top_k) if top_k else self.top_k)
        matches = []
        for index, (section, score) in enumerate(scored_sections[:k]):
            text = section.get("section_content") or ""
            snippet = self._build_snippet(text, query)
            normalized_date = section.get("_normalized_report_date") or self._normalize_report_date(
                section.get("report_date")
            )
            alias_id = None
            if section.get("report_id") and normalized_date:
                alias_id = f"report:{section['report_id']}:{normalized_date}"
            matches.append(
                {
                    "section_id": section.get("section_id"),
                    "report_id": section.get("report_id"),
                    "report_type": section.get("report_type"),
                    "report_date": section.get("report_date"),
                    "patient_id": section.get("patient_id"),
                    "section_name": section.get("section_name"),
                    "text": text,
                    "snippet": snippet,
                    "score": float(score),
                    "normalized_report_date": normalized_date,
                    "citation_id": self._build_report_citation_id(section, index),
                    "citation_alias": alias_id,
                }
            )
        return matches

    def _format_response(self, matches: List[Dict[str, Any]], payload: Dict[str, Any]) -> str:
        if not matches:
            return (
                f"No report sections found for patient {payload['patient_id']} "
                "with the provided filters."
            )

        top = matches[0]
        report_info = []
        if top.get("report_type"):
            report_info.append(top["report_type"])
        if top.get("report_date"):
            report_info.append(str(top["report_date"]))

        details = " - ".join(report_info) if report_info else ""
        snippet = top.get("snippet") or (top.get("text", "")[:240] + "...")
        response_lines = [
            f"Top section for patient {payload['patient_id']}: {details}".strip(),
            snippet,
        ]
        return "\n".join(line for line in response_lines if line)

    @staticmethod
    def _parse_json(value: Any) -> Any | None:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return None
        return None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    @staticmethod
    def _build_snippet(text: str, query: str, length: int = 512) -> str:
        if not text:
            return ""
        lowered = text.lower()
        tokens = ReportsRAGTool._tokenize(query)
        if not tokens:
            return text[:length] + ("..." if len(text) > length else "")

        first = tokens[0]
        idx = lowered.find(first)
        if idx == -1:
            return text[:length] + ("..." if len(text) > length else "")

        start = max(0, idx - length // 4)
        end = min(len(text), start + length)
        snippet = text[start:end]
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        return f"{prefix}{snippet}{suffix}"

    @staticmethod
    def _derive_report_date(identifier: str) -> str | None:
        if not identifier:
            return None
        match = re.search(r"\d{2}\.\d{2}\.\d{4}", identifier)
        if match:
            return match.group(0)
        match = re.search(r"\d{4}-\d{2}-\d{2}", identifier)
        if match:
            return match.group(0)
        return None

    @staticmethod
    def _normalize_report_date(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None

        def _try_parse(candidate: str, pattern: str) -> Optional[str]:
            try:
                return datetime.strptime(candidate, pattern).strftime("%Y-%m-%d")
            except ValueError:
                return None

        normalized = text.replace("/", ".")
        patterns = ["%Y-%m-%d", "%d.%m.%Y", "%d-%m-%Y", "%Y.%m.%d"]
        for pattern in patterns:
            parsed = _try_parse(normalized if "." in pattern else text, pattern)
            if parsed:
                return parsed
        try:
            return datetime.fromisoformat(text).date().isoformat()
        except ValueError:
            return None

    @staticmethod
    def _build_report_citation_id(section: Dict[str, Any], index: int) -> str:
        report_id = section.get("report_id") or f"report-{index}"
        section_id = section.get("section_id") or f"section-{index}"
        normalized_date = section.get("normalized_report_date") or ReportsRAGTool._normalize_report_date(
            section.get("report_date")
        )
        return f"report:{report_id}:{section_id}:{normalized_date or 'na'}"


class FullContextTool(BaseTool):
    """
    Generate query-conditioned summaries over the full patient record (reports + labs) in chunks.

    This tool builds a full text context (chronological reports plus labs), splits it into
    fixed chunks, runs a summary LLM call per chunk using the same model endpoint, and returns
    the summaries as context nodes. Raw chunks are not exposed.
    """

    DEFAULT_CHUNKS = 4
    DEFAULT_MAX_TOKENS_SUMMARY = 800
    DEFAULT_MAX_CHUNK_CHARS = 20000   # cap chunk input size to avoid overlength prompts
    DEFAULT_TIMEOUT = 60.0

    SUMMARY_SYSTEM_PROMPT = (
        "You are a careful clinical information extraction assistant. You only use the provided text. "
        "You never guess, infer, or fill in missing details. If something is not explicitly stated, mark it as unknown. "
        "Preserve dates, units, negations, uncertainty, and the original wording in quotes."
    )

    SUMMARY_USER_TEMPLATE = """Task: Create a query-conditioned evidence summary from the patient record chunk below.

Chunk ID: {chunk_id}
User query:
{user_query}

Patient record chunk:
{chunk_text}

Output rules:
- Extract only information relevant to answering the user query. If nothing is relevant, say "No relevant evidence in this chunk."
- Every fact must quote supporting text verbatim (short quotes).
- Do not merge distinct time points; separate them.
- Keep it concise (<= 8 bullet lines)."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        patient_id: str = ReportsRAGTool.DEFAULT_PATIENT_ID,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.db_path = Path(db_path) if db_path else ReportsRAGTool.DB_PATH
        self.patient_id = patient_id
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://10.32.16.43:4000")
        self.model = model or os.getenv("VLLM_MODEL", "gpt-oss-120b")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._metadata = ToolMetadata(
            name="full_context_summaries",
            description=(
                "Summarize the full patient record (reports + labs) into a small set of query-conditioned chunk summaries. "
                "Returns only the summaries, not raw chunks."
            ),
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    # ------------------------- helpers -------------------------
    def _connect_db(self):
        uri = f"file:{self.db_path.resolve()}?mode=ro&immutable=1"
        return sqlite3.connect(uri, uri=True)

    def _fetch_reports(self, conn: sqlite3.Connection, patient_id: str) -> List[str]:
        cursor = conn.execute(
            """
            SELECT report_date, report_type, section_name, section_content
            FROM report_sections
            WHERE patient_id = ?
            ORDER BY report_date, section_order
            """,
            (patient_id,),
        )
        formatted: List[str] = []
        for date, rtype, section_name, content in cursor.fetchall():
            header = f"[{date or 'na'} | {rtype or 'na'} | {section_name or ''}]"
            formatted.append(f"{header} {content or ''}".strip())
        return formatted

    def _fetch_labs(self, conn: sqlite3.Connection, patient_id: str) -> List[str]:
        cursor = conn.execute(
            """
            SELECT canonical_key, value, unit, ref_range, COALESCE(assessment_dt, date || 'T' || IFNULL(time, ''))
            FROM lab_values
            WHERE patient_id = ?
            ORDER BY COALESCE(assessment_dt, date || 'T' || IFNULL(time,'')) ASC
            """,
            (patient_id,),
        )
        formatted: List[str] = []
        for key, value, unit, ref_range, when in cursor.fetchall():
            stamp = when or "(no date)"
            pretty_val = value or "n/a"
            unit_suffix = f" {unit}" if unit else ""
            range_suffix = f" (ref {ref_range})" if ref_range else ""
            formatted.append(f"{stamp}: {key} = {pretty_val}{unit_suffix}{range_suffix}")
        return formatted

    @staticmethod
    def _build_full_context(patient_id: str, reports: List[str], labs: List[str]) -> str:
        sections: List[str] = []
        sections.append(f"Patient ID: {patient_id}")
        sections.append("== Reports ==")
        sections.extend(reports or ["(no reports)"])
        sections.append("== Labs ==")
        sections.extend(labs or ["(no labs)"])
        return "\n".join(sections)

    @staticmethod
    def _chunk_text(text: str, num_chunks: int) -> List[str]:
        lines = text.splitlines()
        total_lines = len(lines)
        if total_lines == 0 or num_chunks <= 0:
            return []
        chunk_size = math.ceil(total_lines / num_chunks)
        return ["\n".join(lines[i : i + chunk_size]) for i in range(0, total_lines, chunk_size)]

    def _call_summary_llm(
        self,
        chunk_id: str,
        user_query: str,
        chunk_text: str,
        max_tokens: int,
        timeout: float,
    ) -> str:
        if OpenAI is None:  # pragma: no cover
            return "LLM unavailable; raw chunk used."
        prompt = self.SUMMARY_USER_TEMPLATE.format(
            chunk_id=chunk_id, user_query=user_query, chunk_text=chunk_text
        )
        try:
            client = OpenAI(base_url=self.base_url.rstrip("/"), api_key=self.api_key)
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            choices = getattr(resp, "choices", None) or []
            if choices:
                msg = choices[0].get("message") if isinstance(choices[0], dict) else getattr(choices[0], "message", None)
                if msg and (msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)):
                    content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
                    return strip_hidden_reasoning(content)
        except Exception as exc:  # pragma: no cover - robustness
            logger.warning("Chunk summary generation failed for %s: %s", chunk_id, exc)
            return f"No LLM summary (error: {exc}); raw chunk retained."
        return "LLM returned no content for this chunk."

    # ------------------------- main call -------------------------
    def __call__(  # type: ignore[override]
        self,
        query: str,
        patient_id: str | None = None,
        chunk_count: int = DEFAULT_CHUNKS,
        max_tokens_summary: int = DEFAULT_MAX_TOKENS_SUMMARY,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        timeout: float = DEFAULT_TIMEOUT,
        **_: Any,
    ) -> ToolOutput:
        pid = patient_id or self.patient_id
        chunk_count = max(1, int(chunk_count))
        # Clamp summaries to a safe upper bound to prevent negative max_tokens server-side
        max_tokens_summary = max(1, min(800, int(max_tokens_summary)))
        max_chunk_chars = max(1000, int(max_chunk_chars))

        try:
            conn = self._connect_db()
            reports = self._fetch_reports(conn, pid)
            labs = self._fetch_labs(conn, pid)
            conn.close()
        except Exception as exc:
            summary = f"Failed to load context for patient {pid}: {exc}"
            return ToolOutput(
                tool_name=self._metadata.name,
                content=summary,
                raw_input={"kwargs": {"query": query, "patient_id": pid}},
                raw_output={"status": "error", "summary": summary},
                is_error=True,
            )

        full_text = self._build_full_context(pid, reports, labs)
        chunks = self._chunk_text(full_text, chunk_count)
        # Further split overly long chunks by character limit to keep prompts safe
        bounded_chunks: List[str] = []
        for chunk in chunks:
            if len(chunk) <= max_chunk_chars:
                bounded_chunks.append(chunk)
                continue
            for start in range(0, len(chunk), max_chunk_chars):
                bounded_chunks.append(chunk[start : start + max_chunk_chars])
        chunks = bounded_chunks
        if not chunks:
            summary = f"No context available for patient {pid}."
            logger.warning("No reports/labs available for patient %s when building full context summaries.", pid)
            return ToolOutput(
                tool_name=self._metadata.name,
                content=summary,
                raw_input={"kwargs": {"query": query, "patient_id": pid}},
                raw_output={"status": "empty", "summary": summary, "context_nodes": []},
                is_error=True,
            )

        summaries: List[Dict[str, str]] = []
        for idx, chunk in enumerate(chunks, start=1):
            chunk_id = f"{pid}-chunk-{idx}"
            text = self._call_summary_llm(
                chunk_id=chunk_id,
                user_query=query,
                chunk_text=chunk,
                max_tokens=max_tokens_summary,
                timeout=timeout,
            )
            summaries.append({"chunk_id": chunk_id, "summary": text or "No relevant evidence in this chunk."})

        context_nodes: List[Dict[str, Any]] = []
        total = len(summaries)
        for idx, item in enumerate(summaries, start=1):
            summary_text = item["summary"] or "No relevant evidence in this chunk."
            snippet = summary_text[:360] + ("..." if len(summary_text) > 360 else "")
            context_nodes.append(
                {
                    "section_id": f"{pid}_summary_{idx}",
                    "report_id": f"{pid}_full_context",
                    "report_type": "full_context_summary",
                    "report_date": None,
                    "patient_id": pid,
                    "section_name": f"Summary chunk {idx}/{total}",
                    "text": summary_text,
                    "snippet": snippet,
                    "score": 0.0,
                }
            )

        summary_msg = f"Generated {len(context_nodes)} chunk summaries for patient {pid} (reports + labs)."
        result = {
            "status": "ok",
            "summary": summary_msg,
            "context_nodes": context_nodes,
            "data": {
                "patient_id": pid,
                "chunks": len(context_nodes),
                "max_tokens_summary": max_tokens_summary,
            },
        }
        return ToolOutput(
            tool_name=self._metadata.name,
            content=summary_msg,
            raw_input={
                "kwargs": {
                    "query": query,
                    "patient_id": pid,
                    "chunk_count": chunk_count,
                    "max_tokens_summary": max_tokens_summary,
                    "model": self.model,
                    "base_url": self.base_url,
                }
            },
            raw_output=result,
        )
class LabQueryTool(BaseTool):
    """
    Fetch laboratory values by canonical key directly from the lab_values table.
    """

    MARKER_SPLIT_PATTERN = re.compile(r"[;,]+")
    DB_PATH = ReportsRAGTool.DB_PATH

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        patient_id: str,
    ) -> None:
        self.db_path = Path(db_path) if db_path else self.DB_PATH
        self.patient_id = patient_id
        self.available_keys: List[str] = []
        base_descriptor = (
            "Retrieve lab values from the lab_values table. Args: lab_query (JSON array of canonical lab keys), "
            "patient_id?, time_scope ('all', 'latest', 'date', 'range'), date_exact?, date_start?, date_end?, "
            "max_results_per_lab?. Allowed lab keys are determined per patient; the tool returns the list if an "
            "unknown key is requested."
        )
        descriptor = self._append_available_keys(base_descriptor, self.patient_id)
        self._metadata = ToolMetadata(
            name="retrieve_lab_values",
            description=descriptor,
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def __call__(
        self,
        lab_query: Optional[Sequence[str]] = None,
        *,
        time_scope: str | None = None,
        patient_id: str | None = None,
        date_exact: str | None = None,
        date_start: str | None = None,
        date_end: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        max_results_per_lab: Optional[int] = None,
        **kwargs: Any,
    ) -> ToolOutput:
        patient = patient_id or self.patient_id
        if not patient:
            raise ValueError("patient_id is required for retrieve_lab_values.")
        lab_query_items = self._coerce_lab_query_argument(lab_query)
        if not lab_query_items:
            raise ValueError("Provide lab_query as a JSON array of canonical lab keys, e.g., [\"Albumin\"].")

        resolved_start, resolved_end, _, effective_max_results = self._resolve_time_scope(
            time_scope=time_scope,
            date_exact=date_exact,
            date_start=date_start,
            date_end=date_end,
            start_date=start_date,
            end_date=end_date,
            window_policy=None,
            window_days=None,
            max_results_per_lab=max_results_per_lab,
        )
        effective_max_results = 5 if effective_max_results is None else effective_max_results

        conn = sqlite3.connect(self.db_path)
        available_keys = self._available_keys(conn, patient)
        key_lookup = {k.lower(): k for k in available_keys}

        requested: List[str] = []
        missing: List[str] = []
        for item in lab_query_items:
            key = key_lookup.get(item.lower())
            if key:
                requested.append(key)
            else:
                missing.append(item)

        warnings: List[str] = []
        if missing:
            warnings.append(
                f"Unknown lab keys for patient {patient}: {', '.join(missing)}. "
                f"Allowed: {', '.join(available_keys)}"
            )
        if not requested:
            conn.close()
            payload = {"warnings": warnings, "available_keys": available_keys, "values": [], "context_nodes": []}
            return ToolOutput(
                tool_name=self._metadata.name,
                content=json.dumps(payload, ensure_ascii=False),
                raw_input={"lab_query": lab_query_items, "patient_id": patient},
                raw_output=payload,
                is_error=True,
            )

        values, context_nodes = self._fetch_values(
            conn=conn,
            patient_id=patient,
            keys=requested,
            start_date=resolved_start,
            end_date=resolved_end,
            max_results_per_lab=effective_max_results,
        )
        conn.close()

        payload = {
            "available_keys": available_keys,
            "values": values,
            "warnings": warnings,
            "context_nodes": context_nodes,
        }
        return ToolOutput(
            tool_name=self._metadata.name,
            content=json.dumps(payload, ensure_ascii=False),
            raw_input={
                "lab_query": lab_query_items,
                "patient_id": patient,
                "date_start": resolved_start,
                "date_end": resolved_end,
                "max_results_per_lab": effective_max_results,
            },
            raw_output=payload,
            is_error=False,
        )

    def _available_keys(self, conn: sqlite3.Connection, patient_id: str) -> List[str]:
        rows = conn.execute(
            "SELECT DISTINCT canonical_key FROM lab_values WHERE patient_id=? ORDER BY canonical_key",
            (patient_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def _fetch_values(
        self,
        *,
        conn: sqlite3.Connection,
        patient_id: str,
        keys: Sequence[str],
        start_date: Optional[str],
        end_date: Optional[str],
        max_results_per_lab: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        values: List[Dict[str, Any]] = []
        context_nodes: List[Dict[str, Any]] = []
        for key in keys:
            params: List[Any] = [patient_id, key]
            conditions = ["patient_id=?", "canonical_key=?"]
            if start_date:
                conditions.append("date>=?")
                params.append(start_date)
            if end_date:
                conditions.append("date<=?")
                params.append(end_date)
            where_clause = " AND ".join(conditions)
            sql = f"""
            SELECT canonical_key, mapped_name, date, time, assessment_dt, value, value_num, unit, ref_range, source_test_id
            FROM lab_values
            WHERE {where_clause}
            ORDER BY COALESCE(assessment_dt, date || 'T' || IFNULL(time,'')) DESC, date DESC, time DESC
            LIMIT ?
            """
            params.append(max_results_per_lab)
            rows = conn.execute(sql, params).fetchall()
            entry_values: List[Dict[str, Any]] = []
            for row in rows:
                (_key, mapped_name, date, time_str, ts, val, val_num, unit, ref_range, assess_id) = row
                entry = {
                    "canonical_key": key,
                    "mapped_name": mapped_name,
                    "date": date,
                    "time": time_str,
                    "timestamp": ts,
                    "value": val,
                    "value_num": val_num,
                    "unit": unit,
                    "reference_range": ref_range,
                    "assessment_id": assess_id,
                }
                entry_values.append(entry)
                citation_id = f"lab:{patient_id}:{key}:{assess_id or (date or '')}"
                display = mapped_name or key
                context_nodes.append(
                    {
                        "section_id": f"lab::{key}::{assess_id or (date or '')}",
                        "report_type": display,  # surface the lab name in UI labels
                        "section_name": display,
                        "text": f"{date or ''} {time_str or ''}: {val} {unit or ''}".strip(),
                        "lab_id": key,
                        "display_name": display,
                        "test": display,
                        "value": val,
                        "unit": unit,
                        "reference_range": ref_range,
                        "date": date,
                        "time": time_str,
                        "assessment_id": assess_id,
                        "citation_id": citation_id,
                    }
                )
            values.append(
                {
                    "canonical_key": key,
                    "mapped_name": rows[0][1] if rows else key,
                    "values": entry_values,
                }
            )
        return values, context_nodes

    def _resolve_time_scope(
        self,
        *,
        time_scope: str | None,
        date_exact: str | None,
        date_start: str | None,
        date_end: str | None,
        start_date: str | None,
        end_date: str | None,
        window_policy: str | None,
        window_days: Optional[int],
        max_results_per_lab: Optional[int],
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[int]]:
        resolved_start = start_date or date_start
        resolved_end = end_date or date_end
        scope = (time_scope or "").strip().lower()

        if date_exact and not (resolved_start or resolved_end):
            resolved_start = date_exact
            resolved_end = date_exact

        if scope == "date":
            if date_exact:
                resolved_start = date_exact
                resolved_end = date_exact
            elif resolved_start and not resolved_end:
                resolved_end = resolved_start
            elif resolved_end and not resolved_start:
                resolved_start = resolved_end
        elif scope == "all":
            resolved_start = None
            resolved_end = None
        elif scope == "latest":
            if max_results_per_lab is None:
                max_results_per_lab = 1

        return resolved_start, resolved_end, window_policy, max_results_per_lab

    def _coerce_lab_query_argument(self, lab_query: Optional[Sequence[str]]) -> List[str]:
        if lab_query is None:
            return []
        if isinstance(lab_query, str):
            raise ValueError("lab_query must be a list of marker strings, e.g., [\"IgG\", \"IgA\"].")
        if isinstance(lab_query, (list, tuple, set)):
            items: List[str] = []
            for item in lab_query:
                if item is None:
                    continue
                if isinstance(item, str):
                    items.extend(self._split_marker_string(item))
                else:
                    items.extend(self._split_marker_string(str(item)))
            return [val for val in items if val]
        raise ValueError("lab_query must be a list of marker strings, e.g., [\"IgG\", \"IgA\"].")

    def _split_marker_string(self, text: str | None) -> List[str]:
        if not text:
            return []
        return [chunk.strip() for chunk in self.MARKER_SPLIT_PATTERN.split(text) if chunk.strip()]

    def _append_available_keys(self, descriptor: str, patient_id: str) -> str:
        try:
            conn = sqlite3.connect(self.db_path)
            keys = self._available_keys(conn, patient_id)
            conn.close()
            self.available_keys = keys
        except Exception:
            logger.warning("Failed to load available lab keys for patient %s", patient_id, exc_info=True)
            return descriptor
        suffix = (
            f" Available lab keys for patient {patient_id}: {', '.join(keys)}"
            if keys
            else f" No lab keys found for patient {patient_id}."
        )
        return f"{descriptor} {suffix}"


def load_default_tools(
    *,
    patient_id: str,
    db_path: str | Path | None = None,
    use_hybrid: bool = False,
    hybrid_alpha: float = 0.5,
    embed_model: str = "models/distiluse-base-multilingual-cased-v2",
) -> List[BaseTool]:
    """Return the default tool bundle for a specific patient."""

    normalized_patient = str(patient_id).strip()
    if not normalized_patient:
        raise ValueError("patient_id is required to load default tools; provide a valid identifier.")

    reports_tool = ReportsRAGTool(
        db_path=db_path,
        patient_id=normalized_patient,
        use_hybrid=use_hybrid,
        hybrid_alpha=hybrid_alpha,
        embed_model=embed_model,
    )
    stream_tool = FullContextTool(db_path=db_path, patient_id=normalized_patient)
    lab_tool = LabQueryTool(db_path=db_path, patient_id=normalized_patient)
    
    # Add scoring tools
    iss_tool = ISSScoreTool()
    riss_tool = RISSScoreTool()
    r2iss_tool = R2ISSScoreTool()
    ipssr_tool = IPSSRScoreTool()
    hctci_tool = HCTCITool()

    return [reports_tool, stream_tool, lab_tool, iss_tool, riss_tool, r2iss_tool, ipssr_tool, hctci_tool]


def fetch_patient_lab_keys(patient_id: str, db_path: str | Path | None = None) -> List[str]:
    """Return canonical lab keys available for the given patient."""
    if not patient_id:
        return []
    path = Path(db_path) if db_path else LabQueryTool.DB_PATH
    try:
        conn = sqlite3.connect(path)
        cursor = conn.execute(
            "SELECT DISTINCT canonical_key FROM lab_values WHERE patient_id=? ORDER BY canonical_key",
            (patient_id,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [row[0] for row in rows if row and row[0]]
    except Exception:
        logger.warning("Failed to load lab keys for patient %s", patient_id, exc_info=True)
        return []


# ==================== Scoring Tools ====================


class ISSScoreTool(BaseTool):
    """Calculate ISS (International Staging System) score for multiple myeloma."""

    def __init__(self) -> None:
        self._metadata = ToolMetadata(
            name="calculate_iss_score",
            description=(
                "Calculate the ISS (International Staging System) score for multiple myeloma staging. "
                "Requires two specific lab values that must be retrieved before calling this tool: "
                "(1) serum_beta2_microglobulin: β2-microglobulin level in mg/L, "
                "(2) serum_albumin: Albumin level in g/dL. "
                "Returns a JSON string with stage (I/II/III), numeric score, and measurement date. "
                "ISS Stage I: β2M < 3.5 mg/L AND albumin ≥ 3.5 g/dL. "
                "ISS Stage III: β2M ≥ 5.5 mg/L. "
                "ISS Stage II: All other combinations."
            ),
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def __call__(  # type: ignore[override]
        self,
        serum_beta2_microglobulin: float,
        serum_albumin: float,
        measurement_date: str | None = None,
        **_: Any,
    ) -> ToolOutput:
        """
        Calculate ISS score.

        Args:
            serum_beta2_microglobulin: β2-microglobulin level in mg/L (required)
            serum_albumin: Albumin level in g/dL (required)
            measurement_date: Date of measurement (optional, format: YYYY-MM-DD)

        Returns:
            ToolOutput with JSON string containing stage, score, and date
        """
        try:
            beta2m = float(serum_beta2_microglobulin)
            albumin = float(serum_albumin)

            if beta2m < 3.5 and albumin >= 3.5:
                stage, score = "I", 1
            elif beta2m >= 5.5:
                stage, score = "III", 3
            else:
                stage, score = "II", 2

            result = {
                "stage": stage,
                "score": score,
                "beta2_microglobulin_mg_per_L": beta2m,
                "albumin_g_per_dL": albumin,
                "date": measurement_date,
                "source": "calculated",
                "scoring_system": "ISS",
            }

            result_json = json.dumps(result, indent=2)
            return ToolOutput(
                tool_name=self._metadata.name,
                content=result_json,
                raw_input={
                    "kwargs": {
                        "serum_beta2_microglobulin": beta2m,
                        "serum_albumin": albumin,
                        "measurement_date": measurement_date,
                    }
                },
                raw_output=result,
                is_error=False,
            )

        except (ValueError, TypeError) as exc:
            error_msg = f"Invalid input values for ISS calculation: {exc}"
            logger.error(error_msg)
            return ToolOutput(
                tool_name=self._metadata.name,
                content=error_msg,
                raw_input={
                    "kwargs": {
                        "serum_beta2_microglobulin": serum_beta2_microglobulin,
                        "serum_albumin": serum_albumin,
                        "measurement_date": measurement_date,
                    }
                },
                raw_output={"error": str(exc)},
                is_error=True,
            )


class RISSScoreTool(BaseTool):
    """Calculate R-ISS (Revised International Staging System) score."""

    def __init__(self) -> None:
        self._metadata = ToolMetadata(
            name="calculate_riss_score",
            description=(
                "Calculate the R-ISS (Revised International Staging System) score for multiple myeloma. "
                "Requires four specific values that must be retrieved before calling this tool: "
                "(1) serum_beta2_microglobulin: β2-microglobulin in mg/L, "
                "(2) serum_albumin: Albumin in g/dL, "
                "(3) ldh_elevated: True if LDH is above the upper limit of normal, "
                "(4) high_risk_ca: True if high-risk cytogenetic abnormalities are present "
                "(del17p, t(4;14), or t(14;16) detected via FISH or cytogenetics). "
                "Returns JSON with stage (I/II/III), score, and date. "
                "R-ISS Stage I: ISS stage I AND no high-risk CA AND normal LDH. "
                "R-ISS Stage III: ISS stage III AND (high-risk CA OR elevated LDH). "
                "R-ISS Stage II: All others."
            ),
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def __call__(  # type: ignore[override]
        self,
        serum_beta2_microglobulin: float,
        serum_albumin: float,
        ldh_elevated: bool,
        high_risk_ca: bool,
        measurement_date: str | None = None,
        **_: Any,
    ) -> ToolOutput:
        """
        Calculate R-ISS score.

        Args:
            serum_beta2_microglobulin: β2-microglobulin in mg/L
            serum_albumin: Albumin in g/dL
            ldh_elevated: True if LDH above upper limit of normal
            high_risk_ca: True if high-risk cytogenetic abnormalities present
            measurement_date: Date of measurement (format: YYYY-MM-DD)

        Returns:
            ToolOutput with JSON string containing stage, score, and criteria
        """
        try:
            beta2m = float(serum_beta2_microglobulin)
            albumin = float(serum_albumin)
            ldh_high = bool(ldh_elevated)
            ca_high_risk = bool(high_risk_ca)

            # First determine ISS stage
            if beta2m < 3.5 and albumin >= 3.5:
                iss_stage = 1
            elif beta2m >= 5.5:
                iss_stage = 3
            else:
                iss_stage = 2

            # Apply R-ISS criteria
            if iss_stage == 1 and not ca_high_risk and not ldh_high:
                stage, score = "I", 1
            elif iss_stage == 3 and (ca_high_risk or ldh_high):
                stage, score = "III", 3
            else:
                stage, score = "II", 2

            result = {
                "stage": stage,
                "score": score,
                "iss_stage": iss_stage,
                "beta2_microglobulin_mg_per_L": beta2m,
                "albumin_g_per_dL": albumin,
                "ldh_elevated": ldh_high,
                "high_risk_ca": ca_high_risk,
                "date": measurement_date,
                "source": "calculated",
                "scoring_system": "R-ISS",
            }

            result_json = json.dumps(result, indent=2)
            return ToolOutput(
                tool_name=self._metadata.name,
                content=result_json,
                raw_input={
                    "kwargs": {
                        "serum_beta2_microglobulin": beta2m,
                        "serum_albumin": albumin,
                        "ldh_elevated": ldh_high,
                        "high_risk_ca": ca_high_risk,
                        "measurement_date": measurement_date,
                    }
                },
                raw_output=result,
                is_error=False,
            )

        except (ValueError, TypeError) as exc:
            error_msg = f"Invalid input values for R-ISS calculation: {exc}"
            logger.error(error_msg)
            return ToolOutput(
                tool_name=self._metadata.name,
                content=error_msg,
                raw_input={
                    "kwargs": {
                        "serum_beta2_microglobulin": serum_beta2_microglobulin,
                        "serum_albumin": serum_albumin,
                        "ldh_elevated": ldh_elevated,
                        "high_risk_ca": high_risk_ca,
                        "measurement_date": measurement_date,
                    }
                },
                raw_output={"error": str(exc)},
                is_error=True,
            )


class R2ISSScoreTool(BaseTool):
    """Calculate R2-ISS (Second Revision ISS) score."""

    def __init__(self) -> None:
        self._metadata = ToolMetadata(
            name="calculate_r2iss_score",
            description=(
                "Calculate the R2-ISS (Second Revision International Staging System) score. "
                "Requires five specific values that must be retrieved before calling: "
                "(1) serum_beta2_microglobulin: β2-microglobulin in mg/L, "
                "(2) serum_albumin: Albumin in g/dL, "
                "(3) ldh_ratio: LDH divided by upper limit of normal (LDH/ULN), "
                "(4) high_risk_ca_present: True if high-risk cytogenetic abnormalities present "
                "(del17p, t(4;14), t(14;16), del1p, or gain1q), "
                "(5) iss1q_gain: True if ISS stage combined with 1q gain/amplification is present. "
                "Returns JSON with stage (I/II/III/IV), score, and criteria. "
                "R2-ISS provides more refined risk stratification than R-ISS."
            ),
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def __call__(  # type: ignore[override]
        self,
        serum_beta2_microglobulin: float,
        serum_albumin: float,
        ldh_ratio: float,
        high_risk_ca_present: bool,
        iss1q_gain: bool,
        measurement_date: str | None = None,
        **_: Any,
    ) -> ToolOutput:
        """
        Calculate R2-ISS score.

        Args:
            serum_beta2_microglobulin: β2-microglobulin in mg/L
            serum_albumin: Albumin in g/dL
            ldh_ratio: LDH/ULN ratio (LDH divided by upper limit of normal)
            high_risk_ca_present: High-risk CA (del17p, t(4;14), t(14;16), del1p, gain1q)
            iss1q_gain: ISS stage and 1q gain/amplification present
            measurement_date: Date of measurement

        Returns:
            ToolOutput with JSON string containing stage, score, and criteria
        """
        try:
            beta2m = float(serum_beta2_microglobulin)
            albumin = float(serum_albumin)
            ldh_r = float(ldh_ratio)
            ca_high_risk = bool(high_risk_ca_present)
            q1_gain = bool(iss1q_gain)

            # Calculate base ISS stage
            if beta2m < 3.5 and albumin >= 3.5:
                iss_stage = 1
            elif beta2m >= 5.5:
                iss_stage = 3
            else:
                iss_stage = 2

            # Apply R2-ISS criteria (simplified version based on common implementation)
            if iss_stage == 1 and not ca_high_risk and ldh_r <= 1 and not q1_gain:
                stage, score = "I", 1
            elif iss_stage == 3 and ca_high_risk and ldh_r > 1:
                stage, score = "IV", 4
            elif iss_stage == 3 or ca_high_risk or ldh_r > 1 or q1_gain:
                stage, score = "III", 3
            else:
                stage, score = "II", 2

            result = {
                "stage": stage,
                "score": score,
                "iss_stage": iss_stage,
                "beta2_microglobulin_mg_per_L": beta2m,
                "albumin_g_per_dL": albumin,
                "ldh_ratio": ldh_r,
                "high_risk_ca_present": ca_high_risk,
                "iss1q_gain": q1_gain,
                "date": measurement_date,
                "source": "calculated",
                "scoring_system": "R2-ISS",
            }

            result_json = json.dumps(result, indent=2)
            return ToolOutput(
                tool_name=self._metadata.name,
                content=result_json,
                raw_input={
                    "kwargs": {
                        "serum_beta2_microglobulin": beta2m,
                        "serum_albumin": albumin,
                        "ldh_ratio": ldh_r,
                        "high_risk_ca_present": ca_high_risk,
                        "iss1q_gain": q1_gain,
                        "measurement_date": measurement_date,
                    }
                },
                raw_output=result,
                is_error=False,
            )

        except (ValueError, TypeError) as exc:
            error_msg = f"Invalid input values for R2-ISS calculation: {exc}"
            logger.error(error_msg)
            return ToolOutput(
                tool_name=self._metadata.name,
                content=error_msg,
                raw_input={
                    "kwargs": {
                        "serum_beta2_microglobulin": serum_beta2_microglobulin,
                        "serum_albumin": serum_albumin,
                        "ldh_ratio": ldh_ratio,
                        "high_risk_ca_present": high_risk_ca_present,
                        "iss1q_gain": iss1q_gain,
                        "measurement_date": measurement_date,
                    }
                },
                raw_output={"error": str(exc)},
                is_error=True,
            )


class IPSSRScoreTool(BaseTool):
    """Calculate the IPSS-R (Revised International Prognostic Scoring System) for MDS."""

    CYTOGENETIC_POINTS = {
        "very good": 0,
        "very_good": 0,
        "good": 1,
        "intermediate": 2,
        "poor": 3,
        "very poor": 4,
        "very_poor": 4,
    }

    def __init__(self) -> None:
        self._metadata = ToolMetadata(
            name="calculate_ipssr_score",
            description=(
                "Calculate the IPSS-R score (Revised International Prognostic Scoring System) for MDS. "
                "Requires: hemoglobin_g_per_dl, platelets_10e9_per_L, anc_10e9_per_L, marrow_blast_percent, "
                "cytogenetic_risk (Very Good, Good, Intermediate, Poor, Very Poor), and optional measurement_date. "
                "Returns JSON with component points, total score, and risk category (Very Low → Very High)."
            ),
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def __call__(  # type: ignore[override]
        self,
        hemoglobin_g_per_dl: float,
        platelets_10e9_per_L: float,
        anc_10e9_per_L: float,
        marrow_blast_percent: float,
        cytogenetic_risk: str,
        measurement_date: str | None = None,
        **_: Any,
    ) -> ToolOutput:
        """
        Calculate IPSS-R score.

        Args:
            hemoglobin_g_per_dl: Hemoglobin in g/dL
            platelets_10e9_per_L: Platelet count (×10^9/L)
            anc_10e9_per_L: Absolute neutrophil count (×10^9/L)
            marrow_blast_percent: Bone marrow blasts (%)
            cytogenetic_risk: One of Very Good, Good, Intermediate, Poor, Very Poor
            measurement_date: Optional date of measurement (YYYY-MM-DD)
        """
        required_values = {
            "hemoglobin_g_per_dl": hemoglobin_g_per_dl,
            "platelets_10e9_per_L": platelets_10e9_per_L,
            "anc_10e9_per_L": anc_10e9_per_L,
            "marrow_blast_percent": marrow_blast_percent,
            "cytogenetic_risk": cytogenetic_risk,
        }
        missing = [name for name, val in required_values.items() if val is None or val == ""]
        if missing:
            msg = f"Missing required parameters for IPSS-R: {', '.join(sorted(missing))}"
            logger.error(msg)
            return ToolOutput(
                tool_name=self._metadata.name,
                content=msg,
                raw_input={"kwargs": required_values},
                raw_output={"error": msg},
                is_error=True,
            )

        try:
            hgb = float(hemoglobin_g_per_dl)  # type: ignore[arg-type]
            platelets = float(platelets_10e9_per_L)  # type: ignore[arg-type]
            anc = float(anc_10e9_per_L)  # type: ignore[arg-type]
            blasts = float(marrow_blast_percent)  # type: ignore[arg-type]
        except (ValueError, TypeError) as exc:
            msg = f"Invalid numeric input for IPSS-R calculation: {exc}"
            logger.error(msg)
            return ToolOutput(
                tool_name=self._metadata.name,
                content=msg,
                raw_input={"kwargs": required_values},
                raw_output={"error": str(exc)},
                is_error=True,
            )

        cyto_key = (cytogenetic_risk or "").strip().lower().replace("-", "_").replace(" ", "_")
        cyto_points = self.CYTOGENETIC_POINTS.get(cyto_key)
        if cyto_points is None:
            valid_groups = ", ".join(sorted(set(self.CYTOGENETIC_POINTS.keys())))
            msg = f"Invalid cytogenetic_risk '{cytogenetic_risk}'. Valid: {valid_groups}"
            logger.error(msg)
            return ToolOutput(
                tool_name=self._metadata.name,
                content=msg,
                raw_input={"kwargs": {"cytogenetic_risk": cytogenetic_risk}},
                raw_output={"error": msg},
                is_error=True,
            )

        # Component scoring per IPSS-R tables
        if blasts <= 2:
            blast_points = 0
        elif 2 < blasts < 5:
            blast_points = 1
        elif 5 <= blasts <= 10:
            blast_points = 2
        else:
            blast_points = 3

        if hgb >= 10:
            hgb_points = 0
        elif 8 <= hgb < 10:
            hgb_points = 1
        else:
            hgb_points = 1.5

        if platelets >= 100:
            plt_points = 0
        elif 50 <= platelets < 100:
            plt_points = 0.5
        else:
            plt_points = 1

        anc_points = 0 if anc >= 0.8 else 0.5

        total_score = float(cyto_points + blast_points + hgb_points + plt_points + anc_points)

        if total_score <= 1.5:
            risk = "Very Low"
        elif total_score <= 3:
            risk = "Low"
        elif total_score <= 4.5:
            risk = "Intermediate"
        elif total_score <= 6:
            risk = "High"
        else:
            risk = "Very High"

        result = {
            "risk_category": risk,
            "total_score": total_score,
            "components": {
                "cytogenetics_points": cyto_points,
                "marrow_blast_points": blast_points,
                "hemoglobin_points": hgb_points,
                "platelets_points": plt_points,
                "anc_points": anc_points,
            },
            "inputs": {
                "hemoglobin_g_per_dl": hgb,
                "platelets_10e9_per_L": platelets,
                "anc_10e9_per_L": anc,
                "marrow_blast_percent": blasts,
                "cytogenetic_risk": cytogenetic_risk,
                "measurement_date": measurement_date,
            },
            "source": "calculated",
            "scoring_system": "IPSS-R",
        }

        return ToolOutput(
            tool_name=self._metadata.name,
            content=json.dumps(result, indent=2),
            raw_input={
                "kwargs": {
                    "hemoglobin_g_per_dl": hgb,
                    "platelets_10e9_per_L": platelets,
                    "anc_10e9_per_L": anc,
                    "marrow_blast_percent": blasts,
                    "cytogenetic_risk": cytogenetic_risk,
                    "measurement_date": measurement_date,
                }
            },
            raw_output=result,
            is_error=False,
        )


class HCTCITool(BaseTool):
    """Calculate the HCT-CI comorbidity score."""

    WEIGHT_1_FIELDS = [
        "arrhythmia",
        "cardiac_disease",
        "inflammatory_bowel_disease",
        "diabetes_medication",
        "cerebrovascular_disease",
        "psychiatric_disturbance",
        "mild_hepatic_abnormality",
        "obesity_bmi_gt_35",
        "persistent_infection",
    ]
    WEIGHT_2_FIELDS = [
        "rheumatologic_disease",
        "peptic_ulcer",
        "renal_moderate_severe",
        "pulmonary_moderate",
    ]
    WEIGHT_3_FIELDS = [
        "prior_solid_tumor",
        "heart_valve_disease",
        "pulmonary_severe",
        "hepatic_moderate_severe",
    ]

    def __init__(self) -> None:
        self._metadata = ToolMetadata(
            name="calculate_hctci_score",
            description=(
                "Calculate the HCT-CI score from structured comorbidity flags. "
                "Inputs: booleans for arrhythmia, cardiac_disease, inflammatory_bowel_disease, diabetes_medication, "
                "cerebrovascular_disease, psychiatric_disturbance, mild_hepatic_abnormality, obesity_bmi_gt_35, "
                "persistent_infection, rheumatologic_disease, peptic_ulcer, renal_moderate_severe, pulmonary_moderate, "
                "prior_solid_tumor, heart_valve_disease, pulmonary_severe, hepatic_moderate_severe. "
                "Returns total_score and risk_group (0=low, 1-2=intermediate, >=3=high)."
            ),
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def __call__(  # type: ignore[override]
        self,
        *,
        arrhythmia: bool | None = None,
        cardiac_disease: bool | None = None,
        inflammatory_bowel_disease: bool | None = None,
        diabetes_medication: bool | None = None,
        cerebrovascular_disease: bool | None = None,
        psychiatric_disturbance: bool | None = None,
        mild_hepatic_abnormality: bool | None = None,
        obesity_bmi_gt_35: bool | None = None,
        persistent_infection: bool | None = None,
        rheumatologic_disease: bool | None = None,
        peptic_ulcer: bool | None = None,
        renal_moderate_severe: bool | None = None,
        pulmonary_moderate: bool | None = None,
        prior_solid_tumor: bool | None = None,
        heart_valve_disease: bool | None = None,
        pulmonary_severe: bool | None = None,
        hepatic_moderate_severe: bool | None = None,
        **_: Any,
    ) -> ToolOutput:
        """Compute HCT-CI from provided comorbidity flags."""

        inputs = {
            "arrhythmia": arrhythmia,
            "cardiac_disease": cardiac_disease,
            "inflammatory_bowel_disease": inflammatory_bowel_disease,
            "diabetes_medication": diabetes_medication,
            "cerebrovascular_disease": cerebrovascular_disease,
            "psychiatric_disturbance": psychiatric_disturbance,
            "mild_hepatic_abnormality": mild_hepatic_abnormality,
            "obesity_bmi_gt_35": obesity_bmi_gt_35,
            "persistent_infection": persistent_infection,
            "rheumatologic_disease": rheumatologic_disease,
            "peptic_ulcer": peptic_ulcer,
            "renal_moderate_severe": renal_moderate_severe,
            "pulmonary_moderate": pulmonary_moderate,
            "prior_solid_tumor": prior_solid_tumor,
            "heart_valve_disease": heart_valve_disease,
            "pulmonary_severe": pulmonary_severe,
            "hepatic_moderate_severe": hepatic_moderate_severe,
        }
        missing = [name for name, val in inputs.items() if val is None]
        if missing:
            msg = f"Missing required comorbidity flags for HCT-CI: {', '.join(sorted(missing))}"
            logger.error(msg)
            return ToolOutput(
                tool_name=self._metadata.name,
                content=msg,
                raw_input={"kwargs": inputs},
                raw_output={"error": msg},
                is_error=True,
            )

        def _sum_flags(fields: list[str], weight: int) -> tuple[int, dict[str, int]]:
            subtotal = 0
            contributions: dict[str, int] = {}
            for field in fields:
                if bool(inputs[field]):
                    subtotal += weight
                    contributions[field] = weight
            return subtotal, contributions

        total = 0
        components: dict[str, int] = {}
        for fields, weight in (
            (self.WEIGHT_1_FIELDS, 1),
            (self.WEIGHT_2_FIELDS, 2),
            (self.WEIGHT_3_FIELDS, 3),
        ):
            subtotal, contrib = _sum_flags(fields, weight)
            total += subtotal
            components.update(contrib)

        if total == 0:
            risk_group = "low"
        elif total <= 2:
            risk_group = "intermediate"
        else:
            risk_group = "high"

        result = {
            "total_score": total,
            "risk_group": risk_group,
            "components": components,
            "inputs": inputs,
            "source": "calculated",
            "scoring_system": "HCT-CI",
        }

        return ToolOutput(
            tool_name=self._metadata.name,
            content=json.dumps(result, indent=2),
            raw_input={"kwargs": inputs},
            raw_output=result,
            is_error=False,
        )
