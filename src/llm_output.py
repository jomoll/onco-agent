"""Helpers for sanitizing LLM outputs before downstream parsing/storage."""

from __future__ import annotations

import re


_THINK_BLOCK_RE = re.compile(
    r"^\s*(?:<(?:think|thinking)>.*?</(?:think|thinking)>\s*)+",
    flags=re.IGNORECASE | re.DOTALL,
)


def strip_hidden_reasoning(text: str) -> str:
    """Remove leading hidden-reasoning blocks such as ``<think>...</think>``."""
    if not text:
        return text
    return _THINK_BLOCK_RE.sub("", text).lstrip()
