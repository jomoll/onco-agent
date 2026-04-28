"""Vertex AI Gemini agent using google-auth Application Default Credentials."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import requests

logger = logging.getLogger(__name__)

INPUT_PRICE_PER_TOKEN = 0.30 / 1_000_000   # USD per token
OUTPUT_PRICE_PER_TOKEN = 2.50 / 1_000_000  # USD per token

_VERTEX_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

try:  # pragma: no cover - optional dependency
    import google.auth
    import google.auth.transport.requests
    from google.oauth2 import service_account
    _GOOGLE_AUTH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GOOGLE_AUTH_AVAILABLE = False

from llama_index.core.base.llms.types import ChatMessage

from .agent_base import DSPyAgentBase
from .toolkit import BaseTool


@dataclass
class GeminiConfig:
    """Vertex AI connection and model settings."""

    project_id: str
    location: str = "us-central1"
    model: str = "gemini-3.1-pro-preview"
    service_account_file: Optional[str] = None
    temperature: float = 0.0
    max_output_tokens: int = 8192
    system_prompt: str = (
        "You are a curious clinical assistant. Plan your work, use the available tools deliberately, "
        "cite sources for every claim, and respond in German unless the user explicitly writes in English."
    )

    @property
    def endpoint(self) -> str:
        return (
            f"https://aiplatform.googleapis.com/v1/projects/{self.project_id}"
            f"/locations/{self.location}/publishers/google/models/{self.model}:generateContent"
        )


class GeminiChatAgent(DSPyAgentBase):
    """DSPy-enabled agent backed by Vertex AI Gemini via ADC."""

    def __init__(
        self,
        *,
        config: GeminiConfig,
        tools: Optional[Iterable[BaseTool]] = None,
        max_tool_rounds: int = 6,
    ) -> None:
        if not _GOOGLE_AUTH_AVAILABLE:  # pragma: no cover
            raise ImportError(
                "google-auth is required for GeminiChatAgent. "
                "Install it with: pip install google-auth"
            )
        self.config = config
        if config.service_account_file:
            self._credentials = service_account.Credentials.from_service_account_file(
                config.service_account_file, scopes=_VERTEX_SCOPES
            )
        else:
            self._credentials, _ = google.auth.default(scopes=_VERTEX_SCOPES)
        self._auth_request = google.auth.transport.requests.Request()
        super().__init__(tools=tools, max_tool_rounds=max_tool_rounds)

    # ------------------------------------------------------------------
    # DSPyAgentBase bridge
    # ------------------------------------------------------------------
    def _complete(self, prompt: str) -> str:
        result = self._post(self._build_body(prompt))
        self._log_usage(result)
        return self._extract_text(result)

    # ------------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------------
    def chat(self, message: str, **kwargs: Any) -> ChatMessage:
        """Single-turn chat helper."""
        del kwargs
        self.reset()
        self._append_user_message(message)
        result = self._post(self._build_body(message))
        reply_text = self._extract_text(result)
        self._append_assistant_message(reply_text)
        return self._history[-1]

    @property
    def _chat_history(self) -> Iterable[ChatMessage]:
        """Compatibility shim for legacy tests."""
        return self.history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _auth_header(self) -> dict[str, str]:
        if not self._credentials.valid:
            self._credentials.refresh(self._auth_request)
        return {
            "Authorization": f"Bearer {self._credentials.token}",
            "Content-Type": "application/json",
        }

    def _build_body(self, prompt: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_output_tokens,
            },
        }
        if self.config.system_prompt:
            body["system_instruction"] = {"parts": [{"text": self.config.system_prompt}]}
        return body

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        max_retries = 3
        backoff = 2.0
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(
                    self.config.endpoint,
                    json=body,
                    headers=self._auth_header(),
                    timeout=300,
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # pragma: no cover - defensive
                last_exc = exc
                if attempt >= max_retries:
                    break
                time.sleep(backoff)
                backoff *= 2
        raise last_exc  # type: ignore[misc]

    def _log_usage(self, result: dict[str, Any]) -> None:
        usage = result.get("usageMetadata") or {}
        if not usage:
            return
        prompt_tokens = usage.get("promptTokenCount") or 0
        output_tokens = usage.get("candidatesTokenCount") or 0
        thoughts_tokens = usage.get("thoughtsTokenCount") or 0
        cost = prompt_tokens * INPUT_PRICE_PER_TOKEN + (output_tokens + thoughts_tokens) * OUTPUT_PRICE_PER_TOKEN
        logger.info(
            "Gemini usage – thoughts: %s, output: %s, est_cost_usd=%.6f",
            thoughts_tokens,
            output_tokens,
            cost,
        )
        self._emit_event("usage", usage_metadata={**usage, "run_cost_usd": cost})

    @staticmethod
    def _extract_text(result: dict[str, Any]) -> str:
        for candidate in result.get("candidates", []):
            for part in (candidate.get("content") or {}).get("parts", []):
                text = part.get("text")
                if text:
                    return text
        return ""
