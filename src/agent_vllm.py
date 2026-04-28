"""OpenAI-compatible (vLLM) agent that delegates orchestration to the DSPy base class."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
from llama_index.core.base.llms.types import ChatMessage

from .agent_base import DSPyAgentBase
from .llm_output import strip_hidden_reasoning
from .toolkit import BaseTool


@dataclass
class VLLMConfig:
    """Connection and model settings for the vLLM-backed OpenAI endpoint."""

    base_url: str = "http://10.32.16.43:4000"
    api_key: Optional[str] = None
    model: str = "gpt-oss-120b"
    timeout: float = 300.0
    # Optional priority level for LiteLLM priority queue.
    # Accepts integer strings or named tiers: "priority"=0, "normal"=1, "background"=2.
    service_tier: Optional[str] = None

    def priority_int(self) -> Optional[int]:
        """Return service_tier as an integer for vLLM's extra_body priority field."""
        if self.service_tier is None:
            return None
        _named = {"priority": 0, "high": 0, "normal": 1, "low": 2, "background": 2}
        lower = self.service_tier.strip().lower()
        if lower in _named:
            return _named[lower]
        try:
            return int(self.service_tier)
        except ValueError:
            return None
    system_prompt: str = (
        "You are a curious clinical assistant. Plan your work, use the available tools deliberately, "
        "cite sources for every claim, and respond in German unless the user explicitly writes in English."
    )
    client_kwargs: Dict[str, Any] = field(default_factory=dict)
    completion_kwargs: Dict[str, Any] = field(default_factory=dict)

    def model_name(self) -> str:
        return self.model


class VLLMChatAgent(DSPyAgentBase):
    """DSPy-enabled agent for OpenAI-compatible (vLLM) deployments."""

    def __init__(
        self,
        *,
        config: VLLMConfig,
        tools: Optional[Iterable[BaseTool]] = None,
        max_tool_rounds: int = 6,
    ) -> None:
        if OpenAI is None:  # pragma: no cover - dependency missing
            raise ImportError("openai>=1.0 is required for VLLMChatAgent.")
        self.config = config
        self._system_prompt = config.system_prompt
        self._client = self._create_client()
        super().__init__(tools=tools, max_tool_rounds=max_tool_rounds)

    # ------------------------------------------------------------------
    # DSPyAgentBase bridge
    # ------------------------------------------------------------------
    def _complete(self, prompt: str) -> str:
        completion_kwargs = {
            "model": self.config.model_name(),
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
            "timeout": self.config.timeout,
            "stream": False,
        }
        priority = self.config.priority_int()
        if priority is not None:
            completion_kwargs["extra_body"] = {"priority": priority}
        completion_kwargs.update(self.config.completion_kwargs or {})

        # Simple retry for transient network/proxy read errors (e.g., Squid ERR_READ_ERROR)
        max_retries = 3
        backoff = 1.0
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.chat.completions.create(**completion_kwargs)
                return self._extract_text_from_chat(response)
            except Exception as exc:  # pragma: no cover - defensive
                last_exc = exc
                if attempt >= max_retries:
                    break
                time.sleep(backoff)
                backoff *= 2
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------------
    def chat(self, message: str, **kwargs: Any) -> ChatMessage:
        """Single-turn chat helper that can execute lightweight tool calls."""
        del kwargs
        self.reset()
        self._append_user_message(message)
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": message},
        ]

        for _ in range(self._max_tool_rounds):
            request_payload = {
                "model": self.config.model_name(),
                "messages": messages,
                "timeout": self.config.timeout,
            }
            priority = self.config.priority_int()
            if priority is not None:
                request_payload.setdefault("extra_body", {})
                request_payload["extra_body"]["priority"] = priority
            request_payload.update(self.config.completion_kwargs or {})
            response = self._client.chat.completions.create(**request_payload)
            message_payload = self._extract_choice_message(response)
            if not message_payload:
                break

            tool_calls = self._extract_tool_calls(message_payload)
            if tool_calls:
                for call in tool_calls:
                    tool_content = self._invoke_tool_for_chat(call["name"], call["arguments"])
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": call["name"],
                            "content": tool_content,
                        }
                    )
                continue

            content = self._extract_message_content(message_payload)
            messages.append({"role": "assistant", "content": content})
            self._append_assistant_message(content)
            return self._history[-1]

        raise RuntimeError("vLLM chat session ended without producing a response.")

    @property
    def _chat_history(self) -> Iterable[ChatMessage]:
        """Compatibility shim for legacy tests that access _chat_history."""
        return self.history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_client(self) -> OpenAI:
        kwargs = dict(self.config.client_kwargs or {})
        if self.config.api_key:
            kwargs.setdefault("api_key", self.config.api_key)
        kwargs.setdefault("base_url", self.config.base_url.rstrip("/"))
        return OpenAI(**kwargs)

    @staticmethod
    def _extract_text_from_chat(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if choices:
            first = choices[0]
            message = getattr(first, "message", None)
            if message:
                content = getattr(message, "content", None)
                if content:
                    return strip_hidden_reasoning(content)
                # Model returned tool_calls or empty content — return empty
                # rather than stringifying the raw ChatCompletion object.
                return ""
        content = getattr(response, "content", None)
        if content:
            return strip_hidden_reasoning(content)
        # Last resort: if response is already a plain string, return it.
        # Otherwise return empty to avoid leaking raw API objects.
        if isinstance(response, str):
            return response
        return ""

    @staticmethod
    def _extract_choice_message(response: Any) -> Any:
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")
        if not choices:
            return None
        first = choices[0]
        if isinstance(first, dict):
            return first.get("message")
        return getattr(first, "message", None)

    @staticmethod
    def _extract_message_content(message: Any) -> str:
        if isinstance(message, dict):
            return strip_hidden_reasoning(message.get("content") or "")
        return strip_hidden_reasoning(getattr(message, "content", "") or "")

    def _extract_tool_calls(self, message: Any) -> list[dict[str, Any]]:
        if isinstance(message, dict):
            raw_calls = message.get("tool_calls") or []
        else:
            raw_calls = getattr(message, "tool_calls", None) or []
        calls: list[dict[str, Any]] = []
        for idx, call in enumerate(raw_calls, start=1):
            if isinstance(call, dict):
                function = call.get("function") or {}
                name = function.get("name") or call.get("name")
                arguments_raw = function.get("arguments") or call.get("arguments")
                call_id = call.get("id") or f"call-{idx}"
            else:
                function = getattr(call, "function", None)
                name = getattr(function, "name", None) if function else getattr(call, "name", None)
                arguments_raw = getattr(function, "arguments", None) if function else getattr(call, "arguments", None)
                call_id = getattr(call, "id", None) or f"call-{idx}"
            if not name:
                continue
            arguments = self._decode_tool_arguments(arguments_raw)
            calls.append({"id": call_id, "name": name, "arguments": arguments})
        return calls

    @staticmethod
    def _decode_tool_arguments(arguments_raw: Any) -> Dict[str, Any]:
        if isinstance(arguments_raw, dict):
            return dict(arguments_raw)
        if isinstance(arguments_raw, str):
            text = arguments_raw.strip()
            if text:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {}
        return {}
