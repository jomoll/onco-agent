import json
from typing import Any
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

try:  # pragma: no cover - optional dependencies
    from src.agent_vllm import VLLMChatAgent, VLLMConfig
    from src.agent_gemini import GeminiChatAgent, GeminiConfig
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"Agent tests require optional dependencies: {exc}", allow_module_level=True)

from src.toolkit import BaseTool, ToolMetadata, ToolOutput


class WordCountTool(BaseTool):
    def __init__(self) -> None:
        self._metadata = ToolMetadata(
            name="word_count",
            description="Count how many whitespace-delimited words appear in the text.",
        )

    @property
    def metadata(self) -> ToolMetadata:  # type: ignore[override]
        return self._metadata

    def __call__(self, *, text: str | None = None, **kwargs: Any) -> ToolOutput:  # type: ignore[override]
        words = (text or "").split()
        payload = {"word_count": len(words)}
        return ToolOutput(
            tool_name=self._metadata.name,
            content=json.dumps(payload),
            raw_input={"kwargs": {"text": text, **kwargs}},
            raw_output=payload,
        )


class SummationTool(BaseTool):
    def __init__(self) -> None:
        self._metadata = ToolMetadata(
            name="sum_numbers",
            description="Return the arithmetic sum of the provided numbers array.",
        )

    @property
    def metadata(self) -> ToolMetadata:  # type: ignore[override]
        return self._metadata

    def __call__(self, *, numbers: list[float] | None = None, **_: Any) -> ToolOutput:  # type: ignore[override]
        numbers = numbers or []
        payload = {"sum": sum(numbers)}
        return ToolOutput(
            tool_name=self._metadata.name,
            content=json.dumps(payload),
            raw_input={"kwargs": {"numbers": numbers}},
            raw_output=payload,
        )


class EchoTool(BaseTool):
    def __init__(self) -> None:
        self._metadata = ToolMetadata(
            name="echo",
            description="Return whatever arguments were provided.",
        )

    @property
    def metadata(self) -> ToolMetadata:  # type: ignore[override]
        return self._metadata

    def __call__(self, **kwargs: Any) -> ToolOutput:  # type: ignore[override]
        return ToolOutput(
            tool_name=self._metadata.name,
            content=json.dumps({"echo": kwargs}),
            raw_input={"kwargs": kwargs},
            raw_output={"echo": kwargs},
        )


class DummyCompletions:
    def __init__(self):
        self.pending: list[Any] = []

    def create(self, **kwargs):
        del kwargs
        if not self.pending:
            raise AssertionError("No dummy responses queued for VLLM client.")
        return self.pending.pop(0)


class DummyOpenAI:
    def __init__(self, *_, **__):
        self.chat = SimpleNamespace(completions=DummyCompletions())


def _fake_vllm_response(content: str | None, tool_calls: list[dict] | None):
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                    **({"tool_calls": tool_calls} if tool_calls else {}),
                }
            }
        ]
    }


@pytest.fixture
def dummy_vllm_client(monkeypatch):
    client = DummyOpenAI()
    from src import agent_vllm

    monkeypatch.setattr(agent_vllm, "OpenAI", lambda **kwargs: client)
    return client


def test_vllm_agent_executes_tool_and_returns_final_message(dummy_vllm_client):
    tool = WordCountTool()
    config = VLLMConfig(base_url="http://dummy", api_key="key", model="mock")
    agent = VLLMChatAgent(config=config, tools=[tool])

    first = _fake_vllm_response(
        content=None,
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": tool.metadata.name,
                    "arguments": json.dumps({"text": "hi there"}),
                },
            }
        ],
    )
    second = _fake_vllm_response(content="final reply", tool_calls=None)

    dummy_vllm_client.chat.completions.pending = [first, second]

    reply = agent.chat("hello")

    assert reply.content == "final reply"
    first_message = agent._chat_history[0]  # type: ignore[attr-defined]
    assert first_message.role == MessageRole.USER
    assert first_message.content == "hello"
    # Tool output should have been appended for the second model call
    tool_messages = [m for m in agent._chat_history if m.role == MessageRole.TOOL]  # type: ignore[attr-defined]
    assert len(tool_messages) == 1
    assert json.loads(tool_messages[0].content)["word_count"] == 2


def test_vllm_agent_handles_tool_error(dummy_vllm_client):
    class FailingTool(EchoTool):
        def __call__(self, **kwargs):  # type: ignore[override]
            raise RuntimeError("boom")

    tool = FailingTool()
    config = VLLMConfig(base_url="http://dummy", api_key="key", model="mock")
    agent = VLLMChatAgent(config=config, tools=[tool])

    first = _fake_vllm_response(
        content=None,
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": tool.metadata.name,
                    "arguments": "{}",
                },
            }
        ],
    )
    second = _fake_vllm_response(content="done", tool_calls=None)

    dummy_vllm_client.chat.completions.pending = [first, second]
    reply = agent.chat("hello")

    assert reply.content == "done"
    # Tool should have been attempted once; error response is captured as ToolOutput.
    tool_messages = [m for m in agent._chat_history if m.role == MessageRole.TOOL]  # type: ignore[attr-defined]
    assert tool_messages
    assert "Tool error" in tool_messages[0].content


def _make_gemini_response(text: str, tool_call: SimpleNamespace | None):
    parts = []
    if tool_call is not None:
        parts.append(SimpleNamespace(function_call=tool_call.function_call))
    if text:
        parts.append(SimpleNamespace(text=text))
    candidate = SimpleNamespace(content=SimpleNamespace(parts=parts))
    return SimpleNamespace(candidates=[candidate])


@patch("src.agent_gemini.configure")
def test_gemini_agent_executes_tool(mock_configure):
    tool = SummationTool()
    config = GeminiConfig(api_key="dummy", model="gemini-test")

    mock_model = MagicMock()
    tool_call = SimpleNamespace(
        function_call=SimpleNamespace(
            name=tool.metadata.name, args_json=json.dumps({"numbers": [1, 2, 3]})
        )
    )

    # First call requests the tool; second call returns final text.
    mock_model.generate_content.side_effect = [
        _make_gemini_response("", tool_call),
        _make_gemini_response("completed", None),
    ]

    with patch(
        "src.agent_gemini.GenerativeModel",
        return_value=mock_model,
    ):
        agent = GeminiChatAgent(config=config, tools=[tool])
        reply = agent.chat("summarize patient")

    assert reply.content == "completed"
    tool_messages = [m for m in agent._chat_history if m.role == MessageRole.TOOL]  # type: ignore[attr-defined]
    assert tool_messages
    assert json.loads(tool_messages[0].content)["sum"] == 6
    assert mock_configure.called
