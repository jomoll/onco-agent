"""Utility module exposing tool primitives with optional llama_index fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


try:  # pragma: no cover - exercised when llama_index is installed
    from llama_index.core.tools import BaseTool as _BaseTool  # type: ignore
    from llama_index.core.tools import ToolOutput as _ToolOutput  # type: ignore
    from llama_index.core.tools.types import ToolMetadata as _ToolMetadata  # type: ignore

    BaseTool = _BaseTool
    ToolOutput = _ToolOutput
    ToolMetadata = _ToolMetadata
except Exception:  # pragma: no cover - minimal local fallback
    @dataclass
    class ToolMetadata:  # type: ignore[override]
        name: str
        description: str

        def to_openai_tool(self) -> Dict[str, Any]:
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            }

    @dataclass
    class ToolOutput:  # type: ignore[override]
        tool_name: str
        content: str
        raw_input: Dict[str, Any]
        raw_output: Any
        is_error: bool = False

        def __str__(self) -> str:  # to mimic llama_index representation
            return self.content

    class BaseTool:  # type: ignore[override]
        def __call__(self, *args, **kwargs):
            raise NotImplementedError

        @property
        def metadata(self) -> ToolMetadata:  # type: ignore[override]
            raise NotImplementedError

