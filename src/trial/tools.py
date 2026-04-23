from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    name: str
    input: dict
    output: Any | None = None

    @classmethod
    def from_anthropic(cls, block: Any) -> ToolCall:
        """Convert an Anthropic tool_use content block to a ToolCall."""
        return cls(name=block.name, input=block.input)

    @classmethod
    def from_openai(cls, tool_call: Any) -> ToolCall:
        """Convert an OpenAI ChatCompletionMessageToolCall to a ToolCall."""
        return cls(
            name=tool_call.function.name,
            input=json.loads(tool_call.function.arguments),
        )
