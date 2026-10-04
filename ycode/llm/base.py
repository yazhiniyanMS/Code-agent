"""Provider-neutral LLM interface.

The agent only speaks in these types. A provider translates them to and from
its own wire format. Assistant messages carry an opaque ``provider_data``
blob so a provider can replay its native content (e.g. thinking blocks)
exactly as it produced it; other providers ignore it and fall back to the
neutral text/tool_calls fields.
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Union


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Any  # normally a dict; validated before execution


@dataclass
class UserMessage:
    text: str


@dataclass
class AssistantMessage:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    provider: str | None = None
    provider_data: Any = None


@dataclass
class ToolResultBlock:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class ToolResultsMessage:
    results: list[ToolResultBlock]


Message = Union[UserMessage, AssistantMessage, ToolResultsMessage]


class StopReason(enum.Enum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    PAUSE = "pause"
    OTHER = "other"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens


@dataclass
class LLMResponse:
    message: AssistantMessage
    stop_reason: StopReason
    usage: Usage = field(default_factory=Usage)
    stop_detail: str | None = None  # e.g. refusal explanation


@dataclass
class StreamCallbacks:
    """Optional hooks a UI can use to show progress while a response streams."""

    on_text: Callable[[str], None] | None = None
    on_tool_call_start: Callable[[str], None] | None = None

    def text(self, chunk: str) -> None:
        if self.on_text:
            self.on_text(chunk)

    def tool_call_start(self, name: str) -> None:
        if self.on_tool_call_start:
            self.on_tool_call_start(name)


class LLMProvider(abc.ABC):
    """A chat model that supports tool calling."""

    name: str = "provider"
    # False for models that can only answer in text (no tool calling).
    supports_tools: bool = True

    def __init__(self, model: str) -> None:
        self.model = model

    @property
    def display_name(self) -> str:
        return self.model

    @abc.abstractmethod
    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        callbacks: StreamCallbacks | None = None,
    ) -> LLMResponse:
        """Generate the next assistant turn.

        Raises :class:`ycode.errors.LLMError` subclasses for provider failures.
        """
