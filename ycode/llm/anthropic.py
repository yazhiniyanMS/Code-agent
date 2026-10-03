"""Anthropic Claude provider (Messages API with streaming tool use)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import anthropic

from ycode.errors import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMConnectionError,
    LLMError,
    LLMNotFoundError,
    LLMPermissionError,
    LLMRateLimitError,
    LLMResponseError,
    LLMServerError,
    MissingCredentialsError,
    YCodeError,
)
from ycode.llm.base import (
    AssistantMessage,
    LLMProvider,
    LLMResponse,
    Message,
    StopReason,
    StreamCallbacks,
    ToolCall,
    ToolResultsMessage,
    ToolSpec,
    Usage,
    UserMessage,
)

# Models that take `thinking: {"type": "adaptive"}` and `output_config.effort`.
_ADAPTIVE_PREFIXES = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
)
_NO_XHIGH_PREFIXES = ("claude-opus-4-6", "claude-sonnet-4-6")
# Models that accept server-side refusal fallbacks in the `"default"` form.
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

_STOP_REASONS = {
    "end_turn": StopReason.END_TURN,
    "stop_sequence": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
    "refusal": StopReason.REFUSAL,
    "pause_turn": StopReason.PAUSE,
}

_MAX_JSON_RETRIES = 2


def _base_model(model: str) -> str:
    return model.split(".", 1)[1] if model.startswith("anthropic.") else model


def supports_adaptive_thinking(model: str) -> bool:
    return _base_model(model).startswith(_ADAPTIVE_PREFIXES)


def supports_fallbacks(model: str) -> bool:
    return _base_model(model) in _FALLBACK_MODELS


def has_credentials(env: dict[str, str] | None = None) -> bool:
    """Best-effort check that the SDK will find some credential source."""
    env = dict(os.environ) if env is None else env
    if env.get("ANTHROPIC_API_KEY", "").strip() or env.get("ANTHROPIC_AUTH_TOKEN", "").strip():
        return True
    if env.get("ANTHROPIC_PROFILE") or (
        env.get("ANTHROPIC_FEDERATION_RULE_ID") and env.get("ANTHROPIC_IDENTITY_TOKEN_FILE")
    ):
        return True
    config_dir = Path(env.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "anthropic"
    return config_dir.is_dir() and any(config_dir.iterdir())


def translate_error(exc: Exception) -> YCodeError:
    """Map SDK exceptions to YCode errors (most specific first)."""
    if isinstance(exc, anthropic.AuthenticationError):
        return LLMAuthenticationError("Authentication failed: the API key was rejected.")
    if isinstance(exc, anthropic.PermissionDeniedError):
        return LLMPermissionError(f"Permission denied by the API: {exc.message}")
    if isinstance(exc, anthropic.NotFoundError):
        return LLMNotFoundError(f"Not found: {exc.message}")
    if isinstance(exc, anthropic.RateLimitError):
        retry_after = exc.response.headers.get("retry-after") if exc.response is not None else None
        suffix = f" Retry after {retry_after}s." if retry_after else ""
        return LLMRateLimitError(f"Rate limited by the API.{suffix}")
    if isinstance(exc, anthropic.BadRequestError):
        return LLMBadRequestError(f"The API rejected the request: {exc.message}")
    if isinstance(exc, anthropic.APIStatusError):
        if exc.status_code >= 500:
            return LLMServerError(f"API server error ({exc.status_code}): {exc.message}")
        return LLMError(f"API error ({exc.status_code}): {exc.message}")
    if isinstance(exc, anthropic.APITimeoutError):
        return LLMConnectionError("The request to the API timed out.")
    if isinstance(exc, anthropic.APIConnectionError):
        return LLMConnectionError("Could not connect to the Anthropic API.")
    if isinstance(exc, anthropic.APIResponseValidationError):
        return LLMResponseError(f"The API returned an unexpected response: {exc}")
    if isinstance(exc, TypeError) and "authentication" in str(exc).lower():
        return MissingCredentialsError("No Anthropic API credentials found.")
    return LLMError(f"Unexpected model error: {type(exc).__name__}: {exc}")


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 64000,
        effort: str | None = "high",
        refusal_fallback: bool = True,
        client: Any = None,
    ) -> None:
        super().__init__(model)
        self.max_tokens = max_tokens
        self.effort = effort
        self.refusal_fallback = refusal_fallback
        self._client = client
        # Through a custom gateway, stick to the plainest request shape.
        self._custom_base_url = bool(os.environ.get("ANTHROPIC_BASE_URL"))

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = anthropic.Anthropic(max_retries=3)
        return self._client

    @property
    def display_name(self) -> str:
        return f"Claude ({self.model})"

    # ---------------------------------------------------------- conversion

    def _tool_param(self, spec: ToolSpec) -> dict[str, Any]:
        param: dict[str, Any] = {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.parameters,
        }
        if not self._custom_base_url:
            # Stream large inputs (file contents) as they are generated. The
            # agent validates every input before running a tool.
            param["eager_input_streaming"] = True
        return param

    def convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []

        def push(role: str, content: Any) -> None:
            if out and out[-1]["role"] == role:
                prev = out[-1]["content"]
                if isinstance(prev, str):
                    prev = [{"type": "text", "text": prev}]
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                out[-1]["content"] = [*prev, *content]
            else:
                out.append({"role": role, "content": content})

        for msg in messages:
            if isinstance(msg, UserMessage):
                push("user", msg.text)
            elif isinstance(msg, ToolResultsMessage):
                push("user", [
                    {
                        "type": "tool_result",
                        "tool_use_id": r.tool_call_id,
                        "content": r.content or "(no output)",
                        **({"is_error": True} if r.is_error else {}),
                    }
                    for r in msg.results
                ])
            elif isinstance(msg, AssistantMessage):
                if msg.provider == self.name and msg.provider_data:
                    # Replay native blocks unchanged (keeps thinking blocks valid).
                    push("assistant", list(msg.provider_data))
                    continue
                blocks: list[dict[str, Any]] = []
                if msg.text:
                    blocks.append({"type": "text", "text": msg.text})
                for call in msg.tool_calls:
                    args = call.arguments if isinstance(call.arguments, dict) else {}
                    blocks.append({"type": "tool_use", "id": call.id, "name": call.name, "input": args})
                push("assistant", blocks or [{"type": "text", "text": "(no response)"}])
        return out

    def build_request(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": self.convert_messages(messages),
            "tools": [self._tool_param(t) for t in tools],
            # Automatic prompt caching: the growing conversation prefix is reused.
            "cache_control": {"type": "ephemeral"},
        }
        if supports_adaptive_thinking(self.model):
            params["thinking"] = {"type": "adaptive"}
            if self.effort:
                effort = self.effort
                if effort == "xhigh" and _base_model(self.model).startswith(_NO_XHIGH_PREFIXES):
                    effort = "high"
                params["output_config"] = {"effort": effort}
        if self.refusal_fallback and not self._custom_base_url and supports_fallbacks(self.model):
            params["fallbacks"] = "default"
            params["betas"] = [_FALLBACK_BETA]
        return params

    # ------------------------------------------------------------- request

    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        callbacks: StreamCallbacks | None = None,
    ) -> LLMResponse:
        callbacks = callbacks or StreamCallbacks()
        params = self.build_request(system, messages, tools)
        json_failures = 0
        while True:
            try:
                with self.client.beta.messages.stream(**params) as stream:
                    for event in stream:
                        etype = getattr(event, "type", None)
                        if etype == "text":
                            callbacks.text(event.text)
                        elif etype == "content_block_start":
                            block = getattr(event, "content_block", None)
                            if getattr(block, "type", None) == "tool_use":
                                callbacks.tool_call_start(block.name)
                    final = stream.get_final_message()
                break
            except ValueError as exc:
                # Tool-input JSON the SDK could not parse at all: re-issue the turn.
                if isinstance(exc, anthropic.AnthropicError):
                    raise translate_error(exc) from None
                json_failures += 1
                if json_failures > _MAX_JSON_RETRIES:
                    raise LLMResponseError("The model produced malformed tool-call JSON repeatedly.") from None
            except (anthropic.AnthropicError, TypeError) as exc:
                raise translate_error(exc) from None
        return self._parse(final)

    def _parse(self, final: Any) -> LLMResponse:
        content = list(getattr(final, "content", None) or [])
        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in content:
            btype = getattr(block, "type", None)
            if btype == "text":
                texts.append(block.text)
            elif btype == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))
        raw_stop = getattr(final, "stop_reason", None)
        stop = _STOP_REASONS.get(raw_stop or "", StopReason.OTHER)
        detail = None
        if stop is StopReason.REFUSAL:
            details = getattr(final, "stop_details", None)
            detail = getattr(details, "explanation", None) or getattr(details, "category", None)
        usage_obj = getattr(final, "usage", None)
        usage = Usage(
            input_tokens=getattr(usage_obj, "input_tokens", 0) or 0,
            output_tokens=getattr(usage_obj, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage_obj, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage_obj, "cache_creation_input_tokens", 0) or 0,
        )
        message = AssistantMessage(
            text="\n\n".join(t for t in texts if t),
            tool_calls=calls,
            provider=self.name,
            provider_data=content,
        )
        return LLMResponse(message=message, stop_reason=stop, usage=usage, stop_detail=detail)
