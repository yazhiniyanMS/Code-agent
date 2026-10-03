from types import SimpleNamespace as NS

import anthropic
import httpx2
import pytest

from ycode.config import Config
from ycode.errors import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMConnectionError,
    LLMNotFoundError,
    LLMRateLimitError,
    LLMResponseError,
    LLMServerError,
    MissingCredentialsError,
)
from ycode.llm import create_provider
from ycode.llm.anthropic import AnthropicProvider, has_credentials, translate_error
from ycode.llm.base import (
    AssistantMessage,
    StopReason,
    StreamCallbacks,
    ToolCall,
    ToolResultBlock,
    ToolResultsMessage,
    ToolSpec,
    UserMessage,
)

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
TOOLS = [ToolSpec("read_file", "Read a file", {"type": "object", "properties": {"path": {"type": "string"}},
                                               "required": ["path"]})]


def status_error(cls, code, headers=None):
    return cls("boom", response=httpx2.Response(code, request=REQUEST, headers=headers or {}), body=None)


class FakeStream:
    def __init__(self, events, final, error=None):
        self.events, self.final, self.error = events, final, error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        yield from self.events
        if self.error:
            raise self.error

    def get_final_message(self):
        return self.final


class FakeClient:
    def __init__(self, *streams):
        self.streams = list(streams)
        self.calls = []
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **params):
        self.calls.append(params)
        item = self.streams.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def final_message(content, stop_reason="end_turn", stop_details=None):
    return NS(content=content, stop_reason=stop_reason, stop_details=stop_details,
              usage=NS(input_tokens=100, output_tokens=20, cache_read_input_tokens=50,
                       cache_creation_input_tokens=None))


def test_streaming_text_and_tool_calls_are_parsed():
    content = [NS(type="thinking", thinking="", signature="sig"), NS(type="text", text="Reading."),
               NS(type="tool_use", id="toolu_1", name="read_file", input={"path": "a.py"})]
    events = [NS(type="text", text="Read"), NS(type="text", text="ing."),
              NS(type="content_block_start", content_block=NS(type="tool_use", name="read_file"))]
    client = FakeClient(FakeStream(events, final_message(content, "tool_use")))
    provider = AnthropicProvider("claude-opus-5-5", client=client)
    seen, started = [], []
    response = provider.complete(system="sys", messages=[UserMessage("hi")], tools=TOOLS,
                                 callbacks=StreamCallbacks(on_text=seen.append, on_tool_call_start=started.append))
    assert "".join(seen) == "Reading."
    assert started == ["read_file"]
    assert response.stop_reason is StopReason.TOOL_USE
    assert response.message.text == "Reading."
    assert response.message.tool_calls == [ToolCall("toolu_1", "read_file", {"path": "a.py"})]
    assert response.message.provider_data == content  # replayed verbatim, thinking included
    assert response.usage.input_tokens == 100 and response.usage.cache_read_tokens == 50


def test_request_shape_for_current_models():
    client = FakeClient(FakeStream([], final_message([NS(type="text", text="ok")])))
    provider = AnthropicProvider("claude-opus-5-5", effort="high", client=client)
    provider.complete(system="sys", messages=[UserMessage("hi")], tools=TOOLS)
    params = client.calls[0]
    assert params["model"] == "claude-opus-5-5"
    assert params["system"] == "sys"
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"] == {"effort": "high"}
    assert params["fallbacks"] == "default" and params["betas"] == ["server-side-fallback-2026-07-01"]
    assert params["cache_control"] == {"type": "ephemeral"}
    assert params["tools"][0]["input_schema"]["required"] == ["path"]
    assert params["tools"][0]["eager_input_streaming"] is True


def test_request_shape_for_older_models_omits_new_features():
    provider = AnthropicProvider("claude-haiku-4-5", client=FakeClient())
    params = provider.build_request("sys", [UserMessage("hi")], TOOLS)
    assert "thinking" not in params and "output_config" not in params and "fallbacks" not in params


def test_fallbacks_can_be_disabled():
    provider = AnthropicProvider("claude-opus-5-5", refusal_fallback=False, client=FakeClient())
    assert "fallbacks" not in provider.build_request("s", [UserMessage("x")], TOOLS)


def test_message_conversion_round_trip():
    provider = AnthropicProvider("claude-opus-5-5", client=FakeClient())
    native = [NS(type="text", text="x")]
    messages = [
        UserMessage("task"),
        AssistantMessage("calling", [ToolCall("t1", "read_file", {"path": "a"})], provider="anthropic",
                         provider_data=native),
        ToolResultsMessage([ToolResultBlock("t1", "contents"), ]),
        AssistantMessage("from another provider", [ToolCall("t2", "read_file", {"path": "b"})], provider="other"),
        ToolResultsMessage([ToolResultBlock("t2", "", is_error=True)]),
        UserMessage("follow-up"),
    ]
    out = provider.convert_messages(messages)
    assert [m["role"] for m in out] == ["user", "assistant", "user", "assistant", "user"]
    assert out[1]["content"] == native
    assert out[2]["content"][0] == {"type": "tool_result", "tool_use_id": "t1", "content": "contents"}
    assert out[3]["content"][1] == {"type": "tool_use", "id": "t2", "name": "read_file", "input": {"path": "b"}}
    # Tool results and the follow-up text are merged into one user turn.
    assert out[4]["content"][0]["is_error"] is True
    assert out[4]["content"][-1] == {"type": "text", "text": "follow-up"}


def test_refusal_stop_reason():
    client = FakeClient(FakeStream([], final_message([], "refusal", NS(explanation="policy", category="cyber"))))
    response = AnthropicProvider("claude-opus-5-5", client=client).complete(
        system="s", messages=[UserMessage("x")], tools=[])
    assert response.stop_reason is StopReason.REFUSAL and response.stop_detail == "policy"


def test_malformed_tool_json_is_retried_then_fails():
    ok = FakeStream([], final_message([NS(type="text", text="ok")]))
    client = FakeClient(FakeStream([], None, error=ValueError("bad json")), ok)
    response = AnthropicProvider("claude-opus-5-5", client=client).complete(
        system="s", messages=[UserMessage("x")], tools=[])
    assert response.message.text == "ok" and len(client.calls) == 2

    bad = [FakeStream([], None, error=ValueError("bad json")) for _ in range(3)]
    with pytest.raises(LLMResponseError):
        AnthropicProvider("claude-opus-5-5", client=FakeClient(*bad)).complete(
            system="s", messages=[UserMessage("x")], tools=[])


@pytest.mark.parametrize("exc, expected", [
    (status_error(anthropic.AuthenticationError, 401), LLMAuthenticationError),
    (status_error(anthropic.NotFoundError, 404), LLMNotFoundError),
    (status_error(anthropic.RateLimitError, 429, {"retry-after": "7"}), LLMRateLimitError),
    (status_error(anthropic.BadRequestError, 400), LLMBadRequestError),
    (status_error(anthropic.InternalServerError, 500), LLMServerError),
    (anthropic.APIConnectionError(request=REQUEST), LLMConnectionError),
    (anthropic.APITimeoutError(request=REQUEST), LLMConnectionError),
    (TypeError("Could not resolve authentication method"), MissingCredentialsError),
])
def test_api_errors_are_translated(exc, expected):
    client = FakeClient(exc)
    with pytest.raises(expected) as info:
        AnthropicProvider("claude-opus-5-5", client=client).complete(system="s", messages=[UserMessage("x")],
                                                                    tools=[])
    assert info.value.hint  # every translated error carries a helpful hint


def test_rate_limit_message_mentions_retry_after():
    err = translate_error(status_error(anthropic.RateLimitError, 429, {"retry-after": "7"}))
    assert "7s" in str(err)


def test_errors_raised_mid_stream_are_translated():
    client = FakeClient(FakeStream([NS(type="text", text="par")], None,
                                   error=anthropic.APIConnectionError(request=REQUEST)))
    with pytest.raises(LLMConnectionError):
        AnthropicProvider("claude-opus-5-5", client=client).complete(system="s", messages=[UserMessage("x")],
                                                                    tools=[])


def test_missing_api_key_fails_gracefully(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert not has_credentials()
    with pytest.raises(MissingCredentialsError) as info:
        create_provider(Config())
    assert "ANTHROPIC_API_KEY" in info.value.hint


def test_provider_created_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    provider = create_provider(Config(model="claude-sonnet-5-5", max_tokens=1000))
    assert isinstance(provider, AnthropicProvider)
    assert provider.model == "claude-sonnet-5-5" and provider.max_tokens == 1000
    assert "sk-ant" not in provider.display_name
