"""End-to-end test of the real Anthropic SDK path against a mock HTTP transport.

No network and no API key: the transport returns canned server-sent events,
and we inspect the JSON request bodies the SDK actually sends.
"""

import json

import anthropic
import httpx2

from ycode.agent.loop import Agent, TaskStatus
from ycode.llm.anthropic import AnthropicProvider
from ycode.tools import ToolRegistry, default_tools


def sse(*events: dict) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def message_start():
    return {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 12, "output_tokens": 1}}}


TOOL_TURN = sse(
    message_start(),
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me look."}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1,
     "content_block": {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {}}},
    {"type": "content_block_delta", "index": 1,
     "delta": {"type": "input_json_delta", "partial_json": "{\"path\": \"hello.txt\"}"}},
    {"type": "content_block_stop", "index": 1},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
     "usage": {"output_tokens": 30}},
    {"type": "message_stop"},
)

FINAL_TURN = sse(
    message_start(),
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "It says hi."}},
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
     "usage": {"output_tokens": 5}},
    {"type": "message_stop"},
)


def test_full_round_trip_through_sdk(workspace, make_ctx):
    (workspace / "hello.txt").write_text("hi\n")
    bodies, headers = [], []
    turns = [TOOL_TURN, FINAL_TURN]

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        headers.append(dict(request.headers))
        return httpx2.Response(200, content=turns.pop(0), headers={"content-type": "text/event-stream"})

    client = anthropic.Anthropic(api_key="sk-ant-test", max_retries=0,
                                 http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    provider = AnthropicProvider("claude-opus-5-5", client=client)
    agent = Agent(provider=provider, registry=ToolRegistry(default_tools()), tool_context=make_ctx(workspace),
                  system_prompt="You are YCode.")
    result = agent.run("What does hello.txt say?")

    assert result.status is TaskStatus.COMPLETED
    assert result.final_text == "It says hi."
    assert agent.usage.output_tokens == 35

    first, second = bodies
    assert first["model"] == "claude-opus-5-5"
    assert first["stream"] is True
    assert first["thinking"] == {"type": "adaptive"}
    assert first["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in headers[0]["anthropic-beta"]
    assert first["messages"] == [{"role": "user", "content": "What does hello.txt say?"}]
    assert any(t["name"] == "read_file" for t in first["tools"])

    # The second request replays the assistant turn and carries the tool result.
    assistant, tool_results = second["messages"][1], second["messages"][2]
    assert assistant["role"] == "assistant"
    assert assistant["content"][1]["type"] == "tool_use"
    assert assistant["content"][1]["input"] == {"path": "hello.txt"}
    block = tool_results["content"][0]
    assert block["type"] == "tool_result" and block["tool_use_id"] == "toolu_1"
    assert "1\thi" in block["content"]
    assert "sk-ant-test" not in json.dumps(second)


def test_http_errors_through_sdk(workspace):
    import pytest

    from ycode.errors import LLMAuthenticationError
    from ycode.llm.base import UserMessage

    def handler(request):
        return httpx2.Response(401, json={"type": "error", "error": {"type": "authentication_error",
                                                                     "message": "invalid x-api-key"}})

    client = anthropic.Anthropic(api_key="bad", max_retries=0,
                                 http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    with pytest.raises(LLMAuthenticationError):
        AnthropicProvider("claude-opus-5-5", client=client).complete(system="s", messages=[UserMessage("x")],
                                                                    tools=[])
