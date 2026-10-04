import pytest

from ycode.config import Config
from ycode.errors import ConfigError
from ycode.llm import create_provider
from ycode.llm.base import AssistantMessage, StopReason, StreamCallbacks, ToolResultBlock, ToolResultsMessage, UserMessage
from ycode.llm.local import LocalProvider



class FakeLM:
    num_params = 7_000_000

    def __init__(self, answer="Use `sorted(items)`."):
        self.answer = answer
        self.calls = []

    def chat(self, turns, *, max_new_tokens, temperature, on_text=None):
        self.calls.append(turns)
        if on_text:
            on_text(self.answer)
        return self.answer


def test_local_provider_answers_without_tools(tmp_path):
    lm = FakeLM()
    provider = LocalProvider(tmp_path, lm=lm)
    streamed = []
    response = provider.complete(
        system="ignored",
        messages=[
            UserMessage("first question"),
            AssistantMessage("first answer", provider="local"),
            ToolResultsMessage([ToolResultBlock("x", "tool output")]),
            UserMessage("[Note: the user interrupted the previous request before it finished.]\n\nhow to sort?"),
        ],
        tools=[],
        callbacks=StreamCallbacks(on_text=streamed.append),
    )
    assert response.stop_reason is StopReason.END_TURN
    assert response.message.text == lm.answer and response.message.tool_calls == []
    assert "".join(streamed) == lm.answer
    assert lm.calls[0] == [("user", "first question"), ("assistant", "first answer"), ("user", "how to sort?")]
    assert provider.supports_tools is False
    assert "7.0M params" in provider.display_name


def test_empty_answer_gets_placeholder(tmp_path):
    response = LocalProvider(tmp_path, lm=FakeLM("")).complete(system="", messages=[UserMessage("x")], tools=[])
    assert "did not produce an answer" in response.message.text


def test_missing_local_model_is_a_clear_error(tmp_path):
    pytest.importorskip("torch")
    with pytest.raises(ConfigError) as info:
        create_provider(Config(provider="local", local_model=str(tmp_path / "none")))
    assert "No trained model" in str(info.value) and "ycode-lm" in info.value.hint


def test_no_api_key_needed_for_local(tmp_path, monkeypatch):
    import ycode.llm.local as local

    monkeypatch.setattr(local.LocalProvider, "_load", lambda self, device: FakeLM())
    provider = create_provider(Config(provider="local", local_model=str(tmp_path)))
    assert isinstance(provider, LocalProvider) and provider.model_dir == tmp_path


def test_config_validation_for_local(workspace):
    from ycode.config import load_config

    cfg = load_config(workspace, env={"YCODE_PROVIDER": "local", "YCODE_LOCAL_MODEL": "/models/mine"})
    assert cfg.provider == "local" and cfg.local_model == "/models/mine"
    with pytest.raises(ConfigError):
        load_config(workspace, env={"YCODE_PROVIDER": "openai"})


def test_cli_local_flag_runs_task_without_api_key(workspace, monkeypatch, capsys):
    import ycode.llm.local as local
    from ycode import cli

    monkeypatch.setattr(local.LocalProvider, "_load", lambda self, device: FakeLM("def add(a, b): return a + b"))
    code = cli.main(["-C", str(workspace), "--local", str(workspace / "m"), "write add"])
    out = capsys.readouterr().out
    assert code == 0
    assert "answer-only mode" in out and "def add(a, b)" in out and "YCode-LM" in out


def test_agent_sends_no_tools_to_local_provider(workspace, make_ctx, tmp_path):
    from ycode.agent.loop import Agent
    from ycode.tools import ToolRegistry, default_tools

    seen = {}

    class Recorder(LocalProvider):
        def complete(self, *, system, messages, tools, callbacks=None):
            seen["tools"] = tools
            return super().complete(system=system, messages=messages, tools=tools, callbacks=callbacks)

    agent = Agent(provider=Recorder(tmp_path, lm=FakeLM()), registry=ToolRegistry(default_tools()),
                  tool_context=make_ctx(workspace), system_prompt="s")
    agent.run("hi")
    assert seen["tools"] == []
