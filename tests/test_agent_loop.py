from ycode.agent.loop import Agent, AgentEvents, TaskStatus
from ycode.errors import LLMRateLimitError
from ycode.llm.base import AssistantMessage, StopReason, ToolResultsMessage, UserMessage
from ycode.tools import ToolRegistry, default_tools

from tests.conftest import FakeProvider, reply


class RecordingEvents(AgentEvents):
    def __init__(self):
        self.events = []

    def on_tool_start(self, call, description):
        self.events.append(("tool_start", call.name, description))

    def on_tool_end(self, call, result):
        self.events.append(("tool_end", call.name, result.is_error))

    def on_plan(self, plan):
        self.events.append(("plan", [s.text for s in plan.steps]))

    def on_notice(self, message, level="info"):
        self.events.append(("notice", level, message))

    def on_text(self, chunk):
        self.events.append(("text", chunk))


def make_agent(workspace, make_ctx, responses, max_steps=50, events=None):
    provider = FakeProvider(responses)
    agent = Agent(provider=provider, registry=ToolRegistry(default_tools()), tool_context=make_ctx(workspace),
                  system_prompt="SYSTEM", max_steps=max_steps, events=events or RecordingEvents())
    return agent, provider


def test_multi_step_task_with_tools(workspace, make_ctx):
    (workspace / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    events = RecordingEvents()
    agent, provider = make_agent(workspace, make_ctx, [
        reply("Let me plan.", [("update_plan", {"steps": [{"step": "Read calc.py"}, {"step": "Fix bug"}]})]),
        reply("", [("read_file", {"path": "calc.py"})]),
        reply("", [("edit_file", {"path": "calc.py", "old_string": "a - b", "new_string": "a + b"})]),
        reply("", [("run_command", {"command": "python -c \"import calc; assert calc.add(2, 3) == 5\""})]),
        reply("Fixed `add` and verified it."),
    ], events=events)
    result = agent.run("Fix the add function")

    assert result.status is TaskStatus.COMPLETED
    assert result.steps == 5
    assert result.final_text == "Fixed `add` and verified it."
    assert (workspace / "calc.py").read_text().endswith("return a + b\n")
    assert result.changed_files == ["calc.py"]
    assert result.commands_run[0][1] == 0
    assert ("plan", ["Read calc.py", "Fix bug"]) in events.events
    assert [e[1] for e in events.events if e[0] == "tool_start"] == [
        "update_plan", "read_file", "edit_file", "run_command"]

    # The model saw the tool results on its next turn.
    second_request = provider.requests[2]["messages"]
    assert isinstance(second_request[-1], ToolResultsMessage)
    assert "return a - b" in second_request[-1].results[0].content
    assert provider.requests[0]["system"] == "SYSTEM"
    assert {t.name for t in provider.requests[0]["tools"]} >= {"read_file", "run_command", "update_plan"}


def test_parallel_tool_calls_return_results_in_one_message(workspace, make_ctx):
    (workspace / "a.txt").write_text("A")
    (workspace / "b.txt").write_text("B")
    agent, provider = make_agent(workspace, make_ctx, [
        reply("", [("read_file", {"path": "a.txt"}), ("read_file", {"path": "b.txt"})]),
        reply("done"),
    ])
    agent.run("read both")
    results = provider.requests[1]["messages"][-1]
    assert isinstance(results, ToolResultsMessage) and len(results.results) == 2
    assert [r.tool_call_id for r in results.results] == ["call_0_read_file", "call_1_read_file"]


def test_tool_errors_are_returned_to_model(workspace, make_ctx):
    agent, provider = make_agent(workspace, make_ctx, [
        reply("", [("read_file", {"path": "missing.py"})]),
        reply("", [("no_such_tool", {})]),
        reply("", [("read_file", {"wrong": 1})]),
        reply("I could not find it."),
    ])
    result = agent.run("look")
    assert result.status is TaskStatus.COMPLETED
    errors = [m.results[0] for m in provider.requests[-1]["messages"] if isinstance(m, ToolResultsMessage)]
    assert all(r.is_error for r in errors) and len(errors) == 3


def test_step_limit(workspace, make_ctx):
    events = RecordingEvents()
    agent, _ = make_agent(workspace, make_ctx, [reply("", [("list_directory", {})]) for _ in range(10)],
                          max_steps=3, events=events)
    result = agent.run("loop forever")
    assert result.status is TaskStatus.STEP_LIMIT
    assert result.steps == 3
    assert any(e[0] == "notice" and "step limit" in e[2] for e in events.events)
    assert agent.last_steps == 3


def test_llm_error_stops_task_cleanly(workspace, make_ctx):
    agent, _ = make_agent(workspace, make_ctx, [LLMRateLimitError("Rate limited")])
    result = agent.run("anything")
    assert result.status is TaskStatus.ERROR
    assert isinstance(result.error, LLMRateLimitError)
    # The conversation can continue afterwards.
    agent.provider.responses = [reply("ok")]
    assert agent.run("again").status is TaskStatus.COMPLETED


def test_interrupt_during_model_call(workspace, make_ctx):
    agent, provider = make_agent(workspace, make_ctx, [KeyboardInterrupt(), reply("resumed")])
    assert agent.run("first").status is TaskStatus.INTERRUPTED
    agent.run("second")
    last_user = provider.requests[-1]["messages"][-1]
    assert isinstance(last_user, UserMessage) and "interrupted" in last_user.text and "second" in last_user.text


def test_interrupt_during_tool_keeps_history_valid(workspace, make_ctx, monkeypatch):
    from ycode.tools import filesystem

    def interrupted(self, args, ctx):
        raise KeyboardInterrupt

    monkeypatch.setattr(filesystem.ListDirectoryTool, "run", interrupted)
    agent, _ = make_agent(workspace, make_ctx, [
        reply("", [("list_directory", {}), ("read_file", {"path": "x"})]),
    ])
    result = agent.run("list")
    assert result.status is TaskStatus.INTERRUPTED
    last = agent.messages[-1]
    assert isinstance(last, ToolResultsMessage)
    # Every tool call has a result, so the next request is well-formed.
    assert [r.tool_call_id for r in last.results] == ["call_0_list_directory", "call_1_read_file"]
    assert all(r.is_error for r in last.results)


def test_truncated_tool_call_not_executed(workspace, make_ctx):
    agent, provider = make_agent(workspace, make_ctx, [
        reply("", [("write_file", {"path": "big.txt", "content": "partial"})], stop=StopReason.MAX_TOKENS),
        reply("ok, smaller pieces"),
    ])
    agent.run("write a big file")
    assert not (workspace / "big.txt").exists()
    results = provider.requests[1]["messages"][-1]
    assert results.results[0].is_error and "truncated" in results.results[0].content


def test_max_tokens_text_is_continued(workspace, make_ctx):
    agent, provider = make_agent(workspace, make_ctx, [
        reply("part one", stop=StopReason.MAX_TOKENS), reply("part two"),
    ])
    result = agent.run("long answer")
    assert result.status is TaskStatus.COMPLETED and result.steps == 2
    assert "cut off" in provider.requests[1]["messages"][-1].text


def test_refusal(workspace, make_ctx):
    events = RecordingEvents()
    agent, _ = make_agent(workspace, make_ctx, [reply("", stop=StopReason.REFUSAL)], events=events)
    result = agent.run("something")
    assert result.status is TaskStatus.REFUSED
    assert not any(isinstance(m, AssistantMessage) for m in agent.messages)


def test_reset_clears_history(workspace, make_ctx):
    agent, _ = make_agent(workspace, make_ctx, [reply("hi")])
    agent.run("hello")
    assert agent.messages
    agent.reset()
    assert agent.messages == [] and agent.plan.is_empty


def test_usage_accumulates(workspace, make_ctx):
    agent, _ = make_agent(workspace, make_ctx, [reply("", [("list_directory", {})]), reply("done")])
    agent.run("x")
    assert agent.usage.input_tokens == 20 and agent.usage.output_tokens == 10
