"""The conversation sent to the model stays within a budget: the oldest tool outputs give way, the newest and
every user and assistant message stay whole. Without this a long session outgrows the model's context and
every request from then on fails."""
from tests.conftest import FakeProvider, reply
from tests.test_agent_loop import RecordingEvents
from ycode.agent.loop import Agent, TaskStatus, _size
from ycode.llm.base import ToolResultsMessage
from ycode.tools import ToolRegistry, default_tools


def _agent(workspace, make_ctx, responses, **kw):
    provider = FakeProvider(responses)
    agent = Agent(provider=provider, registry=ToolRegistry(default_tools()), tool_context=make_ctx(workspace),
                  system_prompt="SYSTEM", events=RecordingEvents(), **kw)
    return agent, provider


def _reads(workspace, n):
    for i in range(n):
        (workspace / f"f{i}.txt").write_text(f"file {i}\n" + "x" * 3000 + "\n")
    return [reply(calls=[("read_file", {"path": f"f{i}.txt"})]) for i in range(n)] + [reply("done")]


def test_old_tool_outputs_give_way_to_stay_within_the_budget(workspace, make_ctx):
    agent, provider = _agent(workspace, make_ctx, _reads(workspace, 6), max_context_chars=8000)
    assert agent.run("read the six files").status is TaskStatus.COMPLETED
    sizes = [sum(_size(m) for m in r["messages"]) for r in provider.requests]
    assert max(sizes) <= 8000, sizes
    last = provider.requests[-1]["messages"]
    results = [m.results[0].content for m in last if isinstance(m, ToolResultsMessage)]
    assert results[0].startswith("[Output removed to keep the conversation within the model's context")
    assert "file 4" in results[-2] and "file 5" in results[-1]  # the two newest stay whole
    assert last[0].text == "read the six files"
    assert any(e[0] == "notice" and "Shortened older tool outputs" in e[2] for e in agent.events.events)
    first = provider.requests[2]["messages"]  # what was already sent stays as it was sent
    assert "file 0" in [m for m in first if isinstance(m, ToolResultsMessage)][0].results[0].content


def test_nothing_is_shortened_within_the_budget(workspace, make_ctx):
    agent, provider = _agent(workspace, make_ctx, _reads(workspace, 6))
    agent.run("read the six files")
    contents = [r.content for m in provider.requests[-1]["messages"] if isinstance(m, ToolResultsMessage)
                for r in m.results]
    assert len(contents) == 6 and not any(c.startswith("[Output removed") for c in contents)
    assert not any(e[0] == "notice" for e in agent.events.events)
