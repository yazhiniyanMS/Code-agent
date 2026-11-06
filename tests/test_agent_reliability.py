"""Agent reliability: stuck loops, duplicate calls in one response, blind or stale overwrites, writes that
fail part-way, and a task summary that does not call a failed run a success."""
import os

import pytest

from tests.conftest import reply
from tests.test_agent_loop import make_agent
from ycode.agent.loop import TaskStatus
from ycode.tools import ToolRegistry, default_tools


@pytest.fixture
def registry():
    return ToolRegistry(default_tools())


def _results(provider, request):
    """The tool results the model saw in its ``request``-th request."""
    return provider.requests[request]["messages"][-1].results


def test_the_same_failing_call_is_flagged_then_stopped(workspace, make_ctx):
    (workspace / "a.py").write_text("x = 1\n")
    bad = ("edit_file", {"path": "a.py", "old_string": "y = 1", "new_string": "y = 2"})
    agent, provider = make_agent(workspace, make_ctx, [reply(calls=[bad]) for _ in range(10)])
    result = agent.run("change y")
    assert result.status is TaskStatus.STUCK and result.steps == 3  # not the 50-step limit
    assert "has failed 2 times" in _results(provider, 2)[0].content  # the model was told before the stop
    notices = [e for e in agent.events.events if e[0] == "notice"]
    assert notices and "edit_file failed 3 times with the same input" in notices[-1][2]


def test_a_failure_that_is_fixed_does_not_count_towards_stopping(workspace, make_ctx):
    read = ("read_file", {"path": "late.py"})
    create = ("write_file", {"path": "late.py", "content": "ok = 1\n"})
    script = [reply(calls=[read]), reply(calls=[read]), reply(calls=[create]), reply(calls=[read]),
              reply(calls=[("read_file", {"path": "gone.py"})]), reply("done")]
    agent, _ = make_agent(workspace, make_ctx, script)
    assert agent.run("x").status is TaskStatus.COMPLETED


def test_identical_calls_in_one_response_run_once(workspace, make_ctx):
    once = ("run_command", {"command": "echo hi"})
    agent, provider = make_agent(workspace, make_ctx, [reply(calls=[once, once]), reply("done")])
    result = agent.run("x")
    assert result.commands_run == [("echo hi", 0)]
    second = _results(provider, 1)[1]
    assert second.is_error and "identical to call 1 in this response" in second.content


def test_write_file_never_overwrites_a_file_it_has_not_read(workspace, make_ctx, registry):
    (workspace / "notes.md").write_text("the user's notes\n")
    ctx = make_ctx(workspace)
    result = registry.dispatch("write_file", {"path": "notes.md", "content": "new\n"}, ctx)
    assert result.is_error and "not read" in result.content
    assert (workspace / "notes.md").read_text() == "the user's notes\n"
    assert not registry.dispatch("read_file", {"path": "notes.md"}, ctx).is_error
    assert not registry.dispatch("write_file", {"path": "notes.md", "content": "new\n"}, ctx).is_error
    assert (workspace / "notes.md").read_text() == "new\n"


def test_a_file_changed_since_it_was_read_is_not_edited_blind(workspace, make_ctx, registry):
    path = workspace / "m.py"
    path.write_text("a = 1\n")
    ctx = make_ctx(workspace)
    registry.dispatch("read_file", {"path": "m.py"}, ctx)
    path.write_text("a = 1\nb = 2  # added by the user meanwhile\n")
    for call in (("edit_file", {"path": "m.py", "old_string": "a = 1", "new_string": "a = 3"}),
                 ("write_file", {"path": "m.py", "content": "a = 3\n"})):
        result = registry.dispatch(*call, ctx)
        assert result.is_error and "changed since you last read it" in result.content
    assert path.read_text() == "a = 1\nb = 2  # added by the user meanwhile\n"
    registry.dispatch("read_file", {"path": "m.py"}, ctx)  # re-read: editing works again
    assert not registry.dispatch("edit_file", {"path": "m.py", "old_string": "a = 1", "new_string": "a = 3"},
                                 ctx).is_error


def test_the_agent_may_keep_editing_what_it_wrote(workspace, make_ctx, registry):
    ctx = make_ctx(workspace)
    assert not registry.dispatch("write_file", {"path": "new.py", "content": "a = 1\n"}, ctx).is_error
    assert not registry.dispatch("edit_file", {"path": "new.py", "old_string": "a = 1", "new_string": "a = 2"},
                                 ctx).is_error
    assert not registry.dispatch("write_file", {"path": "new.py", "content": "a = 3\n"}, ctx).is_error
    (workspace / "untouched.py").write_text("x = 1\n")  # editing a file never read still works: exact match
    assert not registry.dispatch("edit_file", {"path": "untouched.py", "old_string": "x = 1", "new_string": "x = 2"},
                                 ctx).is_error


def test_a_write_that_fails_part_way_leaves_the_file_as_it_was(workspace, make_ctx, registry, monkeypatch):
    path = workspace / "notes.py"
    path.write_text("important = 1\n" * 50)
    ctx = make_ctx(workspace)
    registry.dispatch("read_file", {"path": "notes.py"}, ctx)
    # a lone surrogate (JSON "\ud800") cannot be encoded: the file used to be truncated to 0 bytes first
    result = registry.dispatch("write_file", {"path": "notes.py", "content": "x = '\ud800'\n"}, ctx)
    assert result.is_error and path.read_text() == "important = 1\n" * 50
    from ycode.tools import filesystem

    def full_disk(fd):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(filesystem.os, "fsync", full_disk)
    result = registry.dispatch("write_file", {"path": "notes.py", "content": "smaller\n"}, ctx)
    assert result.is_error and "No space left" in result.content and "unchanged" in result.content
    assert path.read_text() == "important = 1\n" * 50
    assert sorted(os.listdir(workspace)) == ["notes.py"]  # no temporary file left behind


def test_a_write_keeps_the_file_mode(workspace, make_ctx, registry):
    script = workspace / "run.sh"
    script.write_text("echo 1\n")
    script.chmod(0o755)
    ctx = make_ctx(workspace)
    registry.dispatch("read_file", {"path": "run.sh"}, ctx)
    assert not registry.dispatch("edit_file", {"path": "run.sh", "old_string": "1", "new_string": "2"}, ctx).is_error
    assert script.stat().st_mode & 0o777 == 0o755 and script.read_text() == "echo 2\n"


def test_the_summary_does_not_call_a_failed_last_command_a_success(ui):
    from tests.conftest import console_text
    from ycode.agent.loop import TaskResult

    ui.print_task_summary(TaskResult(TaskStatus.COMPLETED, "All done!", 2,
                                     commands_run=[("pytest -q", 1)]), [])
    text = console_text(ui.console)
    assert "the last command failed" in text and "pytest -q" in text
    ui.print_task_summary(TaskResult(TaskStatus.COMPLETED, "All done!", 2, commands_run=[("pytest -q", 0)]), [])
    assert console_text(ui.console).count("the last command failed") == 1
