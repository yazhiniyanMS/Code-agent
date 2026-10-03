import sys

from ycode.tools import ToolRegistry, default_tools
from ycode.tools.terminal import RunCommandTool, execute

PY = sys.executable


def test_execute_captures_streams_exit_code_and_duration(workspace):
    out = execute(f'{PY} -c "import sys; print(\'out\'); print(\'err\', file=sys.stderr); sys.exit(3)"',
                  workspace, timeout=30)
    assert out.stdout.strip() == "out"
    assert out.stderr.strip() == "err"
    assert out.exit_code == 3
    assert out.duration >= 0
    assert not out.timed_out


def test_execute_runs_in_workspace(workspace):
    out = execute(f'{PY} -c "import os; print(os.getcwd())"', workspace, timeout=30)
    assert out.stdout.strip() == str(workspace.resolve())


def test_execute_timeout(workspace):
    out = execute(f'{PY} -c "import time; time.sleep(10)"', workspace, timeout=1)
    assert out.timed_out and out.exit_code is None


def test_api_key_not_passed_to_children(workspace, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")
    out = execute(f'{PY} -c "import os; print(os.environ.get(\'ANTHROPIC_API_KEY\', \'absent\'))"',
                  workspace, timeout=30)
    assert out.stdout.strip() == "absent"


def test_run_command_tool_reports_failure(workspace, make_ctx):
    ctx = make_ctx(workspace)
    result = RunCommandTool().run({"command": f'{PY} -c "raise SystemExit(2)"'}, ctx)
    assert "exit code 2" in result.content
    assert result.display["failed"] is True
    assert ctx.commands_run[-1][1] == 2


def test_run_command_success(workspace, make_ctx):
    result = RunCommandTool().run({"command": "echo hello"}, make_ctx(workspace))
    assert "hello" in result.content and "exit code 0" in result.content
    assert result.summary.startswith("Succeeded")


def test_blocked_command_never_runs(workspace, make_ctx, approver):
    registry = ToolRegistry(default_tools())
    result = registry.dispatch("run_command", {"command": "rm -rf /"}, make_ctx(workspace))
    assert result.is_error and "blocked" in result.content
    assert approver.requests == []


def test_approval_denied(workspace, make_ctx, approver):
    (workspace / "keep.txt").write_text("x")
    registry = ToolRegistry(default_tools())
    result = registry.dispatch("run_command", {"command": "rm keep.txt"}, make_ctx(workspace))
    assert result.is_error and "declined" in result.content
    assert (workspace / "keep.txt").exists()
    assert approver.requests[0].subject == "rm keep.txt"


def test_approval_granted(workspace, make_ctx, approver):
    approver.answer = "yes"
    (workspace / "gone.txt").write_text("x")
    registry = ToolRegistry(default_tools())
    result = registry.dispatch("run_command", {"command": "rm gone.txt"}, make_ctx(workspace))
    assert not result.is_error
    assert not (workspace / "gone.txt").exists()


def test_output_is_truncated(workspace, make_ctx):
    ctx = make_ctx(workspace, max_output_chars=1000)
    result = RunCommandTool().run({"command": f'{PY} -c "print(\'x\' * 50000)"'}, ctx)
    assert "characters omitted" in result.content
    assert len(result.content) < 3000


def test_cwd_outside_workspace_rejected(workspace, make_ctx):
    registry = ToolRegistry(default_tools())
    result = registry.dispatch("run_command", {"command": "ls", "cwd": ".."}, make_ctx(workspace))
    assert result.is_error and "outside" in result.content
