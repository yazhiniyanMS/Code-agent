"""run_command: execute shell commands inside the project directory."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ycode.errors import PermissionDenied, ToolError
from ycode.tools.base import Tool, ToolContext, ToolResult, truncate_middle

MAX_TIMEOUT = 1800

# Never hand credentials to child processes: the model sees their output.
_SECRET_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "YCODE_API_KEY")


@dataclass
class CommandOutput:
    command: str
    exit_code: int | None
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False


def command_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _SECRET_ENV_VARS}
    env.update({
        "CI": env.get("CI", "true"),  # keeps test runners out of watch/interactive mode
        "GIT_PAGER": "cat",
        "PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "NO_COLOR": "1",
        "FORCE_COLOR": "0",
        "PYTHONUNBUFFERED": "1",
    })
    return env


def _kill(proc: subprocess.Popen) -> None:
    try:
        if sys.platform != "win32":
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # pragma: no cover
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass


def execute(command: str, cwd: Path, timeout: int) -> CommandOutput:
    """Run ``command`` through the shell. Ctrl+C kills the child and re-raises."""
    start = time.monotonic()
    popen_kwargs: dict[str, Any] = {}
    if sys.platform != "win32":
        # Own process group: Ctrl+C reaches YCode, which then kills the whole group.
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=command_env(),
        **popen_kwargs,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        _kill(proc)
        out, err = proc.communicate()
        timed_out = True
    except KeyboardInterrupt:
        _kill(proc)
        proc.communicate()
        raise
    return CommandOutput(
        command=command,
        exit_code=None if timed_out else proc.returncode,
        stdout=out.decode("utf-8", errors="replace"),
        stderr=err.decode("utf-8", errors="replace"),
        duration=time.monotonic() - start,
        timed_out=timed_out,
    )


class RunCommandTool(Tool):
    name = "run_command"
    description = (
        "Run a shell command in the project directory and return stdout, stderr, exit code "
        "and duration. Use for builds, tests, linters, package managers and git inspection. "
        "Commands are non-interactive (stdin is closed); pass flags like --yes / --no-watch. "
        "Destructive commands require user approval; some are blocked."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run."},
            "cwd": {"type": "string", "description": "Subdirectory to run in (default project root)."},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default from config)."},
        },
        "required": ["command"],
    }

    def describe(self, args: dict[str, Any]) -> str:
        return f"Running `{args.get('command', '')}`"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command: str = args["command"].strip()
        if not command:
            raise ToolError("command must not be empty")
        cwd = ctx.permissions.paths.resolve(args.get("cwd") or ".")
        if not cwd.is_dir():
            raise ToolError(f"cwd {args.get('cwd')} is not a directory")
        allowed, decision = ctx.permissions.check_command(command)
        if not allowed:
            if decision.level.name == "BLOCKED":
                raise PermissionDenied(f"command blocked by YCode's safety policy ({decision.reason})")
            raise PermissionDenied(f"user declined to run `{command}`")
        timeout = min(max(int(args.get("timeout") or ctx.command_timeout), 1), MAX_TIMEOUT)

        result = execute(command, cwd, timeout)
        ctx.commands_run.append((command, result.exit_code))

        half = ctx.max_output_chars // 2
        stdout = truncate_middle(result.stdout, half if result.stderr else ctx.max_output_chars)
        stderr = truncate_middle(result.stderr, half)
        status = "timed out" if result.timed_out else f"exit code {result.exit_code}"
        parts = [f"$ {command}", f"[{status}, {result.duration:.1f}s]"]
        if result.timed_out:
            parts.append(f"The command was killed after {timeout}s.")
        parts.append(f"--- stdout ---\n{stdout.rstrip()}" if stdout.strip() else "--- stdout ---\n(empty)")
        if stderr.strip():
            parts.append(f"--- stderr ---\n{stderr.rstrip()}")
        failed = result.timed_out or result.exit_code != 0
        tail = (result.stderr or result.stdout).strip().splitlines()
        return ToolResult(
            content="\n".join(parts),
            is_error=False,  # a failing command is information, not a tool malfunction
            summary=f"{'Failed' if failed else 'Succeeded'} ({status}, {result.duration:.1f}s)",
            display={
                "command": command,
                "exit_code": result.exit_code,
                "failed": failed,
                "output_tail": "\n".join(tail[-8:]) if failed else "",
            },
        )
