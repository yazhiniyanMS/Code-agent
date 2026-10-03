"""Built-in slash commands, handled locally (never sent to the model)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from rich.text import Text

from ycode.config import APPROVAL_MODES
from ycode.tools import git

if TYPE_CHECKING:
    from ycode.app import YCodeApp

_COMMAND_RE = re.compile(r"^/([a-zA-Z][\w-]*)(?:\s+(.*))?$", re.DOTALL)


@dataclass(frozen=True)
class Command:
    name: str
    usage: str
    help: str
    handler: Callable[["YCodeApp", str], None]


def _help(app: "YCodeApp", arg: str) -> None:
    app.ui.console.print(Text("  Commands", style="key"))
    app.ui.print_table([(c.usage, c.help) for c in COMMANDS.values() if c.name != "quit"])
    app.ui.console.print()
    app.ui.info("Anything else is sent to the agent as a task. Ctrl+C interrupts the agent;")
    app.ui.info("Ctrl+D or /exit quits. Esc+Enter inserts a newline.")


def _clear(app: "YCodeApp", arg: str) -> None:
    app.agent.reset()
    app.ui.console.clear()
    app.ui.info("Conversation cleared. The agent has forgotten previous messages.")


def _status(app: "YCodeApp", arg: str) -> None:
    ws = app.workspace
    rows = [("Project", ws.name or str(ws)), ("Directory", str(ws))]
    if app.context.is_git_repo:
        rows.append(("Branch", git.current_branch(ws) or "(unknown)"))
        rows.append(("Modified files", str(len(git.status_entries(ws)))))
    else:
        rows.append(("Git", "not a repository"))
    usage = app.agent.usage
    rows += [
        ("Model", app.provider.display_name),
        ("Steps used", f"{app.agent.last_steps}/{app.agent.max_steps} (last task), {app.agent.total_steps} total"),
        ("Tokens", f"{usage.input_tokens:,} in, {usage.output_tokens:,} out, "
                   f"{usage.cache_read_tokens:,} cache reads"),
        ("Approvals", app.permissions.mode),
        ("Messages", str(len(app.agent.messages))),
        ("Files changed", str(len(app.tool_context.changed_files)) + " this session"),
    ]
    if app.context.instructions_path:
        rows.append(("Instructions", app.context.instructions_path))
    app.ui.print_table(rows)


def _diff(app: "YCodeApp", arg: str) -> None:
    if not app.context.is_git_repo:
        app.ui.info("Not a git repository.")
        return
    args = arg.split()
    full = "full" in args
    paths = [a for a in args if a != "full"]
    diff = git.diff_text(app.workspace, paths=paths or None)
    untracked = [p for p in git.untracked_files(app.workspace) if not paths or p in paths]
    if not diff.strip() and not untracked:
        app.ui.info("No changes.")
        return
    lines = diff.splitlines()
    if diff.strip():
        if len(lines) > 400 and not full:
            app.ui.console.print(git.diff_text(app.workspace, paths=paths or None, stat=True).rstrip())
            app.ui.info(f"Diff is {len(lines)} lines. Use `/diff full` or `/diff <path>` to see it.")
        else:
            app.ui.print_diff(diff)
    if untracked:
        app.ui.console.print(Text("  Untracked files:", style="key"))
        for path in untracked:
            app.ui.console.print(Text(f"   A {path}", style="ok"))


def _files(app: "YCodeApp", arg: str) -> None:
    session = sorted(app.tool_context.changed_files)
    if app.context.is_git_repo:
        entries = git.status_entries(app.workspace)
        if entries:
            app.ui.console.print(Text("  Working tree changes:", style="key"))
            app.ui.print_changes(entries)
        else:
            app.ui.info("Working tree clean.")
    if session:
        app.ui.console.print(Text("  Written by YCode this session:", style="key"))
        for path in session:
            app.ui.console.print(f"   {path}")
    elif not app.context.is_git_repo:
        app.ui.info("No files changed this session.")


def _model(app: "YCodeApp", arg: str) -> None:
    name = arg.strip()
    if not name:
        app.ui.info(f"Model: {app.provider.display_name}")
        app.ui.info("Switch with /model <model-id>, e.g. /model claude-sonnet-5-5")
        return
    app.provider.model = name
    app.ui.info(f"Model set to {name} for this session.")


def _plan(app: "YCodeApp", arg: str) -> None:
    app.ui.print_plan(app.agent.plan)


def _approval(app: "YCodeApp", arg: str) -> None:
    mode = arg.strip().lower()
    if not mode:
        app.ui.info(f"Approval mode: {app.permissions.mode} (options: {', '.join(APPROVAL_MODES)})")
        return
    if mode not in APPROVAL_MODES:
        app.ui.info(f"Unknown mode {mode!r}. Options: {', '.join(APPROVAL_MODES)}")
        return
    app.permissions.mode = mode
    note = " Blocked commands stay blocked." if mode == "auto" else ""
    app.ui.info(f"Approval mode set to {mode}.{note}")


def _exit(app: "YCodeApp", arg: str) -> None:
    app.should_exit = True


COMMANDS: dict[str, Command] = {
    c.name: c
    for c in [
        Command("help", "/help", "Show this help", _help),
        Command("clear", "/clear", "Clear the conversation and the screen", _clear),
        Command("status", "/status", "Project, branch, model, steps and token usage", _status),
        Command("diff", "/diff [full] [path...]", "Show the current git diff", _diff),
        Command("files", "/files", "List changed files", _files),
        Command("model", "/model [id]", "Show or switch the model", _model),
        Command("plan", "/plan", "Show the current plan", _plan),
        Command("approval", "/approval [mode]", "Show or set approval mode (strict|normal|auto)", _approval),
        Command("exit", "/exit", "Quit YCode (also /quit, Ctrl+D)", _exit),
        Command("quit", "/quit", "Quit YCode", _exit),
    ]
}


def command_names() -> list[str]:
    return [f"/{name}" for name in COMMANDS]


def is_command(text: str) -> bool:
    match = _COMMAND_RE.match(text.strip())
    return bool(match)


def handle_command(app: "YCodeApp", text: str) -> bool:
    """Run a slash command. Returns False if it was not recognised."""
    match = _COMMAND_RE.match(text.strip())
    if not match:
        return False
    name, arg = match.group(1).lower(), match.group(2) or ""
    command = COMMANDS.get(name)
    if command is None:
        app.ui.info(f"Unknown command /{name}. Type /help for the list of commands.")
        return False
    command.handler(app, arg)
    return True
