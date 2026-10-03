"""Rich-based terminal renderer. Implements the agent's event callbacks."""

from __future__ import annotations

from typing import Iterable

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.status import Status
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from ycode.agent.loop import AgentEvents, TaskResult, TaskStatus, describe_status
from ycode.agent.planner import Plan
from ycode.errors import YCodeError
from ycode.llm.base import AssistantMessage, ToolCall
from ycode.security.permissions import ApprovalRequest
from ycode.tools.base import ToolResult
from ycode.tools.git import FileChange
from ycode.ui.banner import TAGLINE, BannerInfo, banner_lines, info_lines, supports_unicode

THEME = Theme({
    "accent": "bold cyan",
    "muted": "grey62",
    "ok": "green",
    "err": "bold red",
    "warn": "yellow",
    "step": "cyan",
    "banner": "bold cyan",
    "key": "bold",
})

MAX_INLINE_DIFF_LINES = 40


def make_console(theme: str = "default", **kwargs) -> Console:
    return Console(theme=THEME, no_color=(theme == "mono"), highlight=False, **kwargs)


class ConsoleUI(AgentEvents):
    def __init__(self, console: Console | None = None, *, show_diffs: bool = True) -> None:
        self.console = console or make_console()
        self.show_diffs = show_diffs
        self.unicode = supports_unicode(self.console.encoding)
        self._status: Status | None = None
        self._live: Live | None = None
        self._buffer = ""

    # ------------------------------------------------------------ symbols

    @property
    def sym(self) -> dict[str, str]:
        if self.unicode:
            return {"step": "◇", "ok": "✓", "err": "✗", "warn": "⚠", "doing": "▸", "todo": "○"}
        return {"step": "*", "ok": "+", "err": "x", "warn": "!", "doing": ">", "todo": "-"}

    # ------------------------------------------------------------ banner

    def print_banner(self, info: BannerInfo, *, show_hint: bool = True) -> None:
        art = Text("\n".join(banner_lines(unicode=self.unicode)), style="banner")
        width = max(len(line) for line in banner_lines(unicode=self.unicode))
        self.console.print()
        self.console.print(Padding(art, (0, 2), expand=False))
        self.console.print(Padding(Text(TAGLINE.center(width).rstrip(), style="muted"), (0, 2), expand=False))
        self.console.print()
        table = Table.grid(padding=(0, 2))
        table.add_column(style="muted")
        table.add_column()
        for key, value in info_lines(info):
            table.add_row(key, value)
        self.console.print(Padding(table, (0, 2), expand=False))
        self.console.print()
        if show_hint:
            self.console.print(Padding(
                Text("Type a task, or /help for commands. Esc+Enter inserts a newline.", style="muted"),
                (0, 2), expand=False,
            ))
            self.console.print()

    # ------------------------------------------------------- live helpers

    def _start_status(self, message: str) -> None:
        self._stop_live()
        if not self.console.is_terminal:
            return
        if self._status is None:
            self._status = self.console.status(Text(message, style="muted"), spinner="dots")
            self._status.start()
        else:
            self._status.update(Text(message, style="muted"))

    def _stop_status(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    def _stop_live(self) -> None:
        """Finish any streamed text and print it as rendered Markdown."""
        if self._live is not None:
            self._live.stop()
            self._live = None
        if self._buffer.strip():
            self.console.print(Padding(Markdown(self._buffer.strip()), (0, 2), expand=False))
            self.console.print()
        self._buffer = ""

    def stop(self) -> None:
        self._stop_status()
        self._stop_live()

    # ------------------------------------------------------- agent events

    def on_step_start(self, step: int, max_steps: int) -> None:
        self._start_status(f"Thinking... (step {step}/{max_steps})")

    def on_text(self, chunk: str) -> None:
        self._stop_status()
        self._buffer += chunk
        if not self.console.is_terminal:
            return
        preview = Padding(Markdown(self._buffer), (0, 2), expand=False)
        if self._live is None:
            self._live = Live(preview, console=self.console, transient=True,
                              vertical_overflow="ellipsis", refresh_per_second=12)
            self._live.start()
        else:
            self._live.update(preview)

    def on_tool_call_streaming(self, name: str) -> None:
        self._stop_live()
        self._start_status(f"Preparing {name}...")

    def on_response_done(self, message: AssistantMessage) -> None:
        self._stop_status()
        self._stop_live()

    def on_tool_start(self, call: ToolCall, description: str) -> None:
        self._stop_status()
        if call.name == "update_plan":
            return
        self.console.print(Text.assemble(("  " + self.sym["step"] + " ", "step"), (description, "")))
        self._start_status(description + "...")

    def on_tool_end(self, call: ToolCall, result: ToolResult) -> None:
        self._stop_status()
        if call.name == "update_plan" and not result.is_error:
            return
        display = result.display or {}
        failed = result.is_error or display.get("failed")
        mark, style = (self.sym["err"], "err") if failed else (self.sym["ok"], "ok")
        self.console.print(Text.assemble(("  " + mark + " ", style), (result.summary or "Done", "muted")))
        if display.get("output_tail"):
            self.console.print(Padding(Text(display["output_tail"], style="muted"), (0, 6), expand=False))
        if self.show_diffs and display.get("diff"):
            self.print_diff(display["diff"], limit=MAX_INLINE_DIFF_LINES, indent=6)
        self.console.print()

    def on_plan(self, plan: Plan) -> None:
        self._stop_status()
        self.print_plan(plan)

    def on_notice(self, message: str, level: str = "info") -> None:
        self._stop_status()
        style = {"warning": "warn", "error": "err"}.get(level, "muted")
        mark = self.sym["warn"] if level in ("warning", "error") else self.sym["step"]
        self.console.print(Text(f"  {mark} {message}", style=style))

    # --------------------------------------------------------- rendering

    def print_plan(self, plan: Plan) -> None:
        if plan.is_empty:
            self.console.print(Text("  No plan for the current task.", style="muted"))
            return
        self.console.print(Text.assemble(("  " + self.sym["step"] + " ", "step"), ("Plan", "key")))
        marks = {"done": (self.sym["ok"], "ok"), "in_progress": (self.sym["doing"], "accent"),
                 "pending": (self.sym["todo"], "muted"), "skipped": ("-", "muted")}
        for i, step in enumerate(plan.steps, 1):
            mark, style = marks[step.status]
            text_style = "muted" if step.status in ("done", "skipped") else ""
            self.console.print(Text.assemble((f"    {mark} ", style), (f"{i}. {step.text}", text_style)))
        self.console.print()

    def print_diff(self, diff: str, *, limit: int | None = None, indent: int = 2) -> None:
        lines = diff.rstrip("\n").splitlines()
        hidden = 0
        if limit is not None and len(lines) > limit:
            hidden = len(lines) - limit
            lines = lines[:limit]
        syntax = Syntax("\n".join(lines), "diff", theme="ansi_dark", background_color="default",
                        word_wrap=False)
        self.console.print(Padding(syntax, (0, indent), expand=False))
        if hidden:
            self.console.print(Padding(Text(f"... {hidden} more lines (use /diff)", style="muted"), (0, indent), expand=False))

    def print_changes(self, changes: Iterable[FileChange]) -> None:
        for change in changes:
            style = {"A": "ok", "D": "err"}.get(change.short, "warn")
            self.console.print(Text.assemble((f"   {change.short} ", style), change.path))

    def print_task_summary(self, result: TaskResult, changes: list[FileChange]) -> None:
        self.stop()
        if result.error is not None:
            self.print_error(result.error)
        if changes:
            self.console.print(Text("  Changes:", style="key"))
            self.print_changes(changes)
            self.console.print()
        if result.commands_run:
            self.console.print(Text("  Commands:", style="key"))
            for command, code in result.commands_run[-8:]:
                ok = code == 0
                mark, style = (self.sym["ok"], "ok") if ok else (self.sym["err"], "err")
                suffix = "" if ok else (" (timed out)" if code is None else f" (exit {code})")
                self.console.print(Text.assemble((f"   {mark} ", style), command, (suffix, "muted")))
            self.console.print()
        style = {TaskStatus.COMPLETED: "ok", TaskStatus.INTERRUPTED: "warn",
                 TaskStatus.STEP_LIMIT: "warn"}.get(result.status, "err")
        mark = self.sym["ok"] if result.status is TaskStatus.COMPLETED else self.sym["warn"]
        self.console.print(Text.assemble(
            (f"  {mark} {describe_status(result)}", style),
            (f"  ({result.steps} step{'s' if result.steps != 1 else ''})", "muted"),
        ))
        self.console.print()

    def print_error(self, error: BaseException) -> None:
        self.stop()
        message = str(error) or type(error).__name__
        hint = getattr(error, "hint", None) if isinstance(error, YCodeError) else None
        parts = [Text(f"  {self.sym['err']} {message}", style="err")]
        if hint:
            parts.append(Padding(Text(hint, style="muted"), (0, 4), expand=False))
        self.console.print(Group(*parts))

    def info(self, message: str) -> None:
        self.console.print(Text(f"  {message}", style="muted"))

    def print_table(self, rows: list[tuple[str, str]]) -> None:
        table = Table.grid(padding=(0, 2))
        table.add_column(style="muted")
        table.add_column()
        for key, value in rows:
            table.add_row(key, value)
        self.console.print(Padding(table, (0, 2), expand=False))

    # ---------------------------------------------------------- approval

    def ask_approval(self, request: ApprovalRequest) -> str:
        """Interactive y/N/a prompt. Returns 'yes', 'no' or 'always'."""
        self.stop()
        title = {"command": "run", "write": "write to", "read": "read"}.get(request.kind, request.kind)
        self.console.print()
        self.console.print(Text(f"  {self.sym['warn']} YCode wants to {title}:", style="warn"))
        self.console.print()
        self.console.print(Padding(Text(request.subject, style="bold"), (0, 6), expand=False))
        self.console.print()
        if request.reason:
            self.console.print(Text(f"    Reason: {request.reason}", style="muted"))
        try:
            answer = self.console.input("  Allow? [y/N/a=always this session] ").strip().lower()
        except EOFError:
            answer = ""
        self.console.print()
        if answer in ("y", "yes"):
            return "yes"
        if answer in ("a", "always"):
            return "always"
        self.console.print(Text("  Denied.", style="muted"))
        return "no"
