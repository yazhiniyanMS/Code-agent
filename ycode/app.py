"""Wires configuration, context, security, tools, provider, agent and UI together."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ycode import __version__
from ycode.agent.context import ProjectContext, detect_project
from ycode.agent.loop import Agent, TaskResult
from ycode.agent.prompts import build_system_prompt
from ycode.config import Config
from ycode.llm.base import LLMProvider
from ycode.security.permissions import PermissionManager
from ycode.tools import ToolContext, ToolRegistry, default_tools
from ycode.tools import git

MAX_HASHED_BYTES = 10_000_000  # larger changed files are compared by size and mtime
from ycode.ui.banner import BannerInfo
from ycode.ui.console import ConsoleUI


class YCodeApp:
    def __init__(
        self,
        *,
        config: Config,
        workspace: Path,
        provider: LLMProvider,
        ui: ConsoleUI,
        context: ProjectContext | None = None,
    ) -> None:
        self.config = config
        self.workspace = workspace.resolve()
        self.ui = ui
        self.context = context or detect_project(self.workspace, config.ignore_dirs)
        self.permissions = PermissionManager(
            self.workspace,
            mode=config.approval_mode,
            approver=ui.ask_approval,
            allow_outside_workspace=config.allow_outside_workspace,
        )
        self.tool_context = ToolContext(
            workspace=self.workspace,
            permissions=self.permissions,
            ignore_dirs=config.ignore_dirs,
            command_timeout=config.command_timeout,
            max_output_chars=config.max_output_chars,
        )
        self.registry = ToolRegistry(default_tools())
        self.agent = Agent(
            provider=provider,
            registry=self.registry,
            tool_context=self.tool_context,
            system_prompt=build_system_prompt(self.context),
            max_steps=config.max_steps,
            events=ui,
        )
        self.should_exit = False

    @property
    def provider(self) -> LLMProvider:
        return self.agent.provider

    def banner_info(self) -> BannerInfo:
        return BannerInfo(
            version=__version__,
            model=self.provider.display_name,
            directory=str(self.workspace),
            branch=self.context.branch,
            project_types=tuple(self.context.project_types),
            approval_mode=self.permissions.mode,
            instructions_loaded=bool(self.context.instructions),
        )

    def worktree_state(self) -> dict[str, tuple[str, str]] | None:
        """Every file git lists as changed or untracked, with its status and content hash (None without
        git). Taken before and after a task, it shows what the task changed, by any tool or command."""
        if not self.context.is_git_repo:
            return None
        state = {}
        for entry in git.status_entries(self.workspace):
            path = self.workspace / entry.path
            try:
                stat = path.stat()
                digest = (hashlib.sha256(path.read_bytes()).hexdigest() if stat.st_size <= MAX_HASHED_BYTES
                          else f"{stat.st_size}:{stat.st_mtime_ns}")
            except OSError:  # deleted, or a directory
                digest = "-"
            state[entry.path] = (entry.status, digest)
        return state

    def changes_for(self, files: list[str], before: dict | None = None) -> list[git.FileChange]:
        """Git status entries for the files the task changed: those written with the file tools, and, given
        the worktree_state from before the task, every file whose status or content changed during it
        (e.g. through run_command). Files the user had already changed, and the task did not touch, are
        not listed. Without git, the files written with the file tools."""
        if not self.context.is_git_repo:
            return [git.FileChange(" M", f) for f in files]
        after = self.worktree_state() or {}
        changed = set(files) | ({p for p, state in after.items() if before.get(p) != state} if before is not None
                                else set())
        return [e for e in git.status_entries(self.workspace) if e.path in changed]

    def run_task(self, task: str) -> TaskResult:
        before = self.worktree_state()
        result = self.agent.run(task)
        self.ui.print_task_summary(result, self.changes_for(result.changed_files, before))
        return result

    def handle_line(self, line: str) -> None:
        """Process one line of user input: a /command or a task for the agent."""
        from ycode.commands import handle_command, is_command

        text = line.strip()
        if not text:
            return
        if is_command(text):
            handle_command(self, text)
            return
        self.ui.console.print()
        self.run_task(text)
