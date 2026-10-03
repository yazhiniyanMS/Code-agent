"""Wires configuration, context, security, tools, provider, agent and UI together."""

from __future__ import annotations

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

    def changes_for(self, files: list[str]) -> list[git.FileChange]:
        """Git status entries for files the task touched (or a plain list without git)."""
        if not files:
            return []
        if not self.context.is_git_repo:
            return [git.FileChange(" M", f) for f in files]
        wanted = set(files)
        entries = [e for e in git.status_entries(self.workspace) if e.path in wanted]
        return entries

    def run_task(self, task: str) -> TaskResult:
        result = self.agent.run(task)
        self.ui.print_task_summary(result, self.changes_for(result.changed_files))
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
