"""Built-in tools available to the agent."""

from __future__ import annotations

from ycode.tools.base import Tool, ToolContext, ToolRegistry, ToolResult
from ycode.tools.filesystem import EditFileTool, ListDirectoryTool, ReadFileTool, WriteFileTool
from ycode.tools.git import GitBranchTool, GitDiffTool, GitLogTool, GitStatusTool
from ycode.tools.search import FindFilesTool, SearchFilesTool
from ycode.tools.terminal import RunCommandTool

__all__ = ["Tool", "ToolContext", "ToolRegistry", "ToolResult", "default_tools"]


def default_tools() -> list[Tool]:
    return [
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        ListDirectoryTool(),
        SearchFilesTool(),
        FindFilesTool(),
        RunCommandTool(),
        GitStatusTool(),
        GitDiffTool(),
        GitLogTool(),
        GitBranchTool(),
    ]
