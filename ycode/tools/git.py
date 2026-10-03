"""Git awareness: helpers used by the UI plus read-only tools for the model.

Nothing here commits, pushes or otherwise changes repository state; those
go through run_command and the approval system.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ycode.tools.base import Tool, ToolContext, ToolResult, truncate_middle
from ycode.tools.terminal import command_env


@dataclass(frozen=True)
class FileChange:
    status: str  # porcelain XY code, e.g. " M", "A ", "??"
    path: str

    @property
    def short(self) -> str:
        code = self.status.strip()
        if code == "??":
            return "A"  # untracked -> new
        return code[0] if code else "M"


def run_git(args: list[str], cwd: Path, timeout: int = 30) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=command_env(),
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return 127, "", "git is not installed"
    except subprocess.TimeoutExpired:
        return 124, "", "git timed out"
    return proc.returncode, proc.stdout, proc.stderr


def is_git_repo(cwd: Path) -> bool:
    code, out, _ = run_git(["rev-parse", "--is-inside-work-tree"], cwd)
    return code == 0 and out.strip() == "true"


def current_branch(cwd: Path) -> str | None:
    code, out, _ = run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    if code != 0:
        # Fresh repo without commits.
        code, out, _ = run_git(["symbolic-ref", "--short", "HEAD"], cwd)
        if code != 0:
            return None
    name = out.strip()
    if name == "HEAD":
        return "(detached HEAD)"
    return name or None


def status_entries(cwd: Path) -> list[FileChange]:
    code, out, _ = run_git(["status", "--porcelain=v1", "--untracked-files=all"], cwd)
    if code != 0:
        return []
    entries = []
    for line in out.splitlines():
        if len(line) < 4:
            continue
        status, path = line[:2], line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        entries.append(FileChange(status, path.strip('"')))
    return entries


def diff_text(cwd: Path, *, staged: bool = False, paths: list[str] | None = None, stat: bool = False) -> str:
    args = ["diff", "--no-color", "--no-ext-diff"]
    if staged:
        args.append("--cached")
    if stat:
        args.append("--stat")
    if paths:
        args += ["--", *paths]
    code, out, err = run_git(args, cwd)
    return out if code == 0 else err


def untracked_files(cwd: Path) -> list[str]:
    return [e.path for e in status_entries(cwd) if e.status == "??"]


class GitStatusTool(Tool):
    name = "git_status"
    description = "Show the current branch and changed/untracked files (git status)."
    parameters = {"type": "object", "properties": {}, "required": []}

    def describe(self, args: dict[str, Any]) -> str:
        return "Checking git status"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if not is_git_repo(ctx.workspace):
            return ToolResult(content="Not a git repository.", summary="Not a git repository")
        code, out, err = run_git(["status", "--short", "--branch", "--untracked-files=all"], ctx.workspace)
        if code != 0:
            return ToolResult.error(err.strip() or "git status failed")
        changes = len(out.strip().splitlines()) - 1
        return ToolResult(
            content=out.strip() or "Clean working tree.",
            summary=f"{max(changes, 0)} changed files",
        )


class GitDiffTool(Tool):
    name = "git_diff"
    description = (
        "Show the git diff of uncommitted changes. Use stat=true for a summary, paths to limit "
        "to files, staged=true for the index. Untracked (new) files are listed separately."
    )
    parameters = {
        "type": "object",
        "properties": {
            "paths": {"type": "array", "items": {"type": "string"}, "description": "Limit to these paths."},
            "staged": {"type": "boolean", "description": "Show staged changes instead."},
            "stat": {"type": "boolean", "description": "Only show a diffstat summary."},
        },
        "required": [],
    }

    def describe(self, args: dict[str, Any]) -> str:
        return "Reviewing git diff"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if not is_git_repo(ctx.workspace):
            return ToolResult(content="Not a git repository.", summary="Not a git repository")
        paths = [str(p) for p in args.get("paths") or []]
        for p in paths:
            ctx.permissions.paths.resolve(p)
        diff = diff_text(ctx.workspace, staged=bool(args.get("staged")), paths=paths or None,
                         stat=bool(args.get("stat")))
        untracked = untracked_files(ctx.workspace)
        body = truncate_middle(diff.strip(), ctx.max_output_chars) or "No differences in tracked files."
        if untracked and not args.get("staged"):
            body += "\n\nUntracked files:\n" + "\n".join(f"  {p}" for p in untracked[:200])
        return ToolResult(content=body, summary=f"Diff: {len(diff.splitlines())} lines")


class GitLogTool(Tool):
    name = "git_log"
    description = "Show recent commits (one line each)."
    parameters = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Number of commits (default 15)."},
            "path": {"type": "string", "description": "Only commits touching this path."},
        },
        "required": [],
    }

    def describe(self, args: dict[str, Any]) -> str:
        return "Reading git log"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        limit = min(max(int(args.get("limit") or 15), 1), 200)
        cmd = ["log", f"-{limit}", "--pretty=format:%h %ad %an: %s", "--date=short"]
        if args.get("path"):
            ctx.permissions.paths.resolve(args["path"])
            cmd += ["--", args["path"]]
        code, out, err = run_git(cmd, ctx.workspace)
        if code != 0:
            return ToolResult.error(err.strip() or "git log failed")
        return ToolResult(content=out.strip() or "No commits yet.", summary=f"{len(out.splitlines())} commits")


class GitBranchTool(Tool):
    name = "git_branch"
    description = "List local branches and show the current one."
    parameters = {"type": "object", "properties": {}, "required": []}

    def describe(self, args: dict[str, Any]) -> str:
        return "Listing git branches"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        code, out, err = run_git(["branch", "--list", "-vv"], ctx.workspace)
        if code != 0:
            return ToolResult.error(err.strip() or "git branch failed")
        return ToolResult(content=out.rstrip() or "No branches yet.", summary="Listed branches")
