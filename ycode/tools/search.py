"""Search tools: search_files (content search) and find_files (glob)."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

from ycode.errors import ToolError
from ycode.tools.base import Tool, ToolContext, ToolResult
from ycode.tools.filesystem import iter_files

MAX_FILE_BYTES = 1_000_000


def _matches_glob(rel: str, pattern: str | None) -> bool:
    if not pattern:
        return True
    name = rel.rsplit("/", 1)[-1]
    patterns = [p.strip() for p in pattern.split(",") if p.strip()]
    return any(
        fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(name, p) or fnmatch.fnmatch(rel, f"**/{p}") for p in patterns
    )


class SearchFilesTool(Tool):
    name = "search_files"
    description = (
        "Search file contents with a regular expression (Python syntax). Returns matching "
        "lines as path:line: text. Use `glob` (e.g. '*.py' or 'src/**/*.ts') to narrow files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression to search for."},
            "path": {"type": "string", "description": "Directory to search (default project root)."},
            "glob": {"type": "string", "description": "Filename filter, comma-separated globs."},
            "case_sensitive": {"type": "boolean", "description": "Default false."},
            "max_results": {"type": "integer", "description": "Default 100."},
        },
        "required": ["pattern"],
    }

    def describe(self, args: dict[str, Any]) -> str:
        where = f" in {args['glob']}" if args.get("glob") else ""
        return f"Searching for /{args.get('pattern', '')}/{where}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        root = ctx.permissions.paths.resolve(args.get("path") or ".")
        if not root.exists():
            raise ToolError(f"Path not found: {args.get('path')}")
        flags = 0 if args.get("case_sensitive") else re.IGNORECASE
        try:
            regex = re.compile(args["pattern"], flags)
        except re.error as exc:
            raise ToolError(f"invalid regular expression: {exc}") from None
        limit = min(max(int(args.get("max_results") or 100), 1), 1000)
        files = [root] if root.is_file() else iter_files(root, ctx.ignore_dirs)
        results: list[str] = []
        files_matched = 0
        for file in files:
            rel = ctx.permissions.paths.relative(file)
            if not _matches_glob(rel, args.get("glob")):
                continue
            try:
                if file.stat().st_size > MAX_FILE_BYTES:
                    continue
                data = file.read_bytes()
            except OSError:
                continue
            if b"\x00" in data[:8192]:
                continue
            hit = False
            for lineno, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                if regex.search(line):
                    hit = True
                    results.append(f"{rel}:{lineno}: {line.strip()[:300]}")
                    if len(results) >= limit:
                        break
            files_matched += hit
            if len(results) >= limit:
                break
        if not results:
            return ToolResult(content="No matches found.", summary="No matches")
        suffix = f"\n[Stopped after {limit} results; refine the pattern.]" if len(results) >= limit else ""
        return ToolResult(
            content="\n".join(results) + suffix,
            summary=f"{len(results)} matches in {files_matched} files",
        )


class FindFilesTool(Tool):
    name = "find_files"
    description = "Find files by name/glob pattern (e.g. '*.test.ts', 'package.json', 'src/**/auth*')."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern (comma-separated for several)."},
            "path": {"type": "string", "description": "Directory to search (default project root)."},
            "max_results": {"type": "integer", "description": "Default 200."},
        },
        "required": ["pattern"],
    }

    def describe(self, args: dict[str, Any]) -> str:
        return f"Finding files matching {args.get('pattern', '')}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        root = ctx.permissions.paths.resolve(args.get("path") or ".")
        if not root.is_dir():
            raise ToolError(f"Directory not found: {args.get('path')}")
        limit = min(max(int(args.get("max_results") or 200), 1), 2000)
        found: list[str] = []
        for file in iter_files(root, ctx.ignore_dirs):
            rel = ctx.permissions.paths.relative(file)
            if _matches_glob(rel, args["pattern"]):
                found.append(rel)
                if len(found) >= limit:
                    break
        if not found:
            return ToolResult(content="No files found.", summary="No files found")
        return ToolResult(content="\n".join(found), summary=f"Found {len(found)} files")


def list_project_files(root: Path, ignore_dirs: tuple[str, ...], limit: int = 2000) -> list[str]:
    out: list[str] = []
    for file in iter_files(root, ignore_dirs):
        out.append(file.relative_to(root).as_posix())
        if len(out) >= limit:
            break
    return out
