"""File tools: read_file, write_file, edit_file, list_directory."""

from __future__ import annotations

import difflib
import os
from pathlib import Path
from typing import Any

from ycode.errors import PermissionDenied, ToolError
from ycode.tools.base import Tool, ToolContext, ToolResult

MAX_READ_BYTES = 2_000_000
DEFAULT_READ_LINES = 2000
MAX_LINE_CHARS = 2000


def _is_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def _read_text(path: Path) -> str:
    if not path.exists():
        raise ToolError(f"File not found: {path}")
    if path.is_dir():
        raise ToolError(f"{path} is a directory; use list_directory instead")
    size = path.stat().st_size
    if size > MAX_READ_BYTES:
        raise ToolError(
            f"{path} is {size:,} bytes, too large to read whole. Use search_files to find "
            "the relevant region, then read_file with offset/limit."
        )
    data = path.read_bytes()
    if _is_binary(data):
        raise ToolError(f"{path} looks like a binary file")
    return data.decode("utf-8", errors="replace")


def make_diff(before: str, after: str, rel: str, context: int = 3) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            n=context,
        )
    )


def _diff_stats(diff: str) -> tuple[int, int]:
    added = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))
    return added, removed


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a text file from the project. Returns numbered lines. For large files, "
        "pass offset (1-based line number) and limit to read a window."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the project root."},
            "offset": {"type": "integer", "description": "1-based line to start from (default 1)."},
            "limit": {"type": "integer", "description": f"Max lines to return (default {DEFAULT_READ_LINES})."},
        },
        "required": ["path"],
    }

    def describe(self, args: dict[str, Any]) -> str:
        return f"Reading {args.get('path', '?')}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.permissions.paths.resolve(args["path"])
        if not ctx.permissions.check_read(path):
            raise PermissionDenied(f"user declined reading {args['path']}")
        text = _read_text(path)
        lines = text.splitlines()
        offset = max(1, int(args.get("offset") or 1))
        limit = max(1, int(args.get("limit") or DEFAULT_READ_LINES))
        window = lines[offset - 1 : offset - 1 + limit]
        width = len(str(offset + len(window)))
        body = "\n".join(
            f"{n:>{width}}\t{line[:MAX_LINE_CHARS]}{' ...[truncated]' if len(line) > MAX_LINE_CHARS else ''}"
            for n, line in enumerate(window, start=offset)
        )
        rel = ctx.permissions.paths.relative(path)
        end = offset + len(window) - 1
        note = ""
        if end < len(lines):
            note = f"\n\n[Showing lines {offset}-{end} of {len(lines)}. Use offset to read more.]"
        if not lines:
            body = "[empty file]"
        return ToolResult(
            content=body + note,
            summary=f"Read {rel} ({len(window)} of {len(lines)} lines)",
        )


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Create a new file or completely overwrite an existing one. Prefer edit_file for "
        "changing part of an existing file. Parent directories are created as needed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the project root."},
            "content": {"type": "string", "description": "Full file content."},
        },
        "required": ["path", "content"],
    }

    def describe(self, args: dict[str, Any]) -> str:
        return f"Writing {args.get('path', '?')}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.permissions.paths.resolve(args["path"], for_write=True)
        rel = ctx.permissions.paths.relative(path)
        if path.is_dir():
            raise ToolError(f"{rel} is a directory")
        existed = path.exists()
        before = _read_text(path) if existed else ""
        if not ctx.permissions.check_write(path):
            raise PermissionDenied(f"user declined writing {rel}")
        content: str = args["content"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        ctx.changed_files.add(rel)
        diff = make_diff(before, content, rel)
        added, removed = _diff_stats(diff)
        verb = "Updated" if existed else "Created"
        return ToolResult(
            content=f"{verb} {rel} ({len(content.splitlines())} lines).",
            summary=f"{verb} {rel} (+{added} -{removed})",
            display={"diff": diff, "path": rel, "new_file": not existed},
        )


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Replace an exact string in a file. old_string must match the file exactly "
        "(including whitespace/indentation) and be unique unless replace_all is true. "
        "Include enough surrounding context to make it unique. Read the file first."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the project root."},
            "old_string": {"type": "string", "description": "Exact text to replace."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)."},
        },
        "required": ["path", "old_string", "new_string"],
    }

    def describe(self, args: dict[str, Any]) -> str:
        return f"Editing {args.get('path', '?')}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.permissions.paths.resolve(args["path"], for_write=True)
        rel = ctx.permissions.paths.relative(path)
        old, new = args["old_string"], args["new_string"]
        if old == new:
            raise ToolError("old_string and new_string are identical; nothing to change")
        if old == "":
            raise ToolError("old_string must not be empty; use write_file to create files")
        before = _read_text(path)
        count = before.count(old)
        if count == 0:
            hint = ""
            stripped = old.strip()
            if stripped and stripped in before:
                hint = " (a whitespace-trimmed version exists - check indentation and line endings)"
            raise ToolError(f"old_string not found in {rel}{hint}. Re-read the file and try again.")
        if count > 1 and not args.get("replace_all"):
            raise ToolError(
                f"old_string occurs {count} times in {rel}. Add more surrounding context "
                "to make it unique, or set replace_all to true."
            )
        if not ctx.permissions.check_write(path):
            raise PermissionDenied(f"user declined editing {rel}")
        after = before.replace(old, new) if args.get("replace_all") else before.replace(old, new, 1)
        path.write_text(after, encoding="utf-8")
        ctx.changed_files.add(rel)
        diff = make_diff(before, after, rel)
        added, removed = _diff_stats(diff)
        replaced = count if args.get("replace_all") else 1
        return ToolResult(
            content=f"Edited {rel}: replaced {replaced} occurrence(s).",
            summary=f"Edited {rel} (+{added} -{removed})",
            display={"diff": diff, "path": rel},
        )


class ListDirectoryTool(Tool):
    name = "list_directory"
    description = (
        "List files and folders as a tree (ignoring node_modules, .git, build outputs, ...). "
        "Use depth to control recursion (default 2)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory relative to the project root (default '.')."},
            "depth": {"type": "integer", "description": "Recursion depth, 1-6 (default 2)."},
        },
        "required": [],
    }
    MAX_ENTRIES = 500

    def describe(self, args: dict[str, Any]) -> str:
        return f"Listing {args.get('path') or '.'}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        root = ctx.permissions.paths.resolve(args.get("path") or ".")
        if not root.exists():
            raise ToolError(f"Directory not found: {args.get('path')}")
        if not root.is_dir():
            raise ToolError(f"{args.get('path')} is a file; use read_file")
        depth = min(max(int(args.get("depth") or 2), 1), 6)
        ignore = set(ctx.ignore_dirs)
        lines: list[str] = []
        count = 0
        truncated = False

        def walk(directory: Path, level: int, prefix: str) -> None:
            nonlocal count, truncated
            try:
                entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except OSError as exc:
                lines.append(f"{prefix}[unreadable: {exc.strerror}]")
                return
            for entry in entries:
                if entry.name in ignore:
                    continue
                if count >= self.MAX_ENTRIES:
                    truncated = True
                    return
                count += 1
                if entry.is_dir() and not entry.is_symlink():
                    lines.append(f"{prefix}{entry.name}/")
                    if level < depth:
                        walk(entry, level + 1, prefix + "  ")
                else:
                    lines.append(f"{prefix}{entry.name}")

        walk(root, 1, "")
        rel = ctx.permissions.paths.relative(root)
        body = "\n".join(lines) if lines else "[empty directory]"
        if truncated:
            body += f"\n... [truncated at {self.MAX_ENTRIES} entries; list a subdirectory]"
        return ToolResult(content=f"{rel}/\n{body}", summary=f"Listed {rel} ({count} entries)")


def iter_files(root: Path, ignore_dirs: tuple[str, ...]):
    """Walk files under ``root`` skipping ignored directories."""
    ignore = set(ignore_dirs)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in ignore)
        for filename in sorted(filenames):
            yield Path(dirpath) / filename
