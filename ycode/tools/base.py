"""Tool abstraction, argument validation and dispatch."""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ycode.errors import PathSecurityError, PermissionDenied, ToolError
from ycode.security.permissions import PermissionManager


@dataclass
class ToolResult:
    """Outcome of a tool call.

    ``content`` goes back to the model; ``summary`` is the one-line status shown
    to the user; ``display`` holds optional extra UI data (e.g. a diff).
    """

    content: str
    is_error: bool = False
    summary: str = ""
    display: dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0

    @classmethod
    def error(cls, message: str, summary: str | None = None) -> "ToolResult":
        return cls(content=f"Error: {message}", is_error=True, summary=summary or message)


@dataclass
class ToolContext:
    workspace: Path
    permissions: PermissionManager
    ignore_dirs: tuple[str, ...]
    command_timeout: int = 120
    max_output_chars: int = 30000
    # Files written during the current session (relative paths).
    changed_files: set[str] = field(default_factory=set)
    # Commands run during the current task: (command, exit_code).
    commands_run: list[tuple[str, int | None]] = field(default_factory=list)


class Tool(abc.ABC):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema ("type": "object")

    @abc.abstractmethod
    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult: ...

    def describe(self, args: dict[str, Any]) -> str:
        """Short human-readable description of a call, e.g. 'Reading src/a.py'."""
        return self.name

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


# ------------------------------------------------------------- validation

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def validate_args(schema: dict[str, Any], args: Any) -> str | None:
    """Minimal JSON-schema check for tool inputs. Returns an error or None."""
    if not isinstance(args, dict):
        return f"arguments must be a JSON object, got {type(args).__name__}"
    props: dict[str, Any] = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in args:
            return f"missing required argument '{name}'"
    for name, value in args.items():
        prop = props.get(name)
        if prop is None:
            if schema.get("additionalProperties") is False:
                return f"unexpected argument '{name}'"
            continue
        expected = prop.get("type")
        if expected and value is not None:
            types = _JSON_TYPES.get(expected, (object,))
            if isinstance(value, bool) and expected in ("integer", "number"):
                return f"argument '{name}' must be {expected}"
            if not isinstance(value, types):
                return f"argument '{name}' must be {expected}, got {type(value).__name__}"
        if "enum" in prop and value not in prop["enum"]:
            return f"argument '{name}' must be one of {prop['enum']}"
    return None


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self._tools.values()]

    def describe(self, name: str, args: Any) -> str:
        tool = self._tools.get(name)
        if tool is None or not isinstance(args, dict):
            return name
        try:
            return tool.describe(args)
        except Exception:  # noqa: BLE001 - description is cosmetic
            return name

    def dispatch(self, name: str, args: Any, ctx: ToolContext) -> ToolResult:
        """Validate and execute a tool call. Never raises for tool-level errors."""
        start = time.monotonic()
        tool = self._tools.get(name)
        if tool is None:
            result = ToolResult.error(
                f"unknown tool '{name}'. Available tools: {', '.join(self._tools)}",
                summary=f"Unknown tool {name}",
            )
        else:
            problem = validate_args(tool.parameters, args)
            if problem:
                result = ToolResult.error(f"invalid arguments for {name}: {problem}", summary="Invalid tool call")
            else:
                try:
                    result = tool.run(args, ctx)
                except PermissionDenied as exc:
                    result = ToolResult(
                        content=f"Permission denied: {exc}. Do not retry the same action; "
                        "choose a different approach or ask the user.",
                        is_error=True,
                        summary=str(exc),
                    )
                except PathSecurityError as exc:
                    result = ToolResult.error(str(exc), summary="Blocked path")
                except ToolError as exc:
                    result = ToolResult.error(str(exc))
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001 - report to the model, don't crash
                    result = ToolResult.error(f"{type(exc).__name__}: {exc}", summary=f"{name} failed")
        result.duration = time.monotonic() - start
        return result


def truncate_middle(text: str, limit: int) -> str:
    """Keep the head and tail of long output; errors are usually at the end."""
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n\n... [{omitted} characters omitted] ...\n\n{text[-tail:]}"
