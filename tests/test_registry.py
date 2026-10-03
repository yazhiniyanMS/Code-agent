from ycode.tools.base import Tool, ToolRegistry, ToolResult, truncate_middle, validate_args

SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}, "n": {"type": "integer"}, "mode": {"type": "string", "enum": ["a", "b"]}},
    "required": ["path"],
}


class Boom(Tool):
    name = "boom"
    description = "always fails"
    parameters = {"type": "object", "properties": {}, "required": []}

    def run(self, args, ctx):
        raise RuntimeError("kaboom")


class Echo(Tool):
    name = "echo"
    description = "echo"
    parameters = SCHEMA

    def run(self, args, ctx):
        return ToolResult(content=args["path"], summary="ok")


def test_validate_args():
    assert validate_args(SCHEMA, {"path": "x"}) is None
    assert "missing required" in validate_args(SCHEMA, {})
    assert "must be integer" in validate_args(SCHEMA, {"path": "x", "n": "3"})
    assert "must be integer" in validate_args(SCHEMA, {"path": "x", "n": True})
    assert "one of" in validate_args(SCHEMA, {"path": "x", "mode": "c"})
    assert "JSON object" in validate_args(SCHEMA, "not a dict")


def test_dispatch_routes_to_tool(workspace, make_ctx):
    registry = ToolRegistry([Echo()])
    result = registry.dispatch("echo", {"path": "hi"}, make_ctx(workspace))
    assert result.content == "hi" and not result.is_error
    assert result.duration >= 0


def test_dispatch_unknown_tool(workspace, make_ctx):
    result = ToolRegistry([Echo()]).dispatch("nope", {}, make_ctx(workspace))
    assert result.is_error and "unknown tool" in result.content and "echo" in result.content


def test_dispatch_malformed_arguments(workspace, make_ctx):
    registry = ToolRegistry([Echo()])
    assert registry.dispatch("echo", {"path": 5}, make_ctx(workspace)).is_error
    assert registry.dispatch("echo", None, make_ctx(workspace)).is_error


def test_dispatch_contains_unexpected_exceptions(workspace, make_ctx):
    result = ToolRegistry([Boom()]).dispatch("boom", {}, make_ctx(workspace))
    assert result.is_error and "RuntimeError: kaboom" in result.content


def test_duplicate_registration_rejected():
    import pytest

    with pytest.raises(ValueError):
        ToolRegistry([Echo(), Echo()])


def test_truncate_middle_keeps_head_and_tail():
    text = "HEAD" + "x" * 10000 + "TAIL"
    out = truncate_middle(text, 300)
    assert out.startswith("HEAD") and out.endswith("TAIL") and "omitted" in out
    assert truncate_middle("short", 300) == "short"
