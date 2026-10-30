import pytest

from ycode.tools import ToolRegistry, default_tools
from ycode.tools.filesystem import EditFileTool, ListDirectoryTool, ReadFileTool, WriteFileTool
from ycode.tools.search import FindFilesTool, SearchFilesTool


@pytest.fixture
def registry():
    return ToolRegistry(default_tools())


def test_read_file_numbers_lines(workspace, make_ctx, registry):
    (workspace / "a.txt").write_text("one\ntwo\nthree\n")
    result = registry.dispatch("read_file", {"path": "a.txt"}, make_ctx(workspace))
    assert not result.is_error
    assert "1\tone" in result.content and "3\tthree" in result.content


def test_read_file_window(workspace, make_ctx):
    (workspace / "big.txt").write_text("\n".join(f"line{i}" for i in range(1, 101)))
    result = ReadFileTool().run({"path": "big.txt", "offset": 10, "limit": 5}, make_ctx(workspace))
    assert "10\tline10" in result.content and "14\tline14" in result.content
    assert "line15" not in result.content
    assert "Showing lines 10-14 of 100" in result.content


def test_read_missing_file_is_reported(workspace, make_ctx, registry):
    result = registry.dispatch("read_file", {"path": "nope.py"}, make_ctx(workspace))
    assert result.is_error and "not found" in result.content.lower()


def test_read_binary_file_rejected(workspace, make_ctx, registry):
    (workspace / "img.bin").write_bytes(b"\x89PNG\x00\x00binary")
    result = registry.dispatch("read_file", {"path": "img.bin"}, make_ctx(workspace))
    assert result.is_error and "binary" in result.content


def test_read_sensitive_file_requires_approval(workspace, make_ctx, registry, approver):
    (workspace / ".env").write_text("SECRET=1\n")
    result = registry.dispatch("read_file", {"path": ".env"}, make_ctx(workspace))
    assert result.is_error and "Permission denied" in result.content
    assert approver.requests[0].kind == "read"
    approver.answer = "yes"
    result = registry.dispatch("read_file", {"path": ".env"}, make_ctx(workspace))
    assert "SECRET=1" in result.content


def test_write_file_creates_dirs_and_tracks_change(workspace, make_ctx):
    ctx = make_ctx(workspace)
    result = WriteFileTool().run({"path": "src/new/mod.py", "content": "x = 1\n"}, ctx)
    assert (workspace / "src/new/mod.py").read_text() == "x = 1\n"
    assert "Created" in result.summary
    assert "src/new/mod.py" in ctx.changed_files
    assert result.display["diff"].startswith("--- a/src/new/mod.py")


def test_write_into_git_dir_blocked(workspace, make_ctx, registry):
    result = registry.dispatch("write_file", {"path": ".git/config", "content": "x"}, make_ctx(workspace))
    assert result.is_error and ".git" in result.content


def test_edit_file_replaces_unique_string(workspace, make_ctx):
    path = workspace / "m.py"
    path.write_text("def f():\n    return 1\n")
    ctx = make_ctx(workspace)
    result = EditFileTool().run({"path": "m.py", "old_string": "return 1", "new_string": "return 2"}, ctx)
    assert path.read_text() == "def f():\n    return 2\n"
    assert "+1 -1" in result.summary
    assert "m.py" in ctx.changed_files


def test_edit_file_ambiguous_match(workspace, make_ctx, registry):
    (workspace / "m.py").write_text("a = 1\na = 1\n")
    ctx = make_ctx(workspace)
    result = registry.dispatch("edit_file", {"path": "m.py", "old_string": "a = 1", "new_string": "a = 2"}, ctx)
    assert result.is_error and "2 times" in result.content
    result = registry.dispatch("edit_file", {"path": "m.py", "old_string": "a = 1", "new_string": "a = 2",
                                             "replace_all": True}, ctx)
    assert not result.is_error
    assert (workspace / "m.py").read_text() == "a = 2\na = 2\n"


def test_edit_file_missing_string(workspace, make_ctx, registry):
    (workspace / "m.py").write_text("    value = 1\n")
    result = registry.dispatch("edit_file", {"path": "m.py", "old_string": "value = 2", "new_string": "x"},
                               make_ctx(workspace))
    assert result.is_error and "not found" in result.content


def test_edit_file_identical_strings(workspace, make_ctx, registry):
    (workspace / "m.py").write_text("a\n")
    result = registry.dispatch("edit_file", {"path": "m.py", "old_string": "a", "new_string": "a"},
                               make_ctx(workspace))
    assert result.is_error


def test_strict_mode_asks_before_writing(workspace, make_ctx, registry, approver):
    result = registry.dispatch("write_file", {"path": "x.txt", "content": "hi"}, make_ctx(workspace, mode="strict"))
    assert result.is_error and not (workspace / "x.txt").exists()
    assert approver.requests[0].kind == "write"


def test_list_directory_ignores_configured_dirs(workspace, make_ctx):
    (workspace / "src").mkdir()
    (workspace / "src" / "main.py").write_text("")
    (workspace / "node_modules" / "pkg").mkdir(parents=True)
    result = ListDirectoryTool().run({}, make_ctx(workspace))
    assert "src/" in result.content and "main.py" in result.content
    assert "node_modules" not in result.content


def test_search_files(workspace, make_ctx):
    (workspace / "src").mkdir()
    (workspace / "src" / "auth.ts").write_text("export function validateToken() {}\n")
    (workspace / "src" / "other.py").write_text("validateToken = None\n")
    (workspace / "node_modules").mkdir()
    (workspace / "node_modules" / "x.ts").write_text("validateToken\n")
    ctx = make_ctx(workspace)
    result = SearchFilesTool().run({"pattern": "validatetoken"}, ctx)
    assert "src/auth.ts:1:" in result.content and "src/other.py:1:" in result.content
    assert "node_modules" not in result.content
    result = SearchFilesTool().run({"pattern": "validateToken", "glob": "*.ts"}, ctx)
    assert "other.py" not in result.content


def test_search_does_not_bypass_the_approval_for_sensitive_files(workspace, make_ctx, registry, approver):
    # read_file asks before sending .env to the model; search_files used to print its lines without asking
    (workspace / ".env").write_text("API_TOKEN=sk-live-1234\n")
    (workspace / "id_rsa").write_text("-----BEGIN KEY-----\n")
    (workspace / "app.py").write_text("token = os.environ['API_TOKEN']\n")
    ctx = make_ctx(workspace)
    result = registry.dispatch("search_files", {"pattern": "API_TOKEN"}, ctx)
    assert "app.py:1:" in result.content and "sk-live" not in result.content
    assert "Matches in .env not shown" in result.content
    assert [(r.kind, r.subject) for r in approver.requests] == [("read", ".env")]  # id_rsa did not match: no prompt
    result = registry.dispatch("search_files", {"pattern": "sk-", "path": ".env"}, ctx)
    assert "sk-live" not in result.content and result.content.startswith("No matches shown.")
    approver.answer = "yes"
    result = registry.dispatch("search_files", {"pattern": "API_TOKEN"}, ctx)
    assert ".env:1: API_TOKEN=sk-live-1234" in result.content and "not shown" not in result.content
    approver.requests.clear()
    result = registry.dispatch("search_files", {"pattern": "API_TOKEN"}, make_ctx(workspace, mode="auto"))
    assert ".env:1:" in result.content and not approver.requests  # as read_file: auto mode does not ask


def test_search_invalid_regex(workspace, make_ctx, registry):
    result = registry.dispatch("search_files", {"pattern": "("}, make_ctx(workspace))
    assert result.is_error and "regular expression" in result.content


def test_find_files(workspace, make_ctx):
    (workspace / "a").mkdir()
    (workspace / "a" / "x.test.ts").write_text("")
    (workspace / "a" / "x.ts").write_text("")
    result = FindFilesTool().run({"pattern": "*.test.ts"}, make_ctx(workspace))
    assert result.content.strip() == "a/x.test.ts"
