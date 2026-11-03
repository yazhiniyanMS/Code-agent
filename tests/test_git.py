from tests.conftest import git
from ycode.tools import ToolRegistry, default_tools
from ycode.tools.git import current_branch, diff_text, is_git_repo, status_entries


def test_not_a_repo(workspace, make_ctx):
    assert not is_git_repo(workspace)
    result = ToolRegistry(default_tools()).dispatch("git_status", {}, make_ctx(workspace))
    assert "Not a git repository" in result.content


def test_status_and_branch(git_workspace, make_ctx):
    assert is_git_repo(git_workspace)
    assert current_branch(git_workspace) == "main"
    assert status_entries(git_workspace) == []
    (git_workspace / "app.py").write_text("print('changed')\n")
    (git_workspace / "new.py").write_text("x = 1\n")
    entries = {e.path: e.short for e in status_entries(git_workspace)}
    assert entries == {"app.py": "M", "new.py": "A"}

    result = ToolRegistry(default_tools()).dispatch("git_status", {}, make_ctx(git_workspace))
    assert "## main" in result.content and "app.py" in result.content
    assert result.summary == "2 changed files"


def test_status_paths_are_real_file_names(git_workspace, make_ctx):
    # porcelain without -z quotes such names: café.py came back as caf\303\251.py, so the task summary,
    # which matches status paths against the files the agent changed, left it out
    for name in ("café.py", "naïve test.py"):
        (git_workspace / name).write_text("x\n")
    git(git_workspace, "mv", "app.py", "renamed_é.py")
    entries = {e.path: e.status for e in status_entries(git_workspace)}
    assert entries == {"café.py": "??", "naïve test.py": "??", "renamed_é.py": "R "}
    assert all((git_workspace / path).exists() for path in entries)
    shown = ToolRegistry(default_tools()).dispatch("git_status", {}, make_ctx(git_workspace)).content
    assert "café.py" in shown and "renamed_é.py" in shown and "\\303" not in shown  # what the model reads


def test_diff_tool(git_workspace, make_ctx):
    (git_workspace / "app.py").write_text("print('changed')\n")
    (git_workspace / "new.py").write_text("x = 1\n")
    assert "+print('changed')" in diff_text(git_workspace)
    registry = ToolRegistry(default_tools())
    result = registry.dispatch("git_diff", {}, make_ctx(git_workspace))
    assert "-print('hello')" in result.content
    assert "Untracked files:" in result.content and "new.py" in result.content
    stat = registry.dispatch("git_diff", {"stat": True}, make_ctx(git_workspace))
    assert "app.py" in stat.content and "1 file changed" in stat.content


def test_log_and_branch_tools(git_workspace, make_ctx):
    registry = ToolRegistry(default_tools())
    log = registry.dispatch("git_log", {"limit": 5}, make_ctx(git_workspace))
    assert "init" in log.content
    branches = registry.dispatch("git_branch", {}, make_ctx(git_workspace))
    assert "main" in branches.content


def test_fresh_repo_without_commits(workspace):
    git(workspace, "init", "-q", "-b", "trunk")
    assert current_branch(workspace) == "trunk"
