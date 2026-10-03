import json

from ycode.agent.context import detect_project, load_instructions
from ycode.agent.prompts import build_system_prompt


def test_detects_node_typescript_react(workspace):
    (workspace / "package.json").write_text(json.dumps({
        "scripts": {"test": "vitest run", "build": "tsc", "typecheck": "tsc --noEmit"},
        "dependencies": {"react": "18", "next": "14"},
        "devDependencies": {"typescript": "5"},
    }))
    (workspace / "tsconfig.json").write_text("{}")
    (workspace / "pnpm-lock.yaml").write_text("")
    (workspace / "node_modules").mkdir()
    ctx = detect_project(workspace, ("node_modules",))
    assert {"Node.js", "TypeScript", "React", "Next.js"} <= set(ctx.project_types)
    assert ctx.package_manager == "pnpm"
    assert "pnpm test" in ctx.verification_hints and "pnpm run typecheck" in ctx.verification_hints
    assert "node_modules/" not in ctx.top_level
    assert not ctx.is_git_repo


def test_detects_python_and_git(git_workspace):
    (git_workspace / "pyproject.toml").write_text("[project]\nname='x'\n")
    (git_workspace / "tests").mkdir()
    ctx = detect_project(git_workspace)
    assert "Python" in ctx.project_types and "pytest" in ctx.verification_hints
    assert ctx.is_git_repo and ctx.branch == "main"


def test_detects_other_ecosystems(workspace):
    (workspace / "Cargo.toml").write_text("")
    (workspace / "go.mod").write_text("")
    (workspace / "pubspec.yaml").write_text("dependencies:\n  flutter:\n    sdk: flutter\n")
    types = detect_project(workspace).project_types
    assert {"Rust", "Go", "Flutter"} <= set(types)


def test_ycode_md_is_loaded_into_prompt(workspace):
    (workspace / "YCODE.md").write_text("# YCode Project Instructions\n\nRun npm test before finishing.\n")
    text, name = load_instructions(workspace)
    assert name == "YCODE.md" and "npm test" in text
    prompt = build_system_prompt(detect_project(workspace))
    assert "<project_instructions>" in prompt and "Run npm test before finishing." in prompt


def test_prompt_without_instructions(workspace):
    prompt = build_system_prompt(detect_project(workspace))
    assert "YCode" in prompt and "project_instructions" not in prompt
    assert str(workspace.resolve()) in prompt
