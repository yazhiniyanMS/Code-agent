"""Repository understanding: what kind of project is this, cheaply.

Only marker/config files are inspected; the agent reads source on demand.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from pathlib import Path

from ycode.tools import git

INSTRUCTION_FILES = ("YCODE.md", ".ycode/YCODE.md")
MAX_INSTRUCTIONS_CHARS = 20000

# marker file -> project type
_MARKERS: list[tuple[str, str]] = [
    ("package.json", "Node.js"),
    ("tsconfig.json", "TypeScript"),
    ("pyproject.toml", "Python"),
    ("setup.py", "Python"),
    ("requirements.txt", "Python"),
    ("Pipfile", "Python"),
    ("pom.xml", "Java"),
    ("build.gradle", "Java"),
    ("build.gradle.kts", "Kotlin"),
    ("Cargo.toml", "Rust"),
    ("go.mod", "Go"),
    ("CMakeLists.txt", "C/C++"),
    ("Makefile", "Make"),
    ("pubspec.yaml", "Dart"),
    ("Gemfile", "Ruby"),
    ("composer.json", "PHP"),
    ("deno.json", "Deno"),
]

_INTERESTING_FILES = (
    "package.json", "tsconfig.json", "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
    "Pipfile", "pom.xml", "build.gradle", "build.gradle.kts", "Cargo.toml", "go.mod", "CMakeLists.txt",
    "Makefile", "pubspec.yaml", "Gemfile", "composer.json", "pytest.ini", "tox.ini", "jest.config.js",
    "jest.config.ts", "vitest.config.ts", "vite.config.ts", "next.config.js", "next.config.mjs",
    "next.config.ts", "nuxt.config.ts", "angular.json", ".eslintrc.json", "eslint.config.js",
    "Dockerfile", "docker-compose.yml", "README.md",
)


@dataclass
class ProjectContext:
    root: Path
    os_name: str
    is_git_repo: bool
    branch: str | None
    project_types: list[str] = field(default_factory=list)
    key_files: list[str] = field(default_factory=list)
    package_manager: str | None = None
    scripts: dict[str, str] = field(default_factory=dict)
    verification_hints: list[str] = field(default_factory=list)
    top_level: list[str] = field(default_factory=list)
    instructions: str | None = None
    instructions_path: str | None = None

    def summary(self) -> str:
        lines = [
            f"Project root: {self.root}",
            f"Operating system: {self.os_name}",
            f"Git repository: {'yes' if self.is_git_repo else 'no'}"
            + (f" (branch: {self.branch})" if self.branch else ""),
            f"Detected project type(s): {', '.join(self.project_types) or 'unknown'}",
        ]
        if self.package_manager:
            lines.append(f"Package manager: {self.package_manager}")
        if self.key_files:
            lines.append(f"Key config files: {', '.join(self.key_files)}")
        if self.scripts:
            scripts = ", ".join(f"{k}: `{v}`" for k, v in list(self.scripts.items())[:15])
            lines.append(f"package.json scripts: {scripts}")
        if self.verification_hints:
            lines.append(f"Likely verification commands: {', '.join(self.verification_hints)}")
        if self.top_level:
            lines.append(f"Top-level entries: {', '.join(self.top_level[:60])}")
        return "\n".join(lines)


def _node_details(root: Path, ctx: ProjectContext) -> None:
    try:
        data = json.loads((root / "package.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    deps = {**(data.get("dependencies") or {}), **(data.get("devDependencies") or {})}
    for dep, label in (("next", "Next.js"), ("react", "React"), ("vue", "Vue"), ("svelte", "Svelte"),
                       ("@angular/core", "Angular"), ("express", "Express"), ("typescript", "TypeScript")):
        if dep in deps and label not in ctx.project_types:
            ctx.project_types.append(label)
    if "TypeScript" not in ctx.project_types and "Node.js" in ctx.project_types:
        ctx.project_types.append("JavaScript")
    scripts = data.get("scripts") or {}
    if isinstance(scripts, dict):
        ctx.scripts = {str(k): str(v) for k, v in scripts.items()}
    if (root / "pnpm-lock.yaml").exists():
        ctx.package_manager = "pnpm"
    elif (root / "yarn.lock").exists():
        ctx.package_manager = "yarn"
    elif (root / "bun.lockb").exists() or (root / "bun.lock").exists():
        ctx.package_manager = "bun"
    else:
        ctx.package_manager = "npm"
    runner = ctx.package_manager
    for name in ("test", "typecheck", "lint", "build"):
        if name in ctx.scripts:
            ctx.verification_hints.append(f"{runner} test" if name == "test" else f"{runner} run {name}")


def detect_project(root: Path, ignore_dirs: tuple[str, ...] = ()) -> ProjectContext:
    root = root.resolve()
    is_repo = git.is_git_repo(root)
    ctx = ProjectContext(
        root=root,
        os_name=f"{platform.system()} {platform.release()}".strip(),
        is_git_repo=is_repo,
        branch=git.current_branch(root) if is_repo else None,
    )
    for marker, label in _MARKERS:
        if (root / marker).exists() and label not in ctx.project_types:
            ctx.project_types.append(label)
    if "Dart" in ctx.project_types:
        try:
            if "flutter" in (root / "pubspec.yaml").read_text(encoding="utf-8"):
                ctx.project_types.append("Flutter")
        except OSError:
            pass
    ctx.key_files = [f for f in _INTERESTING_FILES if (root / f).exists()]

    if "Node.js" in ctx.project_types:
        _node_details(root, ctx)
    if "Python" in ctx.project_types:
        has_tests = (root / "tests").is_dir() or (root / "test").is_dir()
        if has_tests or (root / "pytest.ini").exists():
            ctx.verification_hints.append("pytest")
    if "Rust" in ctx.project_types:
        ctx.verification_hints += ["cargo build", "cargo test"]
    if "Go" in ctx.project_types:
        ctx.verification_hints += ["go build ./...", "go test ./..."]
    if "Java" in ctx.project_types:
        ctx.verification_hints.append("mvn test" if (root / "pom.xml").exists() else "./gradlew test")
    if "Flutter" in ctx.project_types:
        ctx.verification_hints += ["flutter analyze", "flutter test"]
    if "C/C++" in ctx.project_types:
        ctx.verification_hints.append("cmake --build build")

    ignore = set(ignore_dirs)
    try:
        entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        ctx.top_level = [e.name + ("/" if e.is_dir() else "") for e in entries if e.name not in ignore]
    except OSError:
        pass

    ctx.instructions, ctx.instructions_path = load_instructions(root)
    return ctx


def load_instructions(root: Path) -> tuple[str | None, str | None]:
    """Load YCODE.md project instructions, if present."""
    for name in INSTRUCTION_FILES:
        path = root / name
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                continue
            if len(text) > MAX_INSTRUCTIONS_CHARS:
                text = text[:MAX_INSTRUCTIONS_CHARS] + "\n\n[YCODE.md truncated]"
            return (text or None), name
    return None, None
