from __future__ import annotations

import io
import subprocess
from pathlib import Path
from typing import Callable

import pytest
from rich.console import Console

from ycode.llm.base import (
    AssistantMessage,
    LLMProvider,
    LLMResponse,
    StopReason,
    ToolCall,
    Usage,
)
from ycode.security.permissions import PermissionManager
from ycode.tools.base import ToolContext
from ycode.ui.console import THEME, ConsoleUI


@pytest.fixture(autouse=True)
def isolated_home(tmp_path_factory, monkeypatch):
    """Never read the developer's real ~/.ycode or API key during tests."""
    home = tmp_path_factory.mktemp("ycode-home")
    monkeypatch.setenv("YCODE_HOME", str(home))
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "YCODE_MODEL",
                "YCODE_MAX_STEPS", "YCODE_APPROVAL_MODE", "YCODE_EFFORT"):
        monkeypatch.delenv(var, raising=False)
    return home


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def git_workspace(workspace: Path) -> Path:
    git(workspace, "init", "-q", "-b", "main")
    git(workspace, "config", "user.email", "test@example.com")
    git(workspace, "config", "user.name", "Test")
    (workspace / "app.py").write_text("print('hello')\n")
    git(workspace, "add", ".")
    git(workspace, "commit", "-q", "-m", "init")
    return workspace


class ScriptedApprover:
    def __init__(self, answer: str = "no") -> None:
        self.answer = answer
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return self.answer


@pytest.fixture
def approver() -> ScriptedApprover:
    return ScriptedApprover()


@pytest.fixture
def make_ctx(approver) -> Callable[..., ToolContext]:
    def factory(root: Path, mode: str = "normal", **kwargs) -> ToolContext:
        perms = PermissionManager(root, mode=mode, approver=approver)
        return ToolContext(workspace=root.resolve(), permissions=perms,
                           ignore_dirs=("node_modules", ".git", "dist"), **kwargs)
    return factory


def recording_console(width: int = 120) -> Console:
    return Console(file=io.StringIO(), width=width, theme=THEME, force_terminal=False,
                   color_system=None, highlight=False)


def console_text(console: Console) -> str:
    return console.file.getvalue()


@pytest.fixture
def ui() -> ConsoleUI:
    return ConsoleUI(recording_console())


# ------------------------------------------------------------ fake LLM


def reply(text: str = "", calls: list[tuple[str, dict]] | None = None,
          stop: StopReason | None = None) -> LLMResponse:
    tool_calls = [ToolCall(id=f"call_{i}_{name}", name=name, arguments=args)
                  for i, (name, args) in enumerate(calls or [])]
    if stop is None:
        stop = StopReason.TOOL_USE if tool_calls else StopReason.END_TURN
    return LLMResponse(AssistantMessage(text=text, tool_calls=tool_calls, provider="fake"), stop,
                       Usage(input_tokens=10, output_tokens=5))


class FakeProvider(LLMProvider):
    """Returns scripted responses and records every request."""

    name = "fake"

    def __init__(self, responses: list[LLMResponse | BaseException] | None = None) -> None:
        super().__init__("fake-model")
        self.responses = list(responses or [])
        self.requests: list[dict] = []

    def complete(self, *, system, messages, tools, callbacks=None):
        self.requests.append({"system": system, "messages": list(messages), "tools": tools})
        if not self.responses:
            return reply("Done.")
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callbacks and item.message.text:
            callbacks.text(item.message.text)
        return item


# ------------------------------------------------------------- YCode-LM data


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    """A small prepared YCode-LM dataset (skipped when torch/numpy are missing)."""
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    from tests.test_lm import CODE
    from ycode.lm.data import prepare_dataset

    src = tmp_path_factory.mktemp("src")
    for i in range(6):
        (src / f"m{i}.py").write_text(CODE.replace("add", f"add{i}") * 3)
    (src / "skip_me.py").write_text("def secret():\n    return 1\n" * 50)
    out = tmp_path_factory.mktemp("data")
    meta = prepare_dataset([src], out, vocab_size=400, exclude=["*skip_me*"], workers=1, log=lambda *_: None)
    assert meta["files"] == 6
    return out
