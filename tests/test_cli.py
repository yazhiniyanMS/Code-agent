import pytest

from ycode import __version__, cli
from ycode.llm.base import LLMProvider

from tests.conftest import FakeProvider, reply


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_missing_api_key_exits_cleanly(workspace, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    code = cli.main(["-C", str(workspace), "--provider", "anthropic", "do something"])
    out = capsys.readouterr().out
    assert code == 1
    assert "ANTHROPIC_API_KEY" in out
    assert "Traceback" not in out
    assert "AI Coding Agent" in out  # the banner still shows


def test_default_needs_no_api_key(workspace, capsys, monkeypatch):
    """Out of the box, YCode answers with its own bundled model: no key, no training."""
    import ycode.llm.local as local
    from ycode.config import DEFAULT_BUNDLED_MODEL

    seen = {}

    class FakeLM:
        num_params, version = 100_000_000, 4

        def chat(self, turns, *, on_text=None, **_kw):
            if on_text:
                on_text("Use sorted(items).")
            return "Use sorted(items)."

    def fake_load(self, device):
        seen["dir"] = self.model_dir
        return FakeLM()

    import ycode.config as config

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(config, "available_memory_gb", lambda: 16.0)
    monkeypatch.setattr(local.LocalProvider, "_load", fake_load)
    code = cli.main(["-C", str(workspace), "--no-banner", "how do I sort a list?"])
    assert code == 0
    assert seen["dir"].name == DEFAULT_BUNDLED_MODEL
    assert "sorted(items)" in capsys.readouterr().out


def test_bad_config_exits_cleanly(workspace, capsys, isolated_home):
    (isolated_home / "config.toml").write_text("max_steps = -1\n")
    assert cli.main(["-C", str(workspace), "x"]) == 2
    assert "max_steps" in capsys.readouterr().out


def test_missing_directory(capsys, tmp_path):
    assert cli.main(["-C", str(tmp_path / "nope"), "x"]) == 2


def test_one_shot_task(workspace, capsys, monkeypatch):
    provider = FakeProvider([reply("", [("list_directory", {})]), reply("All good.")])
    monkeypatch.setattr(cli, "create_provider", lambda config: provider)
    code = cli.main(["-C", str(workspace), "--no-banner", "--max-steps", "7", "inspect", "the", "project"])
    out = capsys.readouterr().out
    assert code == 0
    assert "All good." in out and "YCode completed the task." in out
    assert provider.requests[0]["messages"][0].text == "inspect the project"


def test_cli_flags_reach_config(workspace, monkeypatch):
    seen = {}

    def fake_create(config):
        seen["config"] = config
        return FakeProvider([reply("ok")])

    monkeypatch.setattr(cli, "create_provider", fake_create)
    cli.main(["-C", str(workspace), "--no-banner", "--model", "claude-sonnet-5-5", "--yes", "-p", "hi"])
    assert seen["config"].model == "claude-sonnet-5-5"
    assert seen["config"].approval_mode == "auto"


def test_repl_handles_commands_tasks_and_exit(workspace, monkeypatch, capsys):
    provider = FakeProvider([reply("Hello there.")])
    monkeypatch.setattr(cli, "create_provider", lambda config: provider)
    lines = iter(["/status", "say hello", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    assert cli.main(["-C", str(workspace), "--no-banner"]) == 0
    out = capsys.readouterr().out
    assert "Steps used" in out and "Hello there." in out and "Goodbye." in out
    assert len(provider.requests) == 1


def test_repl_survives_internal_errors_and_eof(workspace, monkeypatch, capsys):
    class Exploding(LLMProvider):
        name = "boom"

        def complete(self, **kwargs):
            raise RuntimeError("unexpected")

    monkeypatch.setattr(cli, "create_provider", lambda config: Exploding("m"))
    lines = iter(["do it"])

    def fake_input(prompt=""):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)
    assert cli.main(["-C", str(workspace), "--no-banner"]) == 0
    out = capsys.readouterr().out
    assert "Internal error: RuntimeError: unexpected" in out
    assert "Traceback" not in out
