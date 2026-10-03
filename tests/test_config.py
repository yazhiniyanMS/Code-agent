from pathlib import Path

import pytest

from ycode.config import DEFAULT_IGNORE_DIRS, DEFAULT_MODEL, load_config, load_env_files
from ycode.errors import ConfigError


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_defaults(workspace, isolated_home):
    cfg = load_config(workspace, env={})
    assert cfg.model == DEFAULT_MODEL
    assert cfg.max_steps == 50
    assert cfg.approval_mode == "normal"
    assert cfg.ignore_dirs == DEFAULT_IGNORE_DIRS


def test_global_then_project_then_env_then_cli(workspace, isolated_home):
    write(isolated_home / "config.toml", 'model = "global-model"\nmax_steps = 10\ntheme = "mono"\n')
    write(workspace / ".ycode" / "config.toml", 'model = "project-model"\nmax_steps = 20\n')
    cfg = load_config(workspace, env={})
    assert (cfg.model, cfg.max_steps, cfg.theme) == ("project-model", 20, "mono")

    cfg = load_config(workspace, env={"YCODE_MODEL": "env-model", "YCODE_MAX_STEPS": "30"})
    assert (cfg.model, cfg.max_steps) == ("env-model", 30)

    cfg = load_config(workspace, env={"YCODE_MODEL": "env-model"}, cli_overrides={"model": "cli-model",
                                                                                  "max_steps": None})
    assert cfg.model == "cli-model"
    assert cfg.max_steps == 20


def test_project_config_cannot_loosen_security(workspace, isolated_home):
    write(workspace / ".ycode" / "config.toml",
          'approval_mode = "auto"\nallow_outside_workspace = true\n')
    cfg = load_config(workspace, env={})
    assert cfg.approval_mode == "normal"
    assert cfg.allow_outside_workspace is False


def test_project_config_can_tighten_security(workspace, isolated_home):
    write(workspace / ".ycode" / "config.toml", 'approval_mode = "strict"\n')
    assert load_config(workspace, env={}).approval_mode == "strict"


def test_global_config_may_enable_auto(workspace, isolated_home):
    write(isolated_home / "config.toml", 'approval_mode = "auto"\n')
    assert load_config(workspace, env={}).approval_mode == "auto"


def test_extra_ignore_dirs_extend_defaults(workspace, isolated_home):
    write(workspace / ".ycode" / "config.toml", 'extra_ignore_dirs = ["generated"]\n')
    cfg = load_config(workspace, env={})
    assert "generated" in cfg.ignore_dirs and "node_modules" in cfg.ignore_dirs


@pytest.mark.parametrize("text, message", [
    ('max_steps = "lots"', "integer"),
    ('max_steps = 0', "positive"),
    ('approval_mode = "yolo"', "approval_mode"),
    ('unknown_key = 1', "unknown setting"),
    ('model = [', "Invalid TOML"),
    ('effort = "ultra"', "effort"),
])
def test_invalid_config_raises_clear_error(workspace, isolated_home, text, message):
    write(isolated_home / "config.toml", text + "\n")
    with pytest.raises(ConfigError, match=message):
        load_config(workspace, env={})


def test_invalid_env_value(workspace):
    with pytest.raises(ConfigError, match="YCODE_MAX_STEPS|max_steps"):
        load_config(workspace, env={"YCODE_MAX_STEPS": "abc"})


def test_env_file_loading_does_not_override(workspace, monkeypatch):
    write(workspace / ".env", "ANTHROPIC_API_KEY=from-file\nYCODE_MODEL=file-model\n")
    monkeypatch.setenv("YCODE_MODEL", "already-set")
    loaded = load_env_files(workspace)
    import os
    assert workspace.resolve() / ".env" in loaded
    assert os.environ["ANTHROPIC_API_KEY"] == "from-file"
    assert os.environ["YCODE_MODEL"] == "already-set"
