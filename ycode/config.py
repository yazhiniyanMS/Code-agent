"""Configuration loading.

Precedence (lowest to highest):

1. built-in defaults
2. global config      ``~/.ycode/config.toml``
3. project config     ``<project>/.ycode/config.toml``
4. environment        ``YCODE_*`` variables (``.env`` files are loaded first)
5. command-line flags

The project config lives inside a repository you may have just cloned, so it
is not trusted to *loosen* security: it cannot enable ``approval_mode = "auto"``
or ``allow_outside_workspace``. It can only make things stricter.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from dotenv import load_dotenv

from ycode.errors import ConfigError

DEFAULT_MODEL = "claude-opus-5-5"

DEFAULT_IGNORE_DIRS: tuple[str, ...] = (
    "node_modules",
    ".git",
    "dist",
    "build",
    ".next",
    ".nuxt",
    ".cache",
    "venv",
    ".venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "target",
    "coverage",
    ".gradle",
    ".idea",
    ".dart_tool",
    ".tox",
    ".eggs",
)

APPROVAL_MODES = ("strict", "normal", "auto")
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
THEMES = ("default", "mono")

# Ordering of approval modes from most to least strict.
_STRICTNESS = {"strict": 0, "normal": 1, "auto": 2}


def ycode_home() -> Path:
    """Directory for global YCode state (config, history, .env)."""
    override = os.environ.get("YCODE_HOME")
    return Path(override).expanduser() if override else Path.home() / ".ycode"


@dataclass(frozen=True)
class Config:
    model: str = DEFAULT_MODEL
    provider: str = "anthropic"
    max_steps: int = 50
    max_tokens: int = 64000
    effort: str | None = "high"
    approval_mode: str = "normal"
    theme: str = "default"
    command_timeout: int = 120
    max_output_chars: int = 30000
    ignore_dirs: tuple[str, ...] = DEFAULT_IGNORE_DIRS
    allow_outside_workspace: bool = False
    refusal_fallback: bool = True
    # Where each value came from, for /status and debugging.
    sources: tuple[str, ...] = field(default=(), compare=False)

    def with_overrides(self, **overrides: Any) -> "Config":
        clean = {k: v for k, v in overrides.items() if v is not None}
        return _validate(replace(self, **clean)) if clean else self


# ------------------------------------------------------------------ parsing

_FIELD_NAMES = {f.name for f in fields(Config)} - {"sources"}


def _coerce(key: str, value: Any, origin: str) -> Any:
    def bad(expected: str) -> ConfigError:
        return ConfigError(f"{origin}: `{key}` must be {expected}, got {value!r}")

    if key in ("max_steps", "max_tokens", "command_timeout", "max_output_chars"):
        if isinstance(value, bool):
            raise bad("an integer")
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise bad("an integer") from None
        if number <= 0:
            raise bad("a positive integer")
        return number
    if key in ("allow_outside_workspace", "refusal_fallback"):
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("1", "true", "yes", "on"):
            return True
        if isinstance(value, str) and value.strip().lower() in ("0", "false", "no", "off"):
            return False
        raise bad("a boolean")
    if key == "ignore_dirs":
        if isinstance(value, str):
            value = [v.strip() for v in value.split(",")]
        if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
            raise bad("a list of strings")
        return tuple(v for v in value if v)
    if key == "effort":
        if value in (None, "", "none", "default"):
            return None
        return str(value).strip().lower()
    if not isinstance(value, str):
        raise bad("a string")
    return value.strip()


def _validate(config: Config) -> Config:
    if config.approval_mode not in APPROVAL_MODES:
        raise ConfigError(
            f"approval_mode must be one of {', '.join(APPROVAL_MODES)}; got {config.approval_mode!r}"
        )
    if config.effort is not None and config.effort not in EFFORT_LEVELS:
        raise ConfigError(f"effort must be one of {', '.join(EFFORT_LEVELS)}; got {config.effort!r}")
    if config.theme not in THEMES:
        raise ConfigError(f"theme must be one of {', '.join(THEMES)}; got {config.theme!r}")
    if not config.model:
        raise ConfigError("model must not be empty")
    return config


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from None
    except OSError as exc:
        raise ConfigError(f"Could not read {path}: {exc}") from None
    return data


def _apply_mapping(
    values: dict[str, Any], data: Mapping[str, Any], origin: str, *, trusted: bool
) -> None:
    extra_ignore = data.get("extra_ignore_dirs")
    for key, raw in data.items():
        if key == "extra_ignore_dirs":
            continue
        if key not in _FIELD_NAMES:
            raise ConfigError(f"{origin}: unknown setting `{key}`")
        value = _coerce(key, raw, origin)
        if not trusted:
            if key == "allow_outside_workspace" and value:
                continue  # project config cannot widen filesystem access
            if key == "approval_mode" and _STRICTNESS.get(value, 0) > _STRICTNESS.get(
                values.get("approval_mode", "normal"), 1
            ):
                continue  # project config can only tighten approvals
        values[key] = value
    if extra_ignore is not None:
        extra = _coerce("ignore_dirs", extra_ignore, origin)
        base = values.get("ignore_dirs", DEFAULT_IGNORE_DIRS)
        values["ignore_dirs"] = tuple(dict.fromkeys((*base, *extra)))


_ENV_MAP = {
    "YCODE_MODEL": "model",
    "YCODE_MAX_STEPS": "max_steps",
    "YCODE_MAX_TOKENS": "max_tokens",
    "YCODE_EFFORT": "effort",
    "YCODE_APPROVAL_MODE": "approval_mode",
    "YCODE_THEME": "theme",
    "YCODE_COMMAND_TIMEOUT": "command_timeout",
    "YCODE_IGNORE_DIRS": "ignore_dirs",
}


def load_env_files(project_root: Path) -> list[Path]:
    """Load ``.env`` files without overriding variables already set.

    Order: project ``.env``, then ``~/.ycode/.env``, then the YCode checkout's own
    ``.env`` (useful for ``pip install -e .`` setups).
    """
    candidates = [
        project_root / ".env",
        ycode_home() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    loaded: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        load_dotenv(resolved, override=False)
        loaded.append(resolved)
    return loaded


def load_config(
    project_root: Path,
    *,
    env: Mapping[str, str] | None = None,
    global_path: Path | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> Config:
    env = os.environ if env is None else env
    values: dict[str, Any] = {}
    sources: list[str] = ["defaults"]

    global_path = global_path or ycode_home() / "config.toml"
    if global_path.is_file():
        _apply_mapping(values, _read_toml(global_path), str(global_path), trusted=True)
        sources.append(str(global_path))

    project_path = project_root / ".ycode" / "config.toml"
    if project_path.is_file():
        _apply_mapping(values, _read_toml(project_path), str(project_path), trusted=False)
        sources.append(str(project_path))

    env_values = {key: env[name] for name, key in _ENV_MAP.items() if env.get(name, "").strip()}
    if env_values:
        _apply_mapping(values, env_values, "environment", trusted=True)
        sources.append("environment")

    if cli_overrides:
        cli_values = {k: v for k, v in cli_overrides.items() if v is not None}
        if cli_values:
            _apply_mapping(values, cli_values, "command line", trusted=True)
            sources.append("command line")

    return _validate(Config(**values, sources=tuple(sources)))
