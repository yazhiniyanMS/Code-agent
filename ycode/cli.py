"""Command-line entry point: ``ycode`` / ``python -m ycode``."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
import traceback
from pathlib import Path

from ycode import __version__
from ycode.agent.context import detect_project
from ycode.agent.loop import TaskStatus
from ycode.config import APPROVAL_MODES, Config, load_config, load_env_files, ycode_home
from ycode.errors import YCodeError
from ycode.llm import create_provider
from ycode.ui.console import ConsoleUI, make_console


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ycode",
        description="YCode - an AI coding agent for your terminal.",
    )
    parser.add_argument("task", nargs="*", help="Run a single task non-interactively, then exit.")
    parser.add_argument("-p", "--prompt", help="Same as passing a task: run it and exit.")
    parser.add_argument("-m", "--model", help="Model ID (overrides YCODE_MODEL and config files).")
    parser.add_argument("--provider", choices=("anthropic", "local"),
                        help="LLM provider: Claude API (default) or your own local model.")
    parser.add_argument("--local", nargs="?", const="", default=None, metavar="MODEL_DIR",
                        help="Use your own trained model (no API key). Optional path; "
                             "default ~/.ycode/models/ycode-lm.")
    parser.add_argument("--max-steps", type=int, help="Maximum agent steps per task.")
    parser.add_argument("--approval-mode", choices=APPROVAL_MODES, help="How risky actions are approved.")
    parser.add_argument("--yes", action="store_true",
                        help="Shortcut for --approval-mode auto (blocked commands stay blocked).")
    parser.add_argument("-C", "--directory", help="Project directory (default: current directory).")
    parser.add_argument("--no-banner", action="store_true", help="Skip the start-up banner.")
    parser.add_argument("--version", action="version", version=f"YCode {__version__}")
    return parser


def _debug() -> bool:
    return os.environ.get("YCODE_DEBUG", "").lower() in ("1", "true", "yes")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = make_console()
    ui = ConsoleUI(console)

    workspace = Path(args.directory or os.getcwd()).expanduser()
    if not workspace.is_dir():
        ui.print_error(YCodeError(f"Directory not found: {workspace}"))
        return 2
    workspace = workspace.resolve()

    load_env_files(workspace)
    try:
        config = load_config(
            workspace,
            cli_overrides={
                "model": args.model,
                "provider": "local" if args.local is not None else args.provider,
                "local_model": args.local or None,
                "max_steps": args.max_steps,
                "approval_mode": "auto" if args.yes else args.approval_mode,
            },
        )
    except YCodeError as exc:
        ui.print_error(exc)
        return 2

    if config.theme != "default":
        console = make_console(config.theme)
        ui = ConsoleUI(console)

    task = args.prompt or " ".join(args.task).strip() or None
    try:
        return run(config, workspace, ui, task=task, show_banner=not args.no_banner)
    except KeyboardInterrupt:
        ui.stop()
        console.print()
        return 130


def run(config: Config, workspace: Path, ui: ConsoleUI, *, task: str | None, show_banner: bool) -> int:
    from ycode.app import YCodeApp

    with ui.console.status("Inspecting project...", spinner="dots") if ui.console.is_terminal else contextlib.nullcontext():
        context = detect_project(workspace, config.ignore_dirs)

    try:
        provider = create_provider(config)
    except YCodeError as exc:
        if show_banner:
            from ycode.ui.banner import BannerInfo

            ui.print_banner(BannerInfo(version=__version__, model=config.model, directory=str(workspace),
                                       branch=context.branch, project_types=tuple(context.project_types),
                                       approval_mode=config.approval_mode), show_hint=False)
        ui.print_error(exc)
        return 1

    app = YCodeApp(config=config, workspace=workspace, provider=provider, ui=ui, context=context)
    if show_banner:
        ui.print_banner(app.banner_info())
    if not provider.supports_tools:
        ui.on_notice("Local model: answer-only mode. It answers questions and writes code in its reply, "
                     "but cannot read, edit or run files in your project.", "warning")
        ui.console.print()

    if task is not None:
        result = app.run_task(task)
        return 0 if result.status is TaskStatus.COMPLETED else 1
    return repl(app)


def repl(app) -> int:  # noqa: ANN001 - YCodeApp, imported lazily
    from ycode.commands import command_names
    from ycode.ui.prompts import InputReader

    reader = InputReader(ycode_home() / "history", commands=command_names())
    ui = app.ui
    last_interrupt = 0.0
    while not app.should_exit:
        try:
            line = reader.read()
        except KeyboardInterrupt:
            now = time.monotonic()
            if now - last_interrupt < 2.0:
                break
            last_interrupt = now
            ui.info("(Press Ctrl+C again or Ctrl+D to exit)")
            continue
        except EOFError:
            break
        try:
            app.handle_line(line)
        except KeyboardInterrupt:
            ui.stop()
            ui.info("Interrupted.")
        except YCodeError as exc:
            ui.print_error(exc)
        except Exception as exc:  # noqa: BLE001 - keep the session alive
            ui.stop()
            if _debug():
                traceback.print_exc()
            ui.print_error(YCodeError(
                f"Internal error: {type(exc).__name__}: {exc}",
                hint="Set YCODE_DEBUG=1 to see the full traceback.",
            ))
    ui.stop()
    ui.info("Goodbye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
