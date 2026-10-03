"""User input: prompt_toolkit session with history and multiline editing."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

PROMPT = "ycode> "


class InputReader:
    """Reads one request at a time.

    * Enter submits; Esc+Enter (or Alt+Enter) inserts a newline.
    * A line ending in a backslash continues on the next line.
    * Up/Down walk through persistent history (~/.ycode/history).

    Falls back to plain ``input()`` when stdin is not a terminal.
    """

    def __init__(self, history_path: Path | None = None, commands: Iterable[str] = ()) -> None:
        self._session = None
        if sys.stdin.isatty() and sys.stdout.isatty():
            try:
                self._session = self._build_session(history_path, list(commands))
            except Exception:  # noqa: BLE001 - unusual terminals: degrade to input()
                self._session = None

    @staticmethod
    def _build_session(history_path: Path | None, commands: list[str]):
        from prompt_toolkit import PromptSession
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import FileHistory, InMemoryHistory
        from prompt_toolkit.key_binding import KeyBindings

        bindings = KeyBindings()

        @bindings.add("escape", "enter")
        def _newline(event) -> None:  # noqa: ANN001
            event.current_buffer.insert_text("\n")

        @bindings.add("enter")
        def _submit(event) -> None:  # noqa: ANN001
            buffer = event.current_buffer
            if buffer.text.endswith("\\"):
                buffer.delete_before_cursor(1)
                buffer.insert_text("\n")
            else:
                buffer.validate_and_handle()

        history = InMemoryHistory()
        if history_path is not None:
            try:
                history_path.parent.mkdir(parents=True, exist_ok=True)
                history = FileHistory(str(history_path))
            except OSError:
                pass
        completer = WordCompleter(commands, sentence=True) if commands else None
        return PromptSession(
            history=history,
            auto_suggest=AutoSuggestFromHistory(),
            completer=completer,
            complete_while_typing=True,
            reserve_space_for_menu=4,
            key_bindings=bindings,
            multiline=True,
            prompt_continuation=lambda width, line_number, is_soft_wrap: "." * (width - 1) + " ",
        )

    def read(self) -> str:
        """Return the next request. Raises EOFError (Ctrl+D) / KeyboardInterrupt (Ctrl+C)."""
        if self._session is not None:
            from prompt_toolkit.formatted_text import HTML

            return self._session.prompt(HTML("<ansicyan><b>ycode&gt;</b></ansicyan> "))
        lines = []
        prompt = PROMPT
        while True:
            line = input(prompt)
            if line.endswith("\\"):
                lines.append(line[:-1])
                prompt = "...    "
                continue
            lines.append(line)
            return "\n".join(lines)
