"""Local provider: YCode's own from-scratch model (``ycode-lm``), no API key.

A small model trained on a laptop can answer coding questions and write
short functions, but it has not learned the tool-calling protocol, so it
runs in answer-only mode: it never reads, edits or runs anything.
"""

from __future__ import annotations

from pathlib import Path

from ycode.errors import ConfigError
from ycode.llm.base import (
    AssistantMessage,
    LLMProvider,
    LLMResponse,
    Message,
    StopReason,
    StreamCallbacks,
    ToolSpec,
    Usage,
    UserMessage,
)

_NOTE_PREFIX = "[Note:"


class LocalProvider(LLMProvider):
    name = "local"
    supports_tools = False

    def __init__(self, model_dir: Path, *, max_new_tokens: int = 400, temperature: float = 0.7,
                 device: str = "auto", lm=None) -> None:
        super().__init__(str(model_dir))
        self.model_dir = Path(model_dir)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.lm = lm if lm is not None else self._load(device)

    def _load(self, device: str):
        try:
            from ycode.lm.generate import LocalLM
        except ImportError:
            raise ConfigError(
                "The local model needs PyTorch.",
                hint='Install it with: pip install -e ".[local]"',
            ) from None
        try:
            return LocalLM(self.model_dir, device=device)
        except FileNotFoundError:
            raise ConfigError(
                f"No trained model found at {self.model_dir}.",
                hint="Train one first: `ycode-lm prepare --out data/`, `ycode-lm train --data data/ --out "
                     "models/base`, `ycode-lm sft --data data/ --init-from models/base --out "
                     f"{self.model_dir}` (see README).",
            ) from None
        except (ValueError, RuntimeError) as exc:
            raise ConfigError(f"Could not load the local model at {self.model_dir}: {exc}") from None

    @property
    def display_name(self) -> str:
        params = getattr(self.lm, "num_params", 0)
        version = getattr(self.lm, "version", None)
        size = f"{params / 1e6:.1f}M params, " if params else ""
        name = f"YCode-LM v{version}" if isinstance(version, int) else "YCode-LM"
        return f"{name} ({size}local, {self.model_dir})"

    @staticmethod
    def _turns(messages: list[Message]) -> list[tuple[str, str]]:
        turns: list[tuple[str, str]] = []
        for msg in messages:
            if isinstance(msg, UserMessage):
                text = msg.text
                if text.startswith(_NOTE_PREFIX) and "\n\n" in text:
                    text = text.split("\n\n", 1)[1]
                turns.append(("user", text))
            elif isinstance(msg, AssistantMessage) and msg.text:
                turns.append(("assistant", msg.text))
        return turns

    def complete(self, *, system: str, messages: list[Message], tools: list[ToolSpec],
                 callbacks: StreamCallbacks | None = None) -> LLMResponse:
        callbacks = callbacks or StreamCallbacks()
        turns = self._turns(messages)[-6:]
        if not turns or turns[-1][0] != "user":
            turns.append(("user", "Continue."))
        answer = self.lm.chat(turns, max_new_tokens=self.max_new_tokens, temperature=self.temperature,
                              on_text=callbacks.text)
        if not answer.strip():
            answer = "(The local model did not produce an answer. Try rephrasing the question.)"
            callbacks.text(answer)
        return LLMResponse(
            AssistantMessage(text=answer, provider=self.name),
            StopReason.END_TURN,
            Usage(output_tokens=len(answer) // 4),
        )
