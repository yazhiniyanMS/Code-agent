"""Inference: load a trained YCode model and generate answers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import torch

from ycode.lm.data import format_chat
from ycode.lm.tokenizer import END, USER
from ycode.lm.train import load_checkpoint, pick_device


class LocalLM:
    def __init__(self, model_dir: Path | str, *, device: str = "auto") -> None:
        self.device = pick_device(device)
        if self.device == "cpu":
            torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.model, self.tokenizer, self.payload = load_checkpoint(Path(model_dir), self.device)
        self.model.eval()
        self.path = Path(model_dir)

    @property
    def version(self) -> int:
        return int(self.payload.get("config", {}).get("arch_version", 1))

    @property
    def stage(self) -> str:
        return str(self.payload.get("stage", "pretrain"))

    @property
    def num_params(self) -> int:
        return self.model.num_params()

    def _stream(self, prompt_ids: list[int], *, max_new_tokens: int, temperature: float, top_k: int | None,
                top_p: float | None, stop_ids: set[int], on_text: Callable[[str], None] | None) -> str:
        generated: list[int] = []
        emitted = 0

        def on_token(token: int) -> None:
            nonlocal emitted
            generated.append(token)
            text = self.tokenizer.decode(generated, skip_special=True)
            # Hold back a trailing partial UTF-8 character.
            if text.endswith("�"):
                return
            if on_text is not None and len(text) > emitted:
                on_text(text[emitted:])
            emitted = len(text)

        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        self.model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
                            stop_ids=stop_ids, on_token=on_token)
        text = self.tokenizer.decode(generated, skip_special=True)
        if on_text is not None and len(text) > emitted:
            on_text(text[emitted:])
        return text

    def complete(self, prompt: str, *, max_new_tokens: int = 200, temperature: float = 0.8,
                 top_k: int | None = 50, top_p: float | None = 0.95,
                 on_text: Callable[[str], None] | None = None) -> str:
        """Raw continuation, e.g. complete('def fibonacci(n):')."""
        ids = self.tokenizer.encode(prompt, allow_special=False)
        return self._stream(ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
                            top_p=top_p, stop_ids={self.tokenizer.eot_id}, on_text=on_text)

    def chat(self, turns: list[tuple[str, str]], *, max_new_tokens: int = 300, temperature: float = 0.7,
             top_k: int | None = 40, top_p: float | None = 0.9,
             on_text: Callable[[str], None] | None = None) -> str:
        """Answer the last user turn. ``turns`` is a list of (role, text), role in {user, assistant}."""
        text = ""
        for role, content in turns:
            if role == "user":
                text += format_chat(content)
            else:
                text += f"{content.strip()}\n{END}\n"
        ids = self.tokenizer.encode(text)
        # Keep the prompt within the context window (the model's generate()
        # trims from the left, so make sure the newest turn survives).
        stop = {self.tokenizer.token_id(END), self.tokenizer.eot_id, self.tokenizer.token_id(USER)}
        return self._stream(ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
                            top_p=top_p, stop_ids=stop, on_text=on_text).strip()

