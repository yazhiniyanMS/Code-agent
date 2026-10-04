"""Byte-level BPE tokenizer for source code, implemented from scratch.

* Base vocabulary: the 256 byte values, so any text (any language, any
  Unicode) can be encoded with no unknown tokens.
* Merges are learned from the training corpus (classic BPE).
* Text is first split into chunks with a code-aware regex so merges never
  cross identifier/operator/whitespace boundaries; runs of indentation
  become single tokens, which matters a lot for Python.
* Special tokens (``<|endoftext|>``, chat markers) are appended after the
  learned vocabulary and are never produced by merges.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

EOT = "<|endoftext|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END = "<|end|>"
SPECIAL_TOKENS = (EOT, USER, ASSISTANT, END)

# Code-aware pre-tokenization: newline+indent runs, identifiers (with one
# leading space), short digit runs, operator runs, other whitespace.
CHUNK_RE = re.compile(
    r"\n[ \t]*"
    r"| ?[A-Za-z_][A-Za-z0-9_]*"
    r"| ?[0-9]{1,3}"
    r"| ?[^\sA-Za-z0-9_]+"
    r"|[ \t]+(?=\S)|[ \t]+"
    r"|\s"
)


def split_chunks(text: str) -> list[str]:
    return CHUNK_RE.findall(text)


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None,
                 special_tokens: Iterable[str] = SPECIAL_TOKENS) -> None:
        self.merges: list[tuple[int, int]] = list(merges or [])
        self.ranks: dict[tuple[int, int], int] = {pair: i for i, pair in enumerate(self.merges)}
        # id -> bytes for every non-special token
        self.vocab: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.special_tokens = list(special_tokens)
        base = len(self.vocab)
        self.special_ids = {tok: base + i for i, tok in enumerate(self.special_tokens)}
        self.special_by_id = {i: tok for tok, i in self.special_ids.items()}
        self._special_re = (
            re.compile("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")")
            if self.special_tokens else None
        )
        self._cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------- props

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    def token_id(self, special: str) -> int:
        return self.special_ids[special]

    @property
    def eot_id(self) -> int:
        return self.special_ids[EOT]

    # ---------------------------------------------------------- training

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int, *, min_frequency: int = 2,
              special_tokens: Iterable[str] = SPECIAL_TOKENS, verbose: bool = False) -> "BPETokenizer":
        """Learn ``vocab_size - 256 - len(special)`` merges from ``texts``."""
        special_tokens = list(special_tokens)
        n_merges = vocab_size - 256 - len(special_tokens)
        if n_merges < 0:
            raise ValueError(f"vocab_size must be at least {256 + len(special_tokens)}")

        chunk_counts: Counter[str] = Counter()
        for text in texts:
            chunk_counts.update(split_chunks(text))

        words: list[list[int]] = []
        freqs: list[int] = []
        for chunk, count in chunk_counts.items():
            data = list(chunk.encode("utf-8"))
            if len(data) > 1:
                words.append(data)
                freqs.append(count)

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for idx, word in enumerate(words):
            for pair in zip(word, word[1:]):
                pair_counts[pair] += freqs[idx]
                where[pair].add(idx)

        merges: list[tuple[int, int]] = []
        next_id = 256
        for step in range(n_merges):
            if not pair_counts:
                break
            best = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            pair, count = best
            if count < min_frequency:
                break
            merges.append(pair)
            new_id = next_id
            next_id += 1
            for idx in list(where.get(pair, ())):
                word = words[idx]
                freq = freqs[idx]
                # Remove this word's old pair contributions.
                for p in zip(word, word[1:]):
                    pair_counts[p] -= freq
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                    where[p].discard(idx)
                merged: list[int] = []
                i = 0
                while i < len(word):
                    if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(word[i])
                        i += 1
                words[idx] = merged
                for p in zip(merged, merged[1:]):
                    pair_counts[p] += freq
                    where[p].add(idx)
            where.pop(pair, None)
            if verbose and (step + 1) % 500 == 0:
                print(f"  merges: {step + 1}/{n_merges}")
        return cls(merges, special_tokens)

    # ---------------------------------------------------------- encoding

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            best_rank, best_pos = None, -1
            for i in range(len(ids) - 1):
                rank = self.ranks.get((ids[i], ids[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pos = rank, i
            if best_rank is None:
                break
            ids[best_pos: best_pos + 2] = [256 + best_rank]
        if len(self._cache) < 200_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str, *, allow_special: bool = True) -> list[int]:
        out: list[int] = []
        parts = self._special_re.split(text) if (allow_special and self._special_re) else [text]
        for part in parts:
            if not part:
                continue
            if allow_special and part in self.special_ids:
                out.append(self.special_ids[part])
                continue
            for chunk in split_chunks(part):
                out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: Iterable[int], *, skip_special: bool = False) -> str:
        buf = bytearray()
        pieces: list[str] = []
        for i in ids:
            if i in self.special_by_id:
                if buf:
                    pieces.append(buf.decode("utf-8", errors="replace"))
                    buf = bytearray()
                if not skip_special:
                    pieces.append(self.special_by_id[i])
            elif 0 <= i < len(self.vocab):
                buf.extend(self.vocab[i])
        if buf:
            pieces.append(buf.decode("utf-8", errors="replace"))
        return "".join(pieces)

    # ------------------------------------------------------- persistence

    def to_dict(self) -> dict:
        return {"type": "ycode-bpe", "version": 1, "merges": self.merges,
                "special_tokens": self.special_tokens}

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> "BPETokenizer":
        if data.get("type") != "ycode-bpe":
            raise ValueError("not a YCode tokenizer file")
        return cls([tuple(m) for m in data["merges"]], data.get("special_tokens", SPECIAL_TOKENS))

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
