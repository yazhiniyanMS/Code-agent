"""Corpus collection, instruction-pair extraction and dataset encoding.

Outputs (in the data directory):

* ``tokenizer.json``         the trained BPE tokenizer
* ``train.bin`` / ``val.bin`` pretraining token streams (uint16)
* ``sft_tokens.bin``, ``sft_mask.bin``, ``sft_index.npy``
                             instruction examples (tokens, loss mask, offsets)
* ``meta.json``              statistics and settings
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
import re
import sysconfig
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

from ycode.lm.tokenizer import ASSISTANT, END, USER, BPETokenizer

CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".go", ".rs", ".c", ".h", ".cc", ".cpp",
    ".hpp", ".cs", ".rb", ".php", ".swift", ".scala", ".sh", ".sql", ".lua", ".dart",
}
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", "dist", "build", ".venv", "venv", "env", ".tox",
    "target", ".next", "coverage", ".mypy_cache", ".pytest_cache", "site-packages", "dist-packages",
}
MAX_FILE_BYTES = 500_000

Log = Callable[[str], None]


def default_sources() -> list[Path]:
    """The Python standard library: a large, high-quality, always-available code corpus."""
    return [Path(sysconfig.get_paths()["stdlib"])]


def iter_source_files(sources: Iterable[Path], extensions: set[str] | None = None) -> Iterator[Path]:
    extensions = extensions or CODE_EXTENSIONS
    for source in sources:
        source = Path(source)
        if source.is_file():
            if source.suffix in extensions:
                yield source
            continue
        for path in sorted(source.rglob("*")):
            if any(part in SKIP_DIRS for part in path.relative_to(source).parts[:-1]):
                continue
            if path.suffix in extensions and path.is_file():
                yield path


def read_code_files(sources: Iterable[Path], extensions: set[str] | None = None) -> list[tuple[Path, str]]:
    """Read, filter (binary/minified/huge) and de-duplicate source files."""
    seen: set[str] = set()
    out: list[tuple[Path, str]] = []
    for path in iter_source_files(sources, extensions):
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            raw = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:4096]:
            continue
        text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
        lines = text.splitlines() or [""]
        if not text.strip() or sum(len(l) for l in lines) / len(lines) > 200:
            continue  # empty or minified
        digest = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        out.append((path, text))
    return out


# ------------------------------------------------------------ chat format


def format_chat(prompt: str, response: str | None = None) -> str:
    """The chat template the model is trained on and prompted with."""
    text = f"{USER}\n{prompt.strip()}\n{END}\n{ASSISTANT}\n"
    if response is not None:
        text += f"{response.strip()}\n{END}"
    return text


@dataclass
class InstructionExample:
    prompt: str
    response: str


_PROMPTS_WRITE = (
    "Write a Python function `{sig}` that {task}",
    "Implement `{sig}` in Python. It should {imp}",
    "Create a Python function named `{name}` that {task}",
    "How would I write a Python function to {imp}",
)
_PROMPTS_EXPLAIN = (
    "What does this Python function do?\n```python\n{code}\n```",
    "Explain this code:\n```python\n{code}\n```",
)


def _docstring_summary(doc: str) -> str | None:
    first = textwrap.dedent(doc).strip().split("\n\n")[0]
    summary = " ".join(line.strip() for line in first.splitlines()).strip()
    if not (15 <= len(summary) <= 300) or summary.startswith((">>>", ":", "@")):
        return None
    if re.search(r"(?i)deprecated|todo|xxx|internal use|undocumented", summary):
        return None
    return summary


def _as_task(summary: str) -> str:
    """'Return the sum of x.' -> 'returns the sum of x.'"""
    words = summary.split(" ", 1)
    verb = words[0]
    rest = words[1] if len(words) > 1 else ""
    if verb[:1].isupper() and verb[1:].islower() and verb.isalpha():
        verb = verb.lower()
        if not verb.endswith("s") and verb not in ("is", "has"):
            verb = verb + ("es" if verb.endswith(("sh", "ch", "x", "o")) else "s")
    task = f"{verb} {rest}".strip()
    return task if task.endswith((".", "?", "!")) else task + "."


def _strip_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    clone = ast.parse(ast.unparse(node)).body[0]
    if (clone.body and isinstance(clone.body[0], ast.Expr)
            and isinstance(getattr(clone.body[0], "value", None), ast.Constant)
            and isinstance(clone.body[0].value.value, str)):
        clone.body = clone.body[1:] or [ast.Pass()]
    return ast.unparse(clone)


def extract_python_examples(source: str, rng: random.Random, *, max_chars: int = 1500) -> list[InstructionExample]:
    """Turn documented functions into (instruction, answer) pairs in both directions."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return []
    examples: list[InstructionExample] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name.startswith("_"):
            continue
        doc = ast.get_docstring(node)
        if not doc:
            continue
        summary = _docstring_summary(doc)
        if summary is None:
            continue
        try:
            full = ast.unparse(node)
            bare = _strip_docstring(node)
            args = ast.unparse(node.args)
        except (ValueError, RecursionError):
            continue
        if len(full) > max_chars or len(bare.splitlines()) < 2:
            continue
        if args.startswith("self"):
            args = args[4:].lstrip(", ")
        sig = f"{node.name}({args})"
        task = _as_task(summary)
        imp = summary[0].lower() + summary[1:] if summary[1:2].islower() else summary
        imp = imp if imp.endswith((".", "?", "!")) else imp + "."
        template = rng.choice(_PROMPTS_WRITE)
        examples.append(InstructionExample(
            template.format(sig=sig, name=node.name, task=task, imp=imp),
            f"```python\n{full}\n```",
        ))
        examples.append(InstructionExample(
            rng.choice(_PROMPTS_EXPLAIN).format(code=bare),
            f"The function `{node.name}` {task}",
        ))
    return examples


def load_jsonl_examples(path: Path) -> list[InstructionExample]:
    """User-provided data: {"prompt": ..., "response": ...} per line."""
    examples = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            examples.append(InstructionExample(str(row["prompt"]), str(row["response"])))
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{n}: expected {{\"prompt\": ..., \"response\": ...}} ({exc})") from None
    return examples


# --------------------------------------------------------------- encoding


def encode_example(tok: BPETokenizer, ex: InstructionExample) -> tuple[list[int], list[int]]:
    """Tokens plus a loss mask that is 1 only on the answer (and its end marker)."""
    prompt_ids = tok.encode(format_chat(ex.prompt))
    answer_ids = tok.encode(ex.response.strip(), allow_special=False) + tok.encode("\n" + END)
    return prompt_ids + answer_ids, [0] * len(prompt_ids) + [1] * len(answer_ids)


def prepare_dataset(
    sources: list[Path],
    out_dir: Path,
    *,
    vocab_size: int = 4096,
    val_fraction: float = 0.02,
    extra_sft: list[Path] | None = None,
    tokenizer_sample_chars: int = 8_000_000,
    seed: int = 1337,
    log: Log = print,
) -> dict:
    if not 300 <= vocab_size <= 65535:
        raise ValueError("vocab_size must be between 300 and 65535")
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Collecting code from {', '.join(str(s) for s in sources)} ...")
    files = read_code_files(sources)
    if not files:
        raise ValueError("No source files found. Pass --source with directories that contain code.")
    rng.shuffle(files)
    total_chars = sum(len(t) for _, t in files)
    log(f"  {len(files)} files, {total_chars / 1e6:.1f} MB of code")

    examples: list[InstructionExample] = []
    for path, text in files:
        if path.suffix == ".py":
            examples.extend(extract_python_examples(text, rng))
    for path in extra_sft or []:
        examples.extend(load_jsonl_examples(path))
    rng.shuffle(examples)
    log(f"  {len(examples)} instruction examples")

    sample, size = [], 0
    for _, text in files:
        sample.append(text)
        size += len(text)
        if size >= tokenizer_sample_chars:
            break
    sample += [format_chat(e.prompt, e.response) for e in examples[:5000]]
    log(f"Training tokenizer (vocab {vocab_size}) ...")
    tok = BPETokenizer.train(sample, vocab_size)
    tok.save(out_dir / "tokenizer.json")

    log("Encoding pretraining corpus ...")
    n_val = max(1, int(len(files) * val_fraction))
    splits = {"val": files[:n_val], "train": files[n_val:]} if len(files) > 1 else {"val": files, "train": files}
    counts = {}
    for split, items in splits.items():
        ids: list[int] = []
        for _, text in items:
            ids.extend(tok.encode(text, allow_special=False))
            ids.append(tok.eot_id)
        np.asarray(ids, dtype=np.uint16).tofile(out_dir / f"{split}.bin")
        counts[split] = len(ids)
    log(f"  train {counts['train']:,} tokens, val {counts['val']:,} tokens")

    tokens: list[int] = []
    mask: list[int] = []
    index: list[tuple[int, int]] = []
    for ex in examples:
        t, m = encode_example(tok, ex)
        index.append((len(tokens), len(t)))
        tokens.extend(t)
        mask.extend(m)
    np.asarray(tokens, dtype=np.uint16).tofile(out_dir / "sft_tokens.bin")
    np.asarray(mask, dtype=np.uint8).tofile(out_dir / "sft_mask.bin")
    np.save(out_dir / "sft_index.npy", np.asarray(index, dtype=np.int64).reshape(-1, 2))

    meta = {
        "vocab_size": tok.vocab_size,
        "files": len(files),
        "chars": total_chars,
        "train_tokens": counts["train"],
        "val_tokens": counts["val"],
        "sft_examples": len(examples),
        "sft_tokens": len(tokens),
        "sources": [str(s) for s in sources],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"Wrote dataset to {out_dir}")
    return meta
