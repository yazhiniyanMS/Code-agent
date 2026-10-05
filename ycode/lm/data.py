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


_LICENSE_RE = re.compile(r"(?i)licen[sc]e|copyright|\(c\) \d{4}")


def strip_license_header(text: str) -> str:
    """Drop a leading comment block that is only licence/copyright boilerplate."""
    lines = text.split("\n")
    i = 0
    while i < len(lines) and (lines[i].startswith("#") or not lines[i].strip()):
        i += 1
    header = "\n".join(lines[:i])
    if i > 3 and _LICENSE_RE.search(header) and not lines[0].startswith("#!"):
        return "\n".join(lines[i:])
    return text


def read_code_files(sources: Iterable[Path], extensions: set[str] | None = None,
                    exclude: Iterable[str] = ()) -> list[tuple[Path, str]]:
    """Read, filter (binary/minified/huge/excluded) and de-duplicate source files."""
    import fnmatch

    exclude = list(exclude)
    seen: set[str] = set()
    out: list[tuple[Path, str]] = []
    for path in iter_source_files(sources, extensions):
        if any(fnmatch.fnmatch(str(path), pattern) for pattern in exclude):
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            raw = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:4096]:
            continue
        text = strip_license_header(raw.decode("utf-8", errors="replace").replace("\r\n", "\n"))
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


_PROMPTS_COMPLETE = (
    "Complete this Python function:\n```python\n{stub}\n```",
    "Fill in the body of this function:\n```python\n{stub}\n```",
)
_PROMPTS_DOCSTRING = (
    "Write a docstring for this function:\n```python\n{code}\n```",
    "Document this Python function with a one-line docstring:\n```python\n{code}\n```",
)
_PROMPTS_BUGFIX = (
    "This function has a bug. Find and fix it:\n```python\n{code}\n```",
    "Fix the bug in this code:\n```python\n{code}\n```",
    "Why is this function wrong? Fix it.\n```python\n{code}\n```",
)
_PROMPTS_CLASS = (
    "What is the `{name}` class for?\n```python\n{code}\n```",
    "Explain this class:\n```python\n{code}\n```",
)

# Operator swaps used to inject realistic single-token bugs.
_SWAPS: dict[type, type] = {
    ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add, ast.FloorDiv: ast.Div, ast.Div: ast.FloorDiv,
    ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq, ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is,
    ast.And: ast.Or, ast.Or: ast.And,
}


def inject_bug(func_source: str, rng: random.Random) -> tuple[str, str, str] | None:
    """Return (buggy_source, buggy_line, fixed_line) or None if no mutation applies.

    The original code is the fix, so every generated bug-fix example is correct by
    construction."""
    try:
        tree = ast.parse(func_source)
    except (SyntaxError, ValueError, RecursionError):
        return None
    candidates: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and type(node.op) in _SWAPS:
            candidates.append(node)
        elif isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _SWAPS:
            candidates.append(node)
        elif isinstance(node, ast.BoolOp) and type(node.op) in _SWAPS:
            candidates.append(node)
        elif isinstance(node, ast.Constant) and isinstance(node.value, bool):
            candidates.append(node)
        elif isinstance(node, ast.Constant) and type(node.value) is int and node.value in (0, 1):
            candidates.append(node)
    if not candidates:
        return None
    node = rng.choice(candidates)
    try:
        fixed = ast.unparse(node)
        if isinstance(node, ast.BinOp):
            node.op = _SWAPS[type(node.op)]()
        elif isinstance(node, ast.Compare):
            node.ops = [_SWAPS[type(node.ops[0])]()]
        elif isinstance(node, ast.BoolOp):
            node.op = _SWAPS[type(node.op)]()
        elif isinstance(node.value, bool):
            node.value = not node.value
        else:
            node.value = 1 - node.value
        buggy = ast.unparse(tree)
        original = ast.unparse(ast.parse(func_source))
    except (ValueError, RecursionError):
        return None
    if buggy == original or len(fixed) > 160:
        return None
    # Report the whole line that changed: far more informative than the token alone.
    pairs = [(b, o) for b, o in zip(buggy.splitlines(), original.splitlines()) if b != o]
    if len(pairs) != 1 or len(pairs[0][0].strip()) > 160:
        return None
    return buggy, pairs[0][0].strip(), pairs[0][1].strip()


def _with_short_doc(node: ast.FunctionDef | ast.AsyncFunctionDef, summary: str | None) -> str:
    """The function with its docstring replaced by the one-line summary (or removed)."""
    clone = ast.parse(ast.unparse(node)).body[0]
    has_doc = (clone.body and isinstance(clone.body[0], ast.Expr)
               and isinstance(getattr(clone.body[0], "value", None), ast.Constant)
               and isinstance(clone.body[0].value.value, str))
    body = clone.body[1:] if has_doc else clone.body
    clone.body = ([ast.Expr(ast.Constant(summary))] if summary else []) + (body or [ast.Pass()])
    return ast.unparse(clone)


def _stub(node: ast.FunctionDef | ast.AsyncFunctionDef, doc: str) -> str:
    clone = ast.parse(ast.unparse(node)).body[0]
    clone.body = [ast.Expr(ast.Constant(doc))]
    clone.decorator_list = []
    return ast.unparse(clone)


def _class_skeleton(node: ast.ClassDef) -> str | None:
    lines = [ast.unparse(node).splitlines()[0]]
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            header = ast.unparse(item).splitlines()
            line = next((h for h in header if h.lstrip().startswith(("def ", "async def "))), None)
            if line:
                lines.append("    " + line.strip() + "\n        ...")
    return "\n".join(lines) if len(lines) > 1 else None


def extract_python_examples(source: str, rng: random.Random, *, max_chars: int = 1500) -> list[InstructionExample]:
    """Turn real code into (instruction, answer) pairs.

    Tasks: write a function from its description, explain a function, complete a
    function from its signature + docstring, write a docstring, find and fix an
    injected bug, and explain a class. Answers always come from the real code."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return []
    examples: list[InstructionExample] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            doc = ast.get_docstring(node)
            summary = _docstring_summary(doc) if doc else None
            try:
                skeleton = _class_skeleton(node) if summary else None
            except (ValueError, RecursionError):
                skeleton = None
            if summary and skeleton and len(skeleton) <= max_chars:
                examples.append(InstructionExample(
                    rng.choice(_PROMPTS_CLASS).format(name=node.name, code=skeleton),
                    f"The `{node.name}` class: {summary}",
                ))
            continue
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name.startswith("_"):
            continue
        doc = ast.get_docstring(node)
        summary = _docstring_summary(doc) if doc else None
        try:
            full = ast.unparse(node)
            bare = _strip_docstring(node)
            args = ast.unparse(node.args)
        except (ValueError, RecursionError):
            continue
        n_lines = len(bare.splitlines())
        if len(full) > max_chars or n_lines < 2:
            continue

        # Bug fixing works for undocumented functions too.
        if 3 <= n_lines <= 40 and rng.random() < 0.6:
            try:
                compact = _with_short_doc(node, summary)
            except (ValueError, RecursionError):
                compact = None
            bug = inject_bug(compact, rng) if compact else None
            if bug is not None:
                buggy, bad, good = bug
                examples.append(InstructionExample(
                    rng.choice(_PROMPTS_BUGFIX).format(code=buggy),
                    f"The bug is in `{bad}`. It should be `{good}`.\n\n```python\n{compact}\n```",
                ))
        if summary is None:
            continue
        task = _as_task(summary)
        imp = summary[0].lower() + summary[1:] if summary[1:2].islower() else summary
        imp = imp if imp.endswith((".", "?", "!")) else imp + "."
        # "Write"/"complete" answers must be self-contained functions in a compact style:
        # no methods (their bodies depend on `self`), one-line docstring, short bodies.
        first_arg = node.args.args[0].arg if node.args.args else ""
        standalone = first_arg not in ("self", "cls") and n_lines <= 25
        compact = None
        if standalone:
            try:
                compact = _with_short_doc(node, summary)
            except (ValueError, RecursionError):
                compact = None
        if compact is not None:
            sig = f"{node.name}({args})"
            examples.append(InstructionExample(
                rng.choice(_PROMPTS_WRITE).format(sig=sig, name=node.name, task=task, imp=imp),
                f"```python\n{compact}\n```",
            ))
        examples.append(InstructionExample(
            rng.choice(_PROMPTS_EXPLAIN).format(code=bare),
            f"The function `{node.name}` {task}",
        ))
        if compact is not None and rng.random() < 0.5:
            try:
                stub = _stub(node, summary)
            except (ValueError, RecursionError):
                stub = None
            if stub:
                examples.append(InstructionExample(
                    rng.choice(_PROMPTS_COMPLETE).format(stub=stub), f"```python\n{compact}\n```"))
        if rng.random() < 0.3:
            examples.append(InstructionExample(
                rng.choice(_PROMPTS_DOCSTRING).format(code=bare), f'"""{summary}"""'))
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

_worker_tok: BPETokenizer | None = None


def _init_worker(tok_dict: dict) -> None:
    global _worker_tok
    _worker_tok = BPETokenizer.from_dict(tok_dict)


def _encode_doc(text: str) -> list[int]:
    assert _worker_tok is not None
    return _worker_tok.encode(text, allow_special=False)


def parallel_encode(tok: BPETokenizer, texts: list[str], workers: int | None = None) -> list[list[int]]:
    """Encode documents on all CPU cores (the tokenizer is pure Python)."""
    import multiprocessing as mp
    import os

    workers = workers if workers is not None else (os.cpu_count() or 1)
    if workers <= 1 or len(texts) < 8:
        return [tok.encode(t, allow_special=False) for t in texts]
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(tok.to_dict(),)) as pool:
        return pool.map(_encode_doc, texts, chunksize=max(1, len(texts) // (workers * 8)))


def encode_example(tok: BPETokenizer, ex: InstructionExample) -> tuple[list[int], list[int]]:
    """Tokens plus a loss mask that is 1 only on the answer (and its end marker)."""
    prompt_ids = tok.encode(format_chat(ex.prompt))
    answer_ids = tok.encode(ex.response.strip(), allow_special=False) + tok.encode("\n" + END)
    return prompt_ids + answer_ids, [0] * len(prompt_ids) + [1] * len(answer_ids)


def collect_examples(files: list[tuple[Path, str]], extra_sft: list[Path] | None,
                     rng: random.Random) -> list[InstructionExample]:
    examples: list[InstructionExample] = []
    for path, text in files:
        if path.suffix == ".py":
            examples.extend(extract_python_examples(text, rng))
    for path in extra_sft or []:
        examples.extend(load_jsonl_examples(path))
    unique: dict[str, InstructionExample] = {}
    for ex in examples:
        unique.setdefault(ex.prompt, ex)
    examples = list(unique.values())
    rng.shuffle(examples)
    return examples


def write_sft_files(tok: BPETokenizer, examples: list[InstructionExample], out_dir: Path) -> int:
    """Encode instruction examples to sft_tokens.bin / sft_mask.bin / sft_index.npy."""
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
    return len(tokens)


def prepare_sft_dataset(sources: list[Path], out_dir: Path, tokenizer_path: Path, *,
                        extra_sft: list[Path] | None = None, exclude: Iterable[str] = (), seed: int = 1337,
                        log: Log = print) -> dict:
    """Rebuild only the instruction data, with an existing tokenizer.

    Lets you improve SFT data and re-tune an already pretrained model, which must
    keep the tokenizer it was trained with."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = BPETokenizer.load(tokenizer_path)
    tok.save(out_dir / "tokenizer.json")
    files = read_code_files(sources, exclude=exclude)
    if not files:
        raise ValueError("No source files found.")
    rng.shuffle(files)
    examples = collect_examples(files, extra_sft, rng)
    n_tokens = write_sft_files(tok, examples, out_dir)
    meta = {"vocab_size": tok.vocab_size, "files": len(files), "sft_examples": len(examples),
            "sft_tokens": n_tokens, "sft_only": True, "sources": [str(s) for s in sources],
            "exclude": list(exclude)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"  {len(examples)} instruction examples, {n_tokens:,} tokens -> {out_dir}")
    return meta


def prepare_dataset(
    sources: list[Path],
    out_dir: Path,
    *,
    vocab_size: int = 4096,
    val_fraction: float = 0.02,
    extra_sft: list[Path] | None = None,
    exclude: Iterable[str] = (),
    workers: int | None = None,
    tokenizer_sample_chars: int = 8_000_000,
    seed: int = 1337,
    log: Log = print,
) -> dict:
    if not 300 <= vocab_size <= 65535:
        raise ValueError("vocab_size must be between 300 and 65535")
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Collecting code from {', '.join(str(s) for s in sources)} ...")
    files = read_code_files(sources, exclude=exclude)
    if not files:
        raise ValueError("No source files found. Pass --source with directories that contain code.")
    rng.shuffle(files)
    total_chars = sum(len(t) for _, t in files)
    log(f"  {len(files)} files, {total_chars / 1e6:.1f} MB of code")

    examples = collect_examples(files, extra_sft, rng)
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
        encoded = parallel_encode(tok, [text for _, text in items], workers)
        ids = np.fromiter((i for doc in encoded for i in (*doc, tok.eot_id)), dtype=np.uint16)
        ids.tofile(out_dir / f"{split}.bin")
        counts[split] = len(ids)
    log(f"  train {counts['train']:,} tokens, val {counts['val']:,} tokens")

    tokens = write_sft_files(tok, examples, out_dir)

    meta = {
        "vocab_size": tok.vocab_size,
        "files": len(files),
        "chars": total_chars,
        "train_tokens": counts["train"],
        "val_tokens": counts["val"],
        "sft_examples": len(examples),
        "sft_tokens": tokens,
        "sources": [str(s) for s in sources],
        "exclude": list(exclude),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"Wrote dataset to {out_dir}")
    return meta
