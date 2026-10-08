"""Answering questions with YCode's own model, made more reliable without any API.

The bundled models are small and were trained on little compute, so their raw answers are
often wrong. This layer improves what reaches the user:

* **Documentation questions** ("what does len() do?", "how do I use json.loads?", "what is
  yield?") are answered from Python's own built-in documentation, which ships with every
  Python install. These answers are exact.
* **"Write a function f(...)" requests** sample several candidates and return the first that
  compiles and defines the requested function. Candidates are only parsed, never run.
* Everything else goes to the model as a normal chat turn.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import keyword
import re
import sys
from typing import Callable

from ycode.lm.evaluate import extract_code

DOC_SOURCE_NOTE = "_(From Python's built-in documentation.)_"
UNVERIFIED_NOTE = ("_(YCode's local model is small and its code is often wrong: check this before "
                   "using it.)_")
CHECKED_NOTE = ("_(Written by YCode's local model. The function name and syntax were checked, not "
                "its logic: test it before relying on it.)_")
_MAX_DOC_CHARS = 1500

_WRITE_WORDS = re.compile(r"\b(write|implement|create|make|build|code|generate|define)\b", re.I)
_ASK_WORDS = re.compile(r"\b(what|whats|what's|how|explain|describe|meaning|mean|means|usage|use|"
                        r"does|do|is|are|purpose|help|doc|docs|documentation)\b", re.I)
_BACKTICKED = re.compile(r"`([A-Za-z_][\w.]*)(?:\([^`]*\))?`")
_CALLED = re.compile(r"\b([A-Za-z_][\w]*(?:\.[A-Za-z_]\w*)*)\(\)")
_DOTTED = re.compile(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\b")
_AFTER_ASK = re.compile(r"\b(?:what\s+is|what's|whats|what\s+are|what\s+does|explain|describe|"
                        r"how\s+(?:do\s+i|to|does)\s+(?:use\s+)?)\s+(?:the\s+|a\s+|an\s+|python'?s?\s+)*"
                        r"`?([A-Za-z_][\w.]*)", re.I)
_FUNCTION_NAME = re.compile(
    r"(?:\bfunction\s+(?:called\s+|named\s+)?`?|\bdef\s+|`)([A-Za-z_]\w*)\s*\(", re.I)
_DIFFERENCE = re.compile(r"\bdifference\s+between\s+(?:a\s+|an\s+|the\s+)?`?([A-Za-z_][\w.]*)`?(?:\(\))?\s+"
                         r"and\s+(?:a\s+|an\s+|the\s+)?`?([A-Za-z_][\w.]*)", re.I)
_STOPWORDS = {"a", "an", "the", "in", "python", "it", "this", "that", "i", "use", "to", "do", "does",
              "difference", "best", "way", "my", "code", "function", "method", "module", "class"}


# ------------------------------------------------------------------ docs


def _resolve(name: str):
    """A Python object for ``name`` from builtins or the standard library, else None.

    Only builtins and standard-library modules are imported, never the user's own code."""
    parts = name.split(".")
    if parts[0] in vars(builtins):
        obj = vars(builtins)[parts[0]]
        rest = parts[1:]
    else:
        obj, rest = None, []
        for cut in range(len(parts), 0, -1):
            module = ".".join(parts[:cut])
            if parts[0] not in sys.stdlib_module_names or parts[0].startswith("_"):
                return None
            try:
                obj = importlib.import_module(module)
            except Exception:  # noqa: BLE001 - not a module at this depth; try a shorter prefix
                continue
            rest = parts[cut:]
            break
        if obj is None:
            return None
    for attr in rest:
        if attr.startswith("_"):
            return None
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            return None
    return obj


def _keyword_doc(word: str) -> str | None:
    """Reference-manual text for a keyword or statement (``for``, ``yield``, ``with``...)."""
    if not keyword.iskeyword(word):
        return None
    try:
        import pydoc
        from pydoc_data.topics import topics
    except ImportError:
        return None
    topic = pydoc.Helper.keywords.get(word)
    topic = topic[0] if isinstance(topic, tuple) else topic
    if isinstance(topic, str):
        topic = topic.split()[0] if topic else None
    text = topics.get(topic or "") if topic else None
    return text.strip() if text else None


def _trim(text: str) -> str:
    text = text.strip()
    if len(text) <= _MAX_DOC_CHARS:
        return text
    cut = text.rfind("\n\n", 0, _MAX_DOC_CHARS)
    return text[: cut if cut > 200 else _MAX_DOC_CHARS].rstrip() + "\n\n..."


def _block(text: str) -> str:
    """Show documentation verbatim: its `>>>` examples would otherwise render as quotes."""
    fence = "`" * 3
    return f"{fence}text\n{_trim(text).replace(fence, '~~~')}\n{fence}"


def _describe(name: str, obj) -> str | None:
    doc = inspect.getdoc(obj)
    if not doc:
        return None
    kind = ("module" if inspect.ismodule(obj) else "class" if inspect.isclass(obj)
            else "built-in function" if inspect.isbuiltin(obj) else
            "function" if callable(obj) else type(obj).__name__)
    title = name
    if callable(obj) and not inspect.ismodule(obj):
        try:
            title = f"{name}{inspect.signature(obj)}"
        except (TypeError, ValueError):
            first = doc.splitlines()[0]
            if first.startswith(name.split(".")[-1] + "("):  # e.g. "zip(*iterables, strict=False) --> ..."
                prefix = name.rsplit(".", 1)[0] + "." if "." in name else ""
                signature, _, summary = first.partition(" -->")
                title = prefix + signature
                doc = "\n".join([summary.strip(), *doc.splitlines()[1:]]).strip() or doc
    return f"**`{title}`** ({kind})\n\n{_block(doc)}"


def _doc_for(name: str) -> str | None:
    """The documentation entry for a keyword, builtin or standard-library name, if any."""
    kw = _keyword_doc(name)
    if kw:
        return f"**`{name}`** (Python keyword)\n\n{_block(kw)}"
    obj = _resolve(name)
    return _describe(name, obj) if obj is not None else None


def _candidate_names(question: str) -> list[str]:
    names = _BACKTICKED.findall(question) + _CALLED.findall(question) + _DOTTED.findall(question)
    names += _AFTER_ASK.findall(question)
    seen, out = set(), []
    for name in names:
        name = name.strip(".")
        if name and name.lower() not in _STOPWORDS and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def docs_answer(question: str) -> str | None:
    """Answer "what does X do / how do I use X" from Python's documentation, if X is known."""
    if _WRITE_WORDS.search(question) or not _ASK_WORDS.search(question):
        return None
    pair = _DIFFERENCE.search(question)
    if pair:
        docs = [_doc_for(name.strip(".")) for name in pair.groups()]
        if all(docs):
            return "\n\n---\n\n".join(docs) + f"\n\n{DOC_SOURCE_NOTE}"
    for name in _candidate_names(question):
        doc = _doc_for(name)
        if doc:
            return f"{doc}\n\n{DOC_SOURCE_NOTE}"
    return None


# ------------------------------------------------------------- functions


def requested_function(question: str) -> str | None:
    """The function name a "write a function f(...)" request asks for, if any."""
    if not _WRITE_WORDS.search(question):
        return None
    match = _FUNCTION_NAME.search(question)
    if not match:
        return None
    name = match.group(1)
    return None if keyword.iskeyword(name) or name in vars(builtins) else name


def defines(code: str, name: str) -> bool:
    """True if ``code`` parses and defines a function called ``name`` (it is never executed)."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
               for node in ast.walk(tree))


# ---------------------------------------------------------------- answer


class Assistant:
    """Answers one turn with the docs, a checked function, or a plain model reply."""

    def __init__(self, lm, *, max_new_tokens: int = 400, temperature: float = 0.7, candidates: int = 4) -> None:
        self.lm = lm
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.candidates = max(1, candidates)

    def answer(self, turns: list[tuple[str, str]], on_text: Callable[[str], None] | None = None) -> str:
        emit = on_text or (lambda _text: None)
        question = turns[-1][1] if turns and turns[-1][0] == "user" else ""

        doc = docs_answer(question)
        if doc:
            emit(doc)
            return doc

        name = requested_function(question)
        if name:
            return self._write_function(turns, name, emit)

        return self.lm.chat(turns, max_new_tokens=self.max_new_tokens, temperature=self.temperature,
                            on_text=on_text)

    def _write_function(self, turns, name: str, emit: Callable[[str], None]) -> str:
        first = None
        for i in range(self.candidates):
            # The first try is nearly greedy (the model's best guess); later ones sample for variety.
            temperature = 0.2 if i == 0 else self.temperature
            reply = self.lm.chat(turns, max_new_tokens=self.max_new_tokens, temperature=temperature)
            first = first if first is not None else reply
            code = extract_code(reply)
            if defines(code, name):
                answer = f"```python\n{code}\n```\n\n{CHECKED_NOTE}"
                emit(answer)
                return answer
        answer = f"{(first or '').strip()}\n\n{UNVERIFIED_NOTE}".strip()
        emit(answer)
        return answer
