"""Training-data hygiene: exact duplicates, contradictory answers, and held-out equivalents.

``audit_examples`` reports, and ``clean_examples`` removes, three kinds of problem rows:

* **exact duplicates**: the same (prompt, answer) pair more than once (the copies add weight, not
  information);
* **contradictions**: one task text (function name masked) answered by functions that compute
  different things. Different code that computes the same thing (a comprehension and its loop
  rendering) is a legitimate variation and is kept;
* **held-out equivalents**: an answer that computes a benchmark or dev item (``ycode.lm.leakage``).

Nothing is dropped silently: every removed row is returned with the reason.
"""

from __future__ import annotations

import ast
import random
import re
from collections import defaultdict
from dataclasses import dataclass

from ycode.lm.evaluate import extract_code

_NAME_IN_PROMPT = re.compile(r"`[A-Za-z_]\w*(?=[(`])")
_BUGFIX_PREFIXES = ("This function has a bug", "Fix the bug", "Why is this function wrong")


def task_key(prompt: str) -> str:
    """The prompt with the function name masked, so renamed copies of one task compare equal."""
    return _NAME_IN_PROMPT.sub("`F", prompt)


def is_bugfix(prompt: str) -> bool:
    return prompt.startswith(_BUGFIX_PREFIXES)


@dataclass(frozen=True)
class Removed:
    index: int
    reason: str  # "duplicate" | "contradiction" | "held-out equivalent"
    detail: str


# Argument values tried for a function whose parameter types are unknown, by parameter name.
_BY_NAME = {
    "numbers": "list_int", "items": "list_int", "values": "list_int", "nums": "list_int", "flags": "list_int",
    "words": "list_str", "strings": "list_str", "names": "list_str", "lists": "list_list", "pairs": "pairs",
    "s": "str", "text": "str", "word": "str", "string": "str", "sentence": "str", "prefix": "str",
    "suffix": "str", "c": "str", "ch": "str", "char": "str", "sep": "str", "d": "dict", "a_dict": "dict",
}


def _values(kind: str, rng: random.Random):
    if kind == "list_int":
        return [rng.randint(-20, 50) for _ in range(rng.randint(0, 7))]
    if kind == "list_str":
        return [rng.choice(("a", "bb", "ccc", "Hello", "x y", "")) for _ in range(rng.randint(0, 5))]
    if kind == "list_list":
        return [[rng.randint(-5, 9) for _ in range(rng.randint(0, 3))] for _ in range(rng.randint(0, 4))]
    if kind == "pairs":
        return [(rng.randint(-5, 9), rng.randint(-5, 9)) for _ in range(rng.randint(1, 4))]
    if kind == "str":
        return " ".join(rng.choice(("a", "the", "Cat", "zebra", "x.y", "42", "hello")) for _ in range(rng.randint(0, 4)))
    if kind == "dict":
        return {rng.choice("abcxyz"): rng.randint(-5, 9) for _ in range(rng.randint(0, 4))}
    return rng.choice((rng.randint(-20, 100), rng.randint(0, 12)))  # int


def _signature(code: str) -> tuple[str, list[str]] | None:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return None
    fn = tree.body[0] if tree.body else None
    if not isinstance(fn, ast.FunctionDef) or fn.args.vararg or fn.args.kwarg:
        return None
    return fn.name, [a.arg for a in fn.args.posonlyargs + fn.args.args]


def _outputs(code: str, name: str, probes: list[tuple]) -> list | None:
    """Results (or exception types) of ``code`` on each probe; None if the code cannot run."""
    import copy
    import signal

    namespace: dict = {}

    def on_alarm(_signum, _frame):
        raise TimeoutError

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, 2.0)
    try:
        exec(compile(code, "<answer>", "exec"), namespace)
        fn = namespace[name]
        out = []
        for args in probes:
            try:
                out.append(("ok", repr(fn(*copy.deepcopy(args)))))
            except TimeoutError:
                raise
            except Exception as exc:  # noqa: BLE001
                out.append(("err", type(exc).__name__))
        return out
    except BaseException as exc:  # noqa: BLE001 - definition failed or timed out
        if isinstance(exc, KeyboardInterrupt):
            raise
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def same_behaviour(code_a: str, code_b: str, n_probes: int = 40, seed: int = 0) -> bool | None:
    """True if two single-function answers return the same results on random inputs typed by their
    parameter names (where both return normally); None when it cannot be decided (different arity,
    code that does not run, or fewer than 5 inputs on which both return normally)."""
    sa, sb = _signature(code_a), _signature(code_b)
    if sa is None or sb is None or len(sa[1]) != len(sb[1]):
        return None
    rng = random.Random(seed)
    probes = [tuple(_values(_BY_NAME.get(p, "int"), rng) for p in sa[1]) for _ in range(n_probes)]
    out_a, out_b = _outputs(code_a, sa[0], probes), _outputs(code_b, sb[0], probes)
    if out_a is None or out_b is None:
        return None
    # Compare only where both run normally: on inputs outside the task's domain (max of an empty
    # list) one rendering may raise and another return None, and either is acceptable.
    pairs = [(a, b) for a, b in zip(out_a, out_b) if a[0] == b[0] == "ok"]
    if len(pairs) < 5:
        return None
    return all(a == b for a, b in pairs)


def find_duplicates(examples) -> list[Removed]:
    seen: set[tuple[str, str]] = set()
    out = []
    for i, ex in enumerate(examples):
        key = (ex.prompt, ex.response)
        if key in seen:
            out.append(Removed(i, "duplicate", ex.prompt[:80]))
        seen.add(key)
    return out


def find_contradictions(examples) -> list[Removed]:
    """Write-a-function rows whose task text another row answers with a function that computes
    something different. The behaviour seen first for a task is kept; rows that disagree with it
    are returned."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        if not is_bugfix(ex.prompt):
            groups[task_key(ex.prompt)].append(i)
    out = []
    for task, idx in groups.items():
        bodies: dict[str, list[int]] = defaultdict(list)
        for i in idx:
            code = extract_code(examples[i].response)
            sig = _signature(code)
            # compare bodies with the name normalised (renamed copies are the same answer)
            norm = re.sub(rf"\b{re.escape(sig[0])}\b", "F", code) if sig else code
            bodies[norm].append(i)
        if len(bodies) < 2:
            continue
        variants = list(bodies)
        first = variants[0]
        for other in variants[1:]:
            if same_behaviour(first, other) is False:
                out += [Removed(i, "contradiction", f"{task[:80]} | kept: {first.splitlines()[-1].strip()[:60]} | "
                                                   f"dropped: {other.splitlines()[-1].strip()[:60]}")
                        for i in bodies[other]]
    return out


def find_held_out_equivalents(examples) -> list[Removed]:
    """Rows whose answer computes a benchmark or dev item exactly (ycode.lm.leakage, agreement 1.0)."""
    from ycode.lm.leakage import answer_functions, behaviour_leaks, is_shared_operation

    functions = answer_functions(ex.response for ex in examples)
    leaked = {}
    for lk in behaviour_leaks(functions, near=1.0):
        if not is_shared_operation(lk.source, lk.item):  # reviewed building blocks stay
            leaked.setdefault(lk.source, f"{lk.split} {lk.item} <- {lk.function}")
    return [Removed(i, "held-out equivalent", leaked[code])
            for i, ex in enumerate(examples) if (code := extract_code(ex.response)) in leaked]


def clean_examples(examples, *, dedup: bool = True, contradictions: bool = True, held_out: bool = True):
    """(kept examples, removed rows with reasons). Each check is optional, for ablations."""
    removed: dict[int, Removed] = {}
    if held_out:
        removed.update((r.index, r) for r in find_held_out_equivalents(examples))
    if dedup:
        removed.update((r.index, r) for r in find_duplicates(examples) if r.index not in removed)
    if contradictions:
        removed.update((r.index, r) for r in find_contradictions(examples) if r.index not in removed)
    kept = [ex for i, ex in enumerate(examples) if i not in removed]
    return kept, sorted(removed.values(), key=lambda r: r.index)
