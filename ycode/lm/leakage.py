"""Behavioural leakage check: does any training answer compute a held-out evaluation item?

Name filters (``held_out_names``, ``HELD_OUT_KEYS``, ``DEV_KEYS``) keep benchmark and dev items out
of training by name and concept key. They cannot see a training function that computes the same
thing under another name (``count_occurrences`` works as the benchmark's ``char_count``).

A training function is checked against every benchmark and dev item with the same number of
parameters, bound to the item's name:

1. it must pass the item's own tests, and
2. it must agree with the item's reference solution on probe inputs (the test's own arguments plus
   typed variations of them).

Step 2 matters: the items' tests are short (``fibonacci`` is checked at 0, 1 and 10, so
``n * (n + 1) // 2`` passes them), and passing them alone says little. Functions that agree with
the reference on every probe are reported as equivalent (a leak); those agreeing on most probes
are reported as near-equivalent, for review.
"""

from __future__ import annotations

import ast
import itertools
import json
import re
from dataclasses import dataclass

from ycode.lm.evaluate import extract_code

_SIG = re.compile(r"`([A-Za-z_]\w*)\(([^)`]*)\)`")
MIN_VALID_PROBES = 8  # fewer probes inside the reference's domain than this: no verdict (agreement 0)

# Typed probe pools: each argument of a test call is varied over the pool of its type.
_POOLS = {
    int: [-7, -2, -1, 0, 1, 2, 3, 5, 8, 10, 13, 25, 100],
    float: [-1.5, 0.0, 0.5, 2.25, 10.0, 37.5],
    bool: [True, False],
    str: ["", "a", "abc", "Hello World", "banana", "racecar", "a b  c", "The quick brown fox", "x.y.z",
          "AEIOU aeiou", "12321", "Mississippi", "hello world.", " padded "],
    "list_int": [[], [0], [3], [1, 2, 3], [5, -2, 7, 0], [4, 1, 9, 9, 7], [2, 2, 2], [-3, -1, -4],
                 [10, 20, 30, 40], [1, 2, 3, 4, 5, 6]],
    "list_str": [[], ["a"], ["ab", "c", "def"], ["x", "y", "x"], ["Hello", "world"]],
    "list_list": [[], [[1], [2, 3]], [[], [4]], [[1, 2], [3], [4, 5, 6]]],
    "dict": [{}, {"a": 1}, {"a": 1, "b": 2}, {"x": 0, "y": -1}, {0: "a", "": 1, None: 2}, {False: 0}],
    "tuple": [(), (1,), (1, 2), (3, -1, 4)],
}


# Training functions that compute a held-out item with a shared building block, for a different task:
# reviewed and kept (removing them would remove len(), a constant multiple, a product). Pairs of
# (training function name, held-out item name). Same-task equivalents are removed from the
# generators instead (compose.EQUIVALENT_KEYS, basics.BEHAVIOUR_EXCLUDED).
SHARED_OPERATIONS = frozenset({
    ("count_items", "dict_size"), ("string_length", "dict_size"), ("size", "dict_size"),  # len()
    ("multiply_by_seven", "weeks_to_days"), ("scale_by_seven", "weeks_to_days"),  # 7 * x: the
    ("times_seven", "weeks_to_days"),  # fact that a week has 7 days is not taught by them
    ("simple_interest", "multiply_three_numbers"),  # principal * rate * years
    ("longer_than_three_chars", "longer_than_three"),  # len(s) > 3 on a string vs a list
})


def normalized_body(source: str) -> str:
    """A function's body without its def line and docstring, with its own name replaced by F (so
    renamed copies and documented copies of one function compare equal)."""
    lines = source.strip().splitlines()
    if not lines:
        return ""
    m = re.match(r"\s*def (\w+)\(", lines[0])
    body = "\n".join(line for line in lines[1:] if not line.strip().startswith(('"""', "'''"))).strip()
    return re.sub(rf"\b{re.escape(m.group(1))}\b", "F", body) if m else body


_SHARED: dict[str, set[str]] | None = None


def _shared_bodies() -> dict[str, set[str]]:
    """Held-out item -> normalised bodies of the reviewed shared-operation functions (and of their
    loop renderings)."""
    global _SHARED
    if _SHARED is None:
        import ycode.lm.compose as compose
        from ycode.lm.basics import CONCEPTS

        sources: dict[str, list[str]] = {}
        for c in CONCEPTS:
            for n in (c.name, *c.aliases):
                sources.setdefault(n, []).append(c.source(n))
        for e in compose._raw_exercises(7) + compose.skill_exercises():
            sources.setdefault(e.name, []).append(e.source())
            if (loop := compose.loop_rendering(e)) is not None:
                sources[e.name].append(loop.source())
        _SHARED = {}
        for function, item in SHARED_OPERATIONS:
            _SHARED.setdefault(item, set()).update(normalized_body(src) for src in sources.get(function, []))
    return _SHARED


def is_shared_operation(source: str, item: str) -> bool:
    """True if ``source`` (renamed or not) is one of the reviewed SHARED_OPERATIONS for ``item``."""
    return normalized_body(source) in _shared_bodies().get(item, set())


@dataclass(frozen=True)
class HeldOutItem:
    split: str  # "bench" or "dev"
    name: str
    arity: int
    tests: str
    reference: str  # source of a correct solution named ``name``


@dataclass(frozen=True)
class Leak:
    function: str
    item: str
    split: str
    agreement: float  # share of probes on which the function matches the reference
    probes: int
    source: str


def _arity(params: str) -> int:
    params = params.strip()
    return 0 if not params else len([p for p in params.split(",") if p.strip()])


def held_out_items() -> list[HeldOutItem]:
    """Every benchmark write problem (standard 30, fresh 20; the 15 benchmark bug fixes reuse the
    standard functions) and every dev item, with its reference solution."""
    from ycode.lm.basics import CONCEPTS
    from ycode.lm.compose import _raw_exercises, base_key
    from ycode.lm.devset import DEV_KEYS, dev_problems
    from ycode.lm.evaluate import PROBLEMS, fresh_problems
    from ycode.lm.references import FRESH_REFERENCE, REFERENCE

    refs = {**REFERENCE, **FRESH_REFERENCE}
    dev_refs = {c.name: c.source() for c in CONCEPTS}
    for ex in _raw_exercises(seed=0):  # the same rendering devset.dev_problems uses
        if base_key(ex.key) in DEV_KEYS:
            dev_refs.setdefault(ex.name, ex.source())
    items = []
    for split, problems, sources in (("bench", PROBLEMS + fresh_problems(), refs), ("dev", dev_problems(), dev_refs)):
        for p in problems:
            m = _SIG.search(p.prompt)
            if m and p.name in sources:
                items.append(HeldOutItem(split, p.name, _arity(m.group(2)), p.tests, sources[p.name]))
    return items


def answer_functions(responses) -> dict[str, tuple[str, int]]:
    """Distinct functions written in answers: {source: (name, arity)}. Answers whose code does not
    parse, or does not start with a plain function, are skipped."""
    out: dict[str, tuple[str, int]] = {}
    for response in responses:
        code = extract_code(response)
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError, RecursionError):
            continue
        fn = tree.body[0] if tree.body else None
        if not isinstance(fn, ast.FunctionDef) or fn.args.vararg or fn.args.kwarg:
            continue
        out.setdefault(code, (fn.name, len(fn.args.posonlyargs) + len(fn.args.args)))
    return out


def _kind(value):
    if isinstance(value, bool):
        return bool
    if isinstance(value, (int, float, str)):
        return type(value)
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, list):
        if value and all(isinstance(v, list) for v in value):
            return "list_list"
        if value and all(isinstance(v, str) for v in value):
            return "list_str"
        return "list_int"
    return None


_WORDS = ("a", "an", "the", "cat", "zebra", "extraordinary", "is", "of", "Hello", "WORLD", "banana", "x",
          "level", "noon", "Python", "code", "aeiou", "rhythm", "42", "end.")


def _random_value(kind, rng):
    if kind is bool:
        return rng.random() < 0.5
    if kind is int:
        return rng.randint(-50, 200)
    if kind is float:
        return round(rng.uniform(-100, 100), 2)
    if kind is str:
        return " ".join(rng.choice(_WORDS) for _ in range(rng.randint(0, 5)))
    if kind == "list_int":
        return [rng.randint(-20, 50) for _ in range(rng.randint(0, 8))]
    if kind == "tuple":
        return tuple(rng.randint(-20, 50) for _ in range(rng.randint(0, 6)))
    if kind == "list_str":
        return [rng.choice(_WORDS) for _ in range(rng.randint(0, 5))]
    if kind == "list_list":
        return [[rng.randint(-5, 9) for _ in range(rng.randint(0, 3))] for _ in range(rng.randint(0, 4))]
    if kind == "dict":  # keys of any kind, falsy ones included: a dictionary's keys need not be words
        keys = list(_WORDS) + [0, 1, "", None, 2.5]
        return {rng.choice(keys): rng.randint(-5, 9) for _ in range(rng.randint(0, 4))}
    return None


def probe_inputs(item: HeldOutItem, limit: int = 120, random_probes: int = 60) -> list[tuple]:
    """Argument tuples for comparing a function with the item's reference: the arguments of the
    item's own test calls, typed variations of each argument position, and seeded random values of
    the same types."""
    import random

    calls: list[tuple] = []
    for node in ast.walk(ast.parse(item.tests)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == item.name and not node.keywords:
            try:
                args = tuple(ast.literal_eval(a) for a in node.args)
            except (ValueError, SyntaxError):
                continue
            if len(args) == item.arity and args not in calls:
                calls.append(args)
    if not calls:
        return []
    kinds = [sorted({_kind(c[pos]) for c in calls} - {None}, key=str) for pos in range(item.arity)]
    pools = []
    for pos in range(item.arity):
        values = [c[pos] for c in calls]
        for kind in kinds[pos]:
            pool = _POOLS.get(kind, [])
            if kind == "tuple":  # tuples of numbers: vary them like lists of numbers
                pool = pool + [tuple(v) for v in _POOLS["list_int"]]
            values += [v for v in pool if v not in values]
        pools.append(values)
    per_arg = max(2, int((limit - random_probes) ** (1 / max(1, item.arity)))) if item.arity else 1
    probes = list(calls)
    for args in itertools.product(*(p[:per_arg] for p in pools)):
        if args not in probes:
            probes.append(args)
    probes = probes[:max(len(calls), limit - random_probes)]
    rng = random.Random(item.name)
    for _ in range(random_probes * 3):
        if len(probes) >= limit or not all(kinds):
            break
        args = tuple(_random_value(rng.choice(k), rng) for k in kinds)
        if args not in probes:
            probes.append(args)
    return probes


_PROBE_PROGRAM = """
import copy, json, sys
{candidate}

{reference_renamed}

def _out(fn, args):
    try:
        return ("ok", repr(fn(*copy.deepcopy(args))))
    except Exception as exc:
        return ("err", type(exc).__name__)

probes = {probes!r}
same = valid = 0
for args in probes:
    a, b = _out({cand_name}, args), _out(__reference, args)
    if b[0] == "err":
        continue  # outside the reference's domain: says nothing
    valid += 1
    if a == b:
        same += 1
    elif a[0] == b[0] == "ok":
        try:
            same += abs(float(eval(a[1])) - float(eval(b[1]))) < 1e-9
        except Exception:
            pass
print(json.dumps([same, valid]))
"""


def agreement(source: str, name: str, item: HeldOutItem, probes: list[tuple], *, isolate: bool,
              timeout: float = 5.0) -> float:
    """Share of probes where the function returns what the reference returns. Probes on which the
    reference raises are skipped; with fewer than MIN_VALID_PROBES left the result is 0."""
    reference = re.sub(rf"\b{re.escape(item.name)}\b", "__reference", item.reference)
    program = _PROBE_PROGRAM.format(candidate=source, reference_renamed=reference, probes=probes, cand_name=name)
    if isolate:
        import subprocess
        import sys
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            try:
                proc = subprocess.run([sys.executable, "-I", "-c", program], cwd=tmp, capture_output=True,
                                      text=True, timeout=timeout, stdin=subprocess.DEVNULL)
            except subprocess.TimeoutExpired:
                return 0.0
        try:
            same, total = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return 0.0
        return same / total if total >= MIN_VALID_PROBES else 0.0
    import contextlib
    import io

    from ycode.lm.basics import _passes

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        if not _passes(program, "", timeout=timeout):
            return 0.0
    same, total = json.loads(buf.getvalue().strip().splitlines()[-1])
    return same / total if total >= MIN_VALID_PROBES else 0.0


def behaviour_leaks(functions: dict[str, tuple[str, int]], items: list[HeldOutItem] | None = None, *,
                    isolate: bool = False, timeout: float = 1.0, near: float = 0.8) -> list[Leak]:
    """Functions that pass an item's tests and agree with its reference on at least ``near`` of the
    probes (agreement 1.0 = equivalent). ``isolate`` runs each check in a subprocess (for code that is
    not our own, e.g. replayed library functions); otherwise in-process with a time limit."""
    from ycode.lm.basics import _passes
    from ycode.lm.evaluate import run_tests

    items = held_out_items() if items is None else items
    probes = {(i.split, i.name): probe_inputs(i) for i in items}
    leaks = []
    for source, (name, arity) in functions.items():
        for item in items:
            if item.arity != arity:
                continue
            program = f"{source}\n\n{item.name} = {name}" if name != item.name else source
            if isolate:
                passed = run_tests(program, item.tests, timeout=max(timeout, 2.0))
            else:
                passed = _passes(program, item.tests, timeout=timeout)
            if not passed:
                continue
            p = probes[(item.split, item.name)]
            score = agreement(source, name, item, p, isolate=isolate) if p else 0.0
            if score >= near:
                leaks.append(Leak(name, item.name, item.split, round(score, 3), len(p), source))
    return leaks

