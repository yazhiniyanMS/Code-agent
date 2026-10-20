"""V8 stage 3: compositionally structured programs, ``aggregate(transform(selection(numbers)))``.

A small grammar of list primitives, written for V8, composes into many distinct programs whose task
text is composed the same way ("returns the largest of the squares of the numbers after the first
two"). Each combination is a ycode.lm.idioms.Program (family "comp:<aggregate>"), so the curriculum's
verification, held-out checks, prompt templates and renaming apply unchanged.

``composition_programs`` keeps one combination per behaviour (on 40 seeded inputs), drops constant
ones, ones that compute a function round 54 was trained on (the v5 catalogue) and ones that share an
AST skeleton or behaviour with a held-out item, the V7 neighbour set or a V8 cross-family test
program. ``composition_split`` holds out combinations whose primitives all occur in training (a
systematic-generalisation test: new combinations of known parts).
"""
from __future__ import annotations

import random

from ycode.lm.idioms import Program, _run, make_inputs

# (key, noun phrase completing "the ... ", expression for the selected list)
SELECT = (
    ("all", "numbers in the list", "numbers"),
    ("pos", "positive numbers in the list", "[x for x in numbers if x > 0]"),
    ("even", "even numbers in the list", "[x for x in numbers if x % 2 == 0]"),
    ("odd", "odd numbers in the list", "[x for x in numbers if x % 2 != 0]"),
    ("big", "numbers in the list that are greater than 5", "[x for x in numbers if x > 5]"),
    ("after2", "numbers that come after the first two in the list", "numbers[2:]"),
    ("first3", "first three numbers of the list", "numbers[:3]"),
    ("evenpos", "numbers at even positions of the list (0, 2, 4, ...)", "numbers[::2]"),
    ("rev", "numbers of the list in reverse order", "numbers[::-1]"),
    ("sorted", "numbers of the list from smallest to largest", "sorted(numbers)"),
    ("distinct", "different numbers in the list, each once, in order of first appearance", "list(dict.fromkeys(numbers))"),
)
# (key, phrase around the selection, element expression in x)
TRANSFORM = (
    ("id", "{sel}", "x"),
    ("sq", "squares of the {sel}", "x * x"),
    ("dbl", "doubles of the {sel}", "2 * x"),
    ("abs", "absolute values of the {sel}", "abs(x)"),
    ("neg", "negatives of the {sel}", "-x"),
    ("inc", "{sel}, each increased by one", "x + 1"),
    ("mod3", "remainders of the {sel} divided by 3", "x % 3"),
)
# (key, task template with {t}, comprehension implementation, loop implementation) over ys
AGGREGATE = (
    ("sum", "returns the sum of the {t}.", "return sum(ys)",
     "total = 0\nfor y in ys:\n    total += y\nreturn total"),
    ("max", "returns the largest of the {t} (None if there are none).", "return max(ys) if ys else None",
     "best = None\nfor y in ys:\n    if best is None or y > best:\n        best = y\nreturn best"),
    ("min", "returns the smallest of the {t} (None if there are none).", "return min(ys) if ys else None",
     "best = None\nfor y in ys:\n    if best is None or y < best:\n        best = y\nreturn best"),
    ("prod", "returns the product of the {t} (1 if there are none).", "result = 1\nfor y in ys:\n    result *= y\nreturn result",
     "result = 1\nfor y in ys:\n    result = result * y\nreturn result"),
    ("list", "returns a list of the {t}.", "return ys", "result = []\nfor y in ys:\n    result.append(y)\nreturn result"),
    ("first", "returns the first of the {t} (None if there are none).", "return ys[0] if ys else None",
     "for y in ys:\n    return y\nreturn None"),
    ("last", "returns the last of the {t} (None if there are none).", "return ys[-1] if ys else None",
     "result = None\nfor y in ys:\n    result = y\nreturn result"),
    ("ndistinct", "returns how many different values there are among the {t}.", "return len(set(ys))",
     "seen = []\nfor y in ys:\n    if y not in seen:\n        seen.append(y)\nreturn len(seen)"),
    ("count", "returns how many of the {t} are greater than 3.", "return sum(1 for y in ys if y > 3)",
     "count = 0\nfor y in ys:\n    if y > 3:\n        count += 1\nreturn count"),
)


def _name(a: str, t: str, s: str) -> str:
    parts = [a] + ([] if t == "id" else [t]) + ([] if s == "all" else [s])
    return "_".join(parts if len(parts) > 1 else parts + ["all"])


PARTS: dict[str, tuple[str, str, str]] = {}  # program name -> (selection, transform, aggregate) keys


def _program(sel, tr, agg, inputs) -> Program:
    skey, sphrase, sexpr = sel
    tkey, tphrase, texpr = tr
    akey, atask, comp, loop = agg
    selected = sexpr if tkey == "id" else f"[{texpr} for x in {sexpr}]"
    ys = f"ys = {selected}"
    task = atask.format(t=tphrase.format(sel=sphrase))
    bodies = (f"{ys}\n{comp}", f"{ys}\n{loop}")
    name = _name(akey, tkey, skey)
    PARTS[name] = (skey, tkey, akey)
    return Program(f"comp:{akey}", name, "numbers", (task,), bodies, 2, tuple(inputs))


def all_combinations(seed: int = 0) -> list[Program]:
    """Every combination, verified by execution (both implementations agree, output not constant),
    one per behaviour."""
    rng = random.Random(seed)
    inputs = make_inputs(("ints",), rng)
    seen, out = set(), []
    for agg in AGGREGATE:
        for tr in TRANSFORM:
            for sel in SELECT:
                p = _program(sel, tr, agg, inputs)
                results = [_run(p.source(i), p.name, list(inputs)) for i in range(len(p.bodies))]
                if any(r is None for r in results) or repr(results[0]) != repr(results[1]):
                    raise AssertionError(f"{p.name}: implementations disagree or fail")
                key = repr(results[0])
                if len({repr(o) for o in results[0]}) < 2 or key in seen:
                    continue
                seen.add(key)
                out.append(p)
    return out


def composition_programs(seed: int = 0) -> tuple[list[Program], dict[str, int]]:
    """``all_combinations`` minus held-out and already-trained behaviours: (kept, drop counts)."""
    import json
    from pathlib import Path

    from ycode.lm import idioms
    from ycode.lm.leakage import skeleton

    combos = all_combinations(seed)
    kept, dropped = idioms.held_out_check(combos)
    counts = {"held_out": len(combos) - len(kept)}
    root = Path(__file__).resolve().parents[2]
    avoid = idioms.neighbour_skeletons()
    cross = root / "experiments" / "v8" / "sets" / "t_families.json"
    cross_refs = [t["reference"] for t in json.loads(cross.read_text())["tasks"]] if cross.exists() else []
    avoid |= {s for s in map(skeleton, cross_refs) if s}
    out = []
    counts["test_skeleton"] = counts["trained_v5"] = 0
    trained = _v5_behaviours(kept)
    for p in kept:
        if any(skeleton(p.source(i)) in avoid for i in range(len(p.bodies))):
            counts["test_skeleton"] += 1
        elif p.name in trained:
            counts["trained_v5"] += 1
        else:
            out.append(p)
    return out, counts


def _v5_behaviours(programs: list[Program]) -> set[str]:
    """Names of programs that compute (on their inputs) a function of the v5 training catalogue."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "v7"))
    from mechanism import catalogue  # noqa: E402

    one_arg = [c for c in catalogue().values() if c["arity"] == 1]
    hits = set()
    inputs = list(programs[0].inputs) if programs else []
    want = {p.name: repr(_run(p.source(0), p.name, inputs)) for p in programs}
    for c in one_arg:
        got = _run(c["source"], c["name"], inputs)
        if got is None:
            continue
        g = repr(got)
        hits |= {n for n, w in want.items() if w == g}
    return hits


def composition_split(programs: list[Program], held_out: int = 60, more: int = 60,
                      seed: int = 8) -> dict[str, list[Program]]:
    """train (COMP's programs), more (MORE's: a subset of train), test (held-out combinations whose
    selection, transform and aggregate each occur in train)."""
    rng = random.Random(seed)
    order = list(programs)
    rng.shuffle(order)
    test, train = [], []
    for p in order:
        if len(test) < held_out:
            rest = [q for q in order if q is not p and q not in test]
            parts = _parts(p)
            if all(any(part in _parts(q) for q in rest) for part in parts):
                test.append(p)
                continue
        train.append(p)
    return {"train": train, "more": train[:more], "test": test}


def _parts(p: Program) -> set[str]:
    """The program's selection, transform and aggregate."""
    sel, tr, agg = PARTS[p.name]
    return {f"s:{sel}", f"t:{tr}", f"a:{agg}"}
