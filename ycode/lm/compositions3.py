"""V10 R2: grammar v3, cross-family compositions ``aggregate(view(source(input)))``.

Each program chains operations of three different V8 training families: a source turns the input into
a list of integers (the list itself, the digits of n, the lengths of the words of a text, the lengths
of the runs of equal neighbours, the alphabet positions of the letters of a string); a view reshapes
that list (neighbour differences, the larger or smaller of each neighbouring pair, the running
maximum, sorted, distinct, odd positions, leaders, all but the first); an aggregate summarises it,
including training-family algorithms (longest rising run, record count, best run sum). Every part
names its training family; none comes from a held-out family or a dev, unseen or benchmark item.

Hygiene is grammar v2's (``ycode.lm.compositions2``): held-out items by name, skeleton and behaviour;
functions round 54 was trained on; and any combination that shares a skeleton with, or computes on
its own probe inputs, a cross-family task, a program of a held-out family, a within-family test or a
neighbour task. The cross-family tasks some combination computes are recorded before training.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ycode.lm.idioms import Program, _run, make_inputs

ROOT = Path(__file__).resolve().parents[2]

# (key, parameter, input kind, phrase, code building xs (an expression, or lines), name words, source family)
SOURCE = (
    ("list", "numbers", "ints", "the list", "numbers", "", "map_filtered"),
    ("digits", "n", "nonneg", "the digits of n", "[int(c) for c in str(n)]", "digits", "count_digit"),
    ("wordlens", "text", "text", "the lengths of the words of text", "[len(w) for w in text.split()]", "word_lengths",
     "group_by_length"),
    ("runs", "numbers", "ints", "the lengths of the runs of equal neighbouring numbers in the list",
     "xs = []\nfor i, x in enumerate(numbers):\n    if i > 0 and x == numbers[i - 1]:\n        xs[-1] += 1\n"
     "    else:\n        xs.append(1)", "run_lengths", "longest_run"),
    ("letters", "s", "str", "the alphabet positions of the lower-case letters of s (a = 1, b = 2, ...)",
     "[ord(c) - 96 for c in s if c.islower()]", "letter_positions", "shift_letters"),
)
# (key, phrase around {src}, code from xs to ys (an expression, or lines), name words, source family)
VIEW = (
    ("all", "{src}", "xs", "", "map_filtered"),
    ("steps", "the differences between neighbouring values of {src} (each minus the one before it)",
     "[b - a for a, b in zip(xs, xs[1:])]", "steps", "neighbour_steps"),
    ("pairmax", "the larger value of each neighbouring pair of {src}", "[max(a, b) for a, b in zip(xs, xs[1:])]",
     "pair_maxima", "pairwise_extreme"),
    ("pairmin", "the smaller value of each neighbouring pair of {src}", "[min(a, b) for a, b in zip(xs, xs[1:])]",
     "pair_minima", "pairwise_extreme"),
    ("runmax", "the running largest values of {src} (at each position, the largest value so far)",
     "ys = []\nfor x in xs:\n    ys.append(x if not ys or x > ys[-1] else ys[-1])", "running_max", "running_extreme"),
    ("sorted", "{src} sorted from smallest to largest", "sorted(xs)", "sorted", "sort_numbers_by"),
    ("distinct", "the different values of {src}, each once, in order of first appearance", "list(dict.fromkeys(xs))",
     "distinct", "sorted_distinct"),
    ("oddpos", "the values at odd positions (1, 3, 5, ...) of {src}", "xs[1::2]", "odd_positions",
     "sum_every_kth_position"),
    ("leaders", "the values of {src} that are larger than every value after them",
     "[x for i, x in enumerate(xs) if all(x > y for y in xs[i + 1:])]", "leaders", "leaders"),
    ("after1", "the values of {src} after the first one", "xs[1:]", "after_first", "sum_after_first"),
)
# (key, task template with {t}, comprehension-style body, loop body, name words, source family) over ys
AGGREGATE = (
    ("sum", "returns the sum of {t}.", "return sum(ys)", "total = 0\nfor y in ys:\n    total += y\nreturn total",
     "sum", "sum_of_first"),
    ("max", "returns the largest of {t} (None if there are none).", "return max(ys) if ys else None",
     "best = None\nfor y in ys:\n    if best is None or y > best:\n        best = y\nreturn best", "max", "extreme_of_first"),
    ("min", "returns the smallest of {t} (None if there are none).", "return min(ys) if ys else None",
     "best = None\nfor y in ys:\n    if best is None or y < best:\n        best = y\nreturn best", "min", "extreme_of_first"),
    ("count", "returns how many values there are in {t}.", "return len(ys)",
     "count = 0\nfor y in ys:\n    count += 1\nreturn count", "count", "count_singletons"),
    ("list", "returns a list of {t}.", "return list(ys)", "result = []\nfor y in ys:\n    result.append(y)\nreturn result",
     "list", "map_filtered"),
    ("last", "returns the last of {t} (None if there are none).", "return ys[-1] if ys else None",
     "result = None\nfor y in ys:\n    result = y\nreturn result", "last", "last_position"),
    ("count_pos", "returns how many of {t} are positive.", "return sum(1 for y in ys if y > 0)",
     "count = 0\nfor y in ys:\n    if y > 0:\n        count += 1\nreturn count", "count_positive", "count_by_sign"),
    ("rising", "returns the length of the longest stretch of strictly increasing neighbouring values in {t} "
     "(0 if there are none).",
     "best = run = 0\nfor i, y in enumerate(ys):\n    run = run + 1 if i > 0 and y > ys[i - 1] else 1\n"
     "    best = max(best, run)\nreturn best",
     "best = 0\nrun = 0\nprev = None\nfor y in ys:\n    if prev is not None and y > prev:\n        run += 1\n"
     "    else:\n        run = 1\n    if run > best:\n        best = run\n    prev = y\nreturn best",
     "longest_rise", "longest_monotone_run"),
    ("records", "returns how many of {t} are larger than every value before them.",
     "return sum(1 for i, y in enumerate(ys) if all(y > z for z in ys[:i]))",
     "count = 0\nbest = None\nfor y in ys:\n    if best is None or y > best:\n        count += 1\n        best = y\n"
     "return count", "records", "count_records"),
    ("bestrun", "returns the largest sum of a stretch of one or more neighbouring values of {t} (None if there are "
     "none).",
     "if not ys:\n    return None\nbest = cur = ys[0]\nfor y in ys[1:]:\n    cur = max(y, cur + y)\n"
     "    best = max(best, cur)\nreturn best",
     "if not ys:\n    return None\nbest = None\nfor i in range(len(ys)):\n    total = 0\n    for y in ys[i:]:\n"
     "        total += y\n        if best is None or total > best:\n            best = total\nreturn best",
     "best_run_sum", "best_run_sum"),
)

PARTS: dict[str, tuple[str, str, str]] = {}  # program name -> (source, view, aggregate)


def _lines(code: str, var: str) -> str:
    return code if "\n" in code else f"{var} = {code}"


def _program(src, view, agg, inputs) -> Program:
    skey, param, _, sphrase, scode, sname, _ = src
    vkey, vphrase, vcode, vname, _ = view
    akey, atask, comp, loop, aname, _ = agg
    task = atask.format(t=vphrase.format(src=sphrase))
    prefix = f"{_lines(scode, 'xs')}\n{_lines(vcode, 'ys')}"
    bodies = (f"{prefix}\n{comp}", f"{prefix}\n{loop}")
    words = [w for w in (aname, vname, sname) if w]
    name = "_".join(words if len(words) > 1 else words + ["all"])
    PARTS[name] = (skey, vkey, akey)
    return Program(f"comp3:{akey}", name, param, (task,), bodies, 2, tuple(inputs))


def all_combinations(seed: int = 0) -> list[Program]:
    """Every combination, verified by execution (both implementations agree, output not constant),
    one per behaviour within each source."""
    rng = random.Random(seed)
    out = []
    for src in SOURCE:
        inputs = make_inputs((src[2],), rng)
        seen = set()
        for agg in AGGREGATE:
            for view in VIEW:
                p = _program(src, view, agg, inputs)
                results = [_run(p.source(i), p.name, list(inputs)) for i in range(len(p.bodies))]
                if any(r is None for r in results) or repr(results[0]) != repr(results[1]):
                    raise AssertionError(f"{p.name}: implementations disagree or fail")
                key = repr(results[0])
                if len({repr(o) for o in results[0]}) < 2 or key in seen:
                    continue
                seen.add(key)
                out.append(p)
    return out


def grammar_programs(seed: int = 0) -> tuple[list[Program], dict]:
    """``all_combinations`` minus held-out and already-trained behaviours (grammar v2's checks), and
    minus every C2 training program's behaviour (so R2 adds only new combinations): (kept, report)."""
    from ycode.lm import compositions2 as G2
    from ycode.lm import idioms
    from ycode.lm.compositions import _v5_behaviours
    from ycode.lm.leakage import skeleton

    combos = all_combinations(seed)
    kept, _ = idioms.held_out_check(combos)
    report = {"combinations": len(combos), "held_out": len(combos) - len(kept)}
    avoid = idioms.neighbour_skeletons()
    cross = json.loads((ROOT / "experiments" / "v8" / "sets" / "t_families.json").read_text())["tasks"]
    avoid |= {s for s in (skeleton(t["reference"]) for t in cross) if s}
    trained = set()
    for src in SOURCE:
        trained |= _v5_behaviours([p for p in kept if PARTS[p.name][0] == src[0]])
    tasks = G2.test_tasks()
    c2 = set(json.loads((ROOT / "ycode" / "lm" / "v9_grammar_split.json").read_text())["train"])
    c2_fns = []
    for p in G2.all_combinations():
        if p.name in c2:
            ns: dict = {}
            exec(p.source(0), ns)
            c2_fns.append((p.params, ns[p.name]))
    expressible: dict[str, list[str]] = {}
    out = []
    report.update({"test_skeleton": 0, "trained_v5": 0, "computes_a_test_task": 0, "a_c2_program": 0})
    for p in kept:
        ns = {}
        exec(p.source(0), ns)
        fn = ns[p.name]
        hits = [(s, n) for s, n, args, want in tasks if G2._computes(fn, args, want)]
        for s, n in hits:
            if s == "t_families":
                expressible.setdefault(n, []).append(p.name)
        own = [repr(_run(p.source(0), p.name, list(p.inputs)))]
        if any(skeleton(p.source(i)) in avoid for i in range(len(p.bodies))):
            report["test_skeleton"] += 1
        elif p.name in trained:
            report["trained_v5"] += 1
        elif hits:
            report["computes_a_test_task"] += 1
        elif any(params == p.params and _same(f, p, own) for params, f in c2_fns):
            report["a_c2_program"] += 1
        else:
            out.append(p)
    report["kept"] = len(out)
    report["kept_by_source"] = {s[0]: sum(PARTS[p.name][0] == s[0] for p in out) for s in SOURCE}
    report["expressible_t_families"] = dict(sorted(expressible.items()))
    return out, report


def _same(fn, p: Program, own: list[str]) -> bool:
    import copy

    try:
        return [repr([fn(*copy.deepcopy(a)) for a in p.inputs])] == own
    except Exception:  # noqa: BLE001
        return False


TRAIN_PER_SOURCE = {"list": 120, "digits": 70, "wordlens": 70, "runs": 50, "letters": 40}  # 350, as C2


def grammar_split(programs: list[Program], train_per_source: dict | None = None, test_per_source: int = 12,
                  trained_sample_per_source: int = 8, seed: int = 9) -> dict[str, list[Program]]:
    """Per source: train (R2's programs), test (combinations not trained whose source, view and
    aggregate each occur in that source's training programs) and a sample of train."""
    rng = random.Random(seed)
    quota = train_per_source or TRAIN_PER_SOURCE
    out = {"train": [], "test": [], "trained_sample": []}
    for src in SOURCE:
        ps = [p for p in programs if PARTS[p.name][0] == src[0]]
        rng.shuffle(ps)
        n = min(quota[src[0]], len(ps) - test_per_source)
        train = ps[:n]
        parts = set().union(*(_parts(p) for p in train))
        test = [p for p in ps[n:] if _parts(p) <= parts][:test_per_source]
        assert len(test) == test_per_source, (src[0], len(test))
        out["train"] += train
        out["test"] += test
        out["trained_sample"] += rng.sample(train, trained_sample_per_source)
    return out


def _parts(p: Program) -> set[str]:
    src, view, agg = PARTS[p.name]
    return {f"v:{view}", f"a:{agg}"}
