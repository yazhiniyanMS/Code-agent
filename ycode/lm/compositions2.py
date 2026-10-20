"""V9-1: grammar v2, ``aggregate(transform(selection(input)))`` over four domains whose parts all come
from the V8 training families.

The domains are an integer list (new views of it: differences between neighbours, sums of consecutive
pairs, odd positions, the last three, the numbers before the first negative, the running maximum), the
digits of a non-negative integer, the characters of a string and the words of a text. Every selection,
transform and aggregate names the training family (``ycode/lm/v8_transfer_split.json``) it was taken
from; none comes from a held-out family or a dev, unseen or benchmark item. Names say what a program
computes: ``count_above_3_...`` (V8's ``count_<selection>`` meant "how many are greater than 3" and
pulled ``count_odd`` towards that meaning).

``grammar_programs`` keeps one combination per behaviour and drops held-out items (name, skeleton,
behaviour), functions round 54 was trained on, and combinations that share a skeleton with, or compute
on its own probe inputs, a cross-family task, a program of a held-out family, a within-family test or a
neighbour task. The cross-family tasks a combination computes are recorded: they are the held-out-family
tasks grammar v2 can express (an upper bound on what V9-1 can show there).
"""
from __future__ import annotations

import ast
import copy
import json
import random
from pathlib import Path

from ycode.lm.idioms import Program, _run, make_inputs

ROOT = Path(__file__).resolve().parents[2]

# domain -> (parameter, input kind, element variable)
DOMAINS = {
    "numbers": ("numbers", "ints", "x"),
    "digits": ("n", "nonneg", "d"),
    "chars": ("s", "str", "c"),
    "words": ("text", "text", "w"),
}

# (domain, key, phrase completing "the ...", code: an expression or lines building xs, source family)
SELECT = (
    ("numbers", "all", "numbers in the list", "numbers", "map_filtered"),
    ("numbers", "steps", "differences between neighbouring numbers of the list (each number minus the one before it)",
     "[b - a for a, b in zip(numbers, numbers[1:])]", "neighbour_steps"),
    ("numbers", "pairs", "sums of the consecutive pairs of the list (positions 0 and 1, 2 and 3, ...; a last "
     "number without a partner counts alone)", "[sum(numbers[i:i + 2]) for i in range(0, len(numbers), 2)]", "chunk_sums"),
    ("numbers", "oddpos", "numbers at odd positions of the list (1, 3, 5, ...)", "numbers[1::2]", "sum_every_kth_position"),
    ("numbers", "last3", "last three numbers of the list", "numbers[-3:]", "last_reversed"),
    ("numbers", "beforeneg", "numbers that come before the first negative number of the list",
     "xs = []\nfor x in numbers:\n    if x < 0:\n        break\n    xs.append(x)", "before_first_negative"),
    ("numbers", "runmax", "running largest values of the list (at each position, the largest number so far)",
     "xs = []\nfor x in numbers:\n    xs.append(x if not xs or x > xs[-1] else xs[-1])", "running_extreme"),
    ("numbers", "after1", "numbers after the first one in the list", "numbers[1:]", "sum_after_first"),
    ("digits", "all", "digits of n", "[int(c) for c in str(n)]", "count_digit"),
    ("digits", "even", "even digits of n", "[int(c) for c in str(n) if int(c) % 2 == 0]", "digits_above"),
    ("digits", "odd", "odd digits of n", "[int(c) for c in str(n) if int(c) % 2 == 1]", "digits_above"),
    ("digits", "big", "digits of n that are greater than 4", "[int(c) for c in str(n) if int(c) > 4]", "digits_above"),
    ("digits", "nonzero", "non-zero digits of n", "[int(c) for c in str(n) if c != '0']", "count_digit"),
    ("digits", "rev", "digits of n from the last to the first", "[int(c) for c in str(n)[::-1]]", "arrange_digits"),
    ("digits", "distinct", "different digits of n, each once, in order of first appearance",
     "[int(c) for c in dict.fromkeys(str(n))]", "missing_digits"),
    ("chars", "all", "characters of s", "list(s)", "truncate"),
    ("chars", "vowels", "vowels of s (a, e, i, o and u, in either case)", "[c for c in s if c.lower() in 'aeiou']",
     "mask_vowels"),
    ("chars", "upper", "upper-case letters of s", "[c for c in s if c.isupper()]", "camel_to_snake"),
    ("chars", "digits", "digit characters of s", "[c for c in s if c.isdigit()]", "split_digit_runs"),
    ("chars", "letters", "letters of s", "[c for c in s if c.isalpha()]", "letter_counts"),
    ("chars", "rev", "characters of s from the last to the first", "list(s[::-1])", "last_reversed"),
    ("chars", "half", "characters of the first half of s (for an odd length, the middle character belongs to the "
     "second half)", "list(s[:len(s) // 2])", "half_of_string"),
    ("words", "all", "words of text (separated by spaces)", "text.split()", "long_words"),
    ("words", "cap", "words of text that start with an upper-case letter", "[w for w in text.split() if w[0].isupper()]",
     "capitalized_words"),
    ("words", "long", "words of text that are longer than 3 letters", "[w for w in text.split() if len(w) > 3]",
     "long_words"),
    ("words", "with_a", "words of text that contain the letter a", "[w for w in text.split() if 'a' in w]",
     "words_containing"),
    ("words", "first2", "first two words of text", "text.split()[:2]", "truncate"),
    ("words", "rev", "words of text in reverse order", "text.split()[::-1]", "last_reversed"),
    ("words", "sorted", "words of text in alphabetical order", "sorted(text.split())", "sort_words_by"),
)
# (domains, key, phrase around {sel}, element expression in v, output type, source family)
TRANSFORM = (
    (("numbers", "digits"), "id", "{sel}", "{v}", "num", "map_filtered"),
    (("numbers", "digits"), "sq", "squares of the {sel}", "{v} * {v}", "num", "sorted_squares"),
    (("numbers", "digits"), "dbl", "doubles of the {sel}", "2 * {v}", "num", "map_filtered"),
    (("numbers",), "abs", "absolute values of the {sel}", "abs({v})", "num", "sorted_squares"),
    (("numbers",), "cap5", "{sel}, each capped at 5", "min({v}, 5)", "num", "clip"),
    (("chars", "words"), "id", "{sel}", "{v}", "str", "truncate"),
    (("chars", "words"), "upper", "{sel} in upper case", "{v}.upper()", "str", "camel_to_snake"),
    (("chars",), "lower", "{sel} in lower case", "{v}.lower()", "str", "camel_to_snake"),
    (("words",), "len", "lengths of the {sel}", "len({v})", "num", "group_by_length"),
    (("words",), "initial", "first letters of the {sel}", "{v}[0]", "str", "numeronyms"),
    (("words",), "backwards", "{sel}, each spelled backwards", "{v}[::-1]", "str", "last_reversed"),
)
# (key, element types, task template with {t}, comprehension-style body, loop body, source family) over ys
AGGREGATE = (
    ("sum", ("num",), "returns the sum of the {t}.", "return sum(ys)",
     "total = 0\nfor y in ys:\n    total += y\nreturn total", "sum_of_first"),
    ("max", ("num",), "returns the largest of the {t} (None if there are none).", "return max(ys) if ys else None",
     "best = None\nfor y in ys:\n    if best is None or y > best:\n        best = y\nreturn best", "extreme_of_first"),
    ("min", ("num",), "returns the smallest of the {t} (None if there are none).", "return min(ys) if ys else None",
     "best = None\nfor y in ys:\n    if best is None or y < best:\n        best = y\nreturn best", "extreme_of_first"),
    ("prod", ("num",), "returns the product of the {t} (1 if there are none).",
     "result = 1\nfor y in ys:\n    result *= y\nreturn result",
     "result = 1\nfor y in ys:\n    result = result * y\nreturn result", "dot_product"),
    ("count_above_3", ("num",), "returns how many of the {t} are greater than 3.", "return sum(1 for y in ys if y > 3)",
     "count = 0\nfor y in ys:\n    if y > 3:\n        count += 1\nreturn count", "count_positive_in_first"),
    ("join", ("str",), "returns the {t} joined into one string{sep}.", "return {j}.join(ys)",
     "result = ''\nfor y in ys:\n    result = result + {jl}y if result else y\nreturn result", "camel_to_snake"),
    ("list", ("num", "str"), "returns a list of the {t}.", "return ys",
     "result = []\nfor y in ys:\n    result.append(y)\nreturn result", "map_filtered"),
    ("count", ("num", "str"), "returns how many {t} there are.", "return len(ys)",
     "count = 0\nfor y in ys:\n    count += 1\nreturn count", "count_singletons"),
    ("ndistinct", ("num", "str"), "returns how many different values there are among the {t}.", "return len(set(ys))",
     "seen = []\nfor y in ys:\n    if y not in seen:\n        seen.append(y)\nreturn len(seen)", "count_singletons"),
    ("first", ("num", "str"), "returns the first of the {t} (None if there are none).", "return ys[0] if ys else None",
     "for y in ys:\n    return y\nreturn None", "extreme_of_first"),
    ("last", ("num", "str"), "returns the last of the {t} (None if there are none).", "return ys[-1] if ys else None",
     "result = None\nfor y in ys:\n    result = y\nreturn result", "last_position"),
    ("sorted_distinct", ("num", "str"), "returns the different {t} in increasing order, each once.",
     "return sorted(set(ys))", "result = []\nfor y in sorted(ys):\n    if y not in result:\n        result.append(y)\n"
     "return result", "sorted_distinct"),
)

PARTS: dict[str, tuple[str, str, str, str]] = {}  # program name -> (domain, selection, transform, aggregate)


SEL_NAMES = {  # (domain, selection) -> the words a name uses for it
    ("numbers", "all"): "", ("numbers", "steps"): "steps", ("numbers", "pairs"): "pair_sums",
    ("numbers", "oddpos"): "odd_positions", ("numbers", "last3"): "last_three", ("numbers", "beforeneg"): "before_negative",
    ("numbers", "runmax"): "running_max", ("numbers", "after1"): "after_first",
    ("digits", "all"): "digits", ("digits", "even"): "even_digits", ("digits", "odd"): "odd_digits",
    ("digits", "big"): "digits_above_4", ("digits", "nonzero"): "nonzero_digits", ("digits", "rev"): "digits_reversed",
    ("digits", "distinct"): "distinct_digits",
    ("chars", "all"): "chars", ("chars", "vowels"): "vowels", ("chars", "upper"): "upper_letters",
    ("chars", "digits"): "digit_chars", ("chars", "letters"): "letters", ("chars", "rev"): "chars_reversed",
    ("chars", "half"): "first_half_chars",
    ("words", "all"): "words", ("words", "cap"): "capitalized_words", ("words", "long"): "long_words",
    ("words", "with_a"): "words_with_a", ("words", "first2"): "first_two_words", ("words", "rev"): "words_reversed",
    ("words", "sorted"): "sorted_words",
}
TR_NAMES = {"id": "", "sq": "squares", "dbl": "doubles", "abs": "abs", "cap5": "capped", "upper": "uppercase",
            "lower": "lowercase", "len": "lengths", "initial": "initials", "backwards": "backwards"}


def _name(domain: str, sel: str, tr: str, agg: str) -> str:
    words = [w for w in (agg, TR_NAMES[tr], SEL_NAMES[(domain, sel)]) if w]
    return "_".join(words if len(words) > 1 else words + ["all"])


def _program(sel, tr, agg, inputs) -> Program | None:
    domain, skey, sphrase, scode, _ = sel
    _, tkey, tphrase, texpr, ttype, _ = tr
    akey, types, atask, comp, loop, _ = agg
    if ttype not in types:
        return None
    param, _, v = DOMAINS[domain]
    elem = texpr.format(v=v)
    if "\n" in scode:  # lines building xs
        lines = scode + ("\nys = xs" if tkey == "id" else f"\nys = [{elem} for {v} in xs]")
    else:
        lines = f"ys = {scode}" if tkey == "id" else f"ys = [{elem} for {v} in {scode}]"
    words = domain == "words"
    join, joinl = ("' '", "' ' + ") if words else ("''", "")
    sep = ", separated by single spaces" if words else ""
    task = atask.format(t=tphrase.format(sel=sphrase), sep=sep)
    bodies = tuple(f"{lines}\n{b.format(j=join, jl=joinl)}" for b in (comp, loop))
    name = _name(domain, skey, tkey, akey)
    PARTS[name] = (domain, skey, tkey, akey)
    return Program(f"comp2:{domain}:{akey}", name, param, (task,), bodies, 2, tuple(inputs))


def all_combinations(seed: int = 0) -> list[Program]:
    """Every type-correct combination, verified by execution (both implementations agree, output not
    constant), one per behaviour within each domain."""
    rng = random.Random(seed)
    out = []
    for domain, (_, kind, _) in DOMAINS.items():
        inputs = make_inputs((kind,), rng)
        seen = set()
        for agg in AGGREGATE:
            for tr in (t for t in TRANSFORM if domain in t[0]):
                for sel in (s for s in SELECT if s[0] == domain):
                    p = _program(sel, tr, agg, inputs)
                    if p is None:
                        continue
                    results = [_run(p.source(i), p.name, list(inputs)) for i in range(len(p.bodies))]
                    if any(r is None for r in results) or repr(results[0]) != repr(results[1]):
                        raise AssertionError(f"{p.name}: implementations disagree or fail")
                    key = repr(results[0])
                    if len({repr(o) for o in results[0]}) < 2 or key in seen:
                        continue
                    seen.add(key)
                    out.append(p)
    return out


def test_tasks() -> list[tuple[str, str, list, list[str]]]:
    """(set, task name, argument tuples, expected reprs) of every task V9-1 must not train on: the
    cross-family set, every program of a held-out family, the within-family tests and the neighbours."""
    from ycode.lm import idioms

    out = []
    for name in ("t_families", "t_within", "neighbours"):
        for t in json.loads((ROOT / "experiments" / "v8" / "sets" / f"{name}.json").read_text())["tasks"]:
            out.append((name, t["name"], [ast.literal_eval(a) for a in t["probe_args"]], t["probe_expected"]))
    split = json.loads((ROOT / "ycode" / "lm" / "v8_transfer_split.json").read_text())
    held = set(split["held_out_families"])
    fams = [f for f in idioms.families("v3") if f.key in held]
    for p in idioms.candidate_programs(family_list=tuple(fams)):
        want = _run(p.source(0), p.name, list(p.inputs))
        if want is not None:
            out.append(("held_out_family", p.name, list(p.inputs), [repr(w) for w in want]))
    return out


def _computes(fn, args: list, want: list[str]) -> bool:
    try:
        return [repr(fn(*copy.deepcopy(a))) for a in args] == want
    except Exception:  # noqa: BLE001 - a combination that cannot take these inputs does not compute the task
        return False


def grammar_programs(seed: int = 0) -> tuple[list[Program], dict]:
    """``all_combinations`` minus held-out and already-trained behaviours: (kept, report). The report
    counts every drop and lists the held-out-family tasks some combination computes (``expressible``)."""
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
    for domain in DOMAINS:  # each domain has its own inputs
        trained |= _v5_behaviours([p for p in kept if PARTS[p.name][0] == domain])
    tasks = test_tasks()
    expressible: dict[str, list[str]] = {}
    out = []
    report.update({"test_skeleton": 0, "trained_v5": 0, "computes_a_test_task": 0})
    for p in kept:
        ns: dict = {}
        exec(p.source(0), ns)
        hits = [(s, n) for s, n, args, want in tasks if _computes(ns[p.name], args, want)]
        for s, n in hits:
            if s == "t_families":
                expressible.setdefault(n, []).append(p.name)
        if any(skeleton(p.source(i)) in avoid for i in range(len(p.bodies))):
            report["test_skeleton"] += 1
        elif p.name in trained:
            report["trained_v5"] += 1
        elif hits:
            report["computes_a_test_task"] += 1
        else:
            out.append(p)
    report["kept"] = len(out)
    report["kept_by_domain"] = {d: sum(PARTS[p.name][0] == d for p in out) for d in DOMAINS}
    report["expressible_t_families"] = dict(sorted(expressible.items()))
    return out, report


TRAIN_PER_DOMAIN = {"numbers": 105, "digits": 85, "chars": 55, "words": 105}  # 350, COMP's 352 programs


def grammar_split(programs: list[Program], train_per_domain: dict | None = None, test_per_domain: int = 15,
                  trained_sample_per_domain: int = 10, seed: int = 9) -> dict[str, list[Program]]:
    """Per domain: train (the C2 arm's programs), test (combinations not trained whose selection,
    transform and aggregate each occur in that domain's training programs) and a sample of train."""
    rng = random.Random(seed)
    quota = train_per_domain or TRAIN_PER_DOMAIN
    out = {"train": [], "test": [], "trained_sample": []}
    for domain in DOMAINS:
        ps = [p for p in programs if PARTS[p.name][0] == domain]
        rng.shuffle(ps)
        train = ps[:quota[domain]]
        parts = set().union(*(_parts(p) for p in train))
        test = [p for p in ps[quota[domain]:] if _parts(p) <= parts][:test_per_domain]
        assert len(test) == test_per_domain, (domain, len(test))
        out["train"] += train
        out["test"] += test
        out["trained_sample"] += rng.sample(train, trained_sample_per_domain)
    return out


def _parts(p: Program) -> set[str]:
    domain, sel, tr, agg = PARTS[p.name]
    return {f"s:{sel}", f"t:{tr}", f"a:{agg}"}
