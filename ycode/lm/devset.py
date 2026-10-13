"""A held-out dev set for choosing fine-tuning checkpoints.

Picking checkpoints by benchmark score turns the benchmark into a training signal. These items are
used instead: like the benchmark, they are never trained on. They are held out of the training
generators by concept (``DEV_KEYS`` for ycode.lm.compose, ``DEV_CONCEPT_NAMES`` for ycode.lm.basics),
not just by seed, because generated answers repeat across seeds. Most are one slot word away from a
concept that *is* trained ("plus seven" when "plus six" is trained), so they test reading the task.
"""

from __future__ import annotations

from ycode.lm.evaluate import Problem

# Compose concepts held out of training.
DEV_KEYS = frozenset({
    ("unary", "add", 7), ("unary", "sub", 5), ("unary", "mul", 4), ("unary", "mod", 3), ("unary", "floordiv", 4),
    ("pred", "gt", 6), ("pred", "lt", 2), ("pred", "mod", 5, 0), ("pred", "ne", 3),
    ("agg", "sum", "odd", "id"), ("agg", "count", "odd", "id"), ("agg", "max", "even", "id"),
    ("agg", "min", "positive", "id"), ("agg", "any", "zero", "id"), ("agg", "all_", "odd", "id"),
    ("agg", "sum", "positive", "square"),
    ("list", "negative", "id"), ("list", "gt5", "id"), ("list", "all", "mul", 3), ("list", "all", "sub", 3),
    ("index", 3), ("index", -2), ("len_gt", 3), ("len_eq", 2),
    ("str_count", "o"), ("str_has", "x"), ("str_ends", "ing"), ("str_starts", "the"), ("str_first", 3),
    ("tern", "mul"), ("conv", "weeks", "days"), ("multiples", 5),
})

# Basics concepts held out of training (their buggy versions below are the dev bug fixes).
DEV_CONCEPT_NAMES = frozenset({
    "cube", "is_negative", "sum_of_squares", "product", "count_greater", "index_of_min", "last_word",
    "remove_spaces", "count_uppercase", "longest_word", "double_all", "is_between", "first_negative",
    "smaller", "running_max",
})

# One realistic bug per dev concept, the same kinds as the benchmark's: operator swaps, flipped
# comparisons, 0/1 starts, off-by-one range ends, the wrong variable returned, x * x -> x * 2.
_DEV_BUGS = {
    "cube": "return x * x * 2",
    "is_negative": "return x > 0",
    "sum_of_squares": "total = 0\nfor i in range(1, n):\n    total += i * i\nreturn total",
    "product": "result = 0\nfor x in numbers:\n    result *= x\nreturn result",
    "count_greater": "count = 0\nfor n in numbers:\n    if n < limit:\n        count += 1\nreturn count",
    "index_of_min": ("best = 0\nfor i in range(1, len(numbers)):\n    if numbers[i] > numbers[best]:\n"
                     "        best = i\nreturn best"),
    "last_word": "return text.split()[0]",
    "remove_spaces": "return s.replace(' ', '_')",
    "count_uppercase": "return sum(1 for c in s if c.islower())",
    "longest_word": "best = ''\nfor word in text.split():\n    if len(word) < len(best):\n        best = word\nreturn best",
    "double_all": "return [n + 2 for n in numbers]",
    "is_between": "return low < x < high",
    "first_negative": "for n in numbers:\n    if n > 0:\n        return n\nreturn None",
    "smaller": "if a < b:\n    return a\nreturn a",
    "running_max": ("result = []\nbest = None\nfor n in numbers:\n    if best is None or n < best:\n        best = n\n"
                    "    result.append(best)\nreturn result"),
}


def _concepts():
    from ycode.lm.basics import CONCEPTS

    found = {c.name: c for c in CONCEPTS if c.name in DEV_CONCEPT_NAMES}
    assert set(found) == DEV_CONCEPT_NAMES, DEV_CONCEPT_NAMES - set(found)
    return [found[n] for n in sorted(found)]


def dev_problems() -> tuple[Problem, ...]:
    """Write-a-function dev problems, in the benchmark's prompt format (one per held-out concept)."""
    from ycode.lm.basics import _rename_tests
    from ycode.lm.compose import _raw_exercises, base_key

    out = [Problem(c.name, f"Write a Python function `{c.name}({c.params})` that {c.task}", _rename_tests(c.tests, c.name))
           for c in _concepts()]
    seen = set()
    for ex in _raw_exercises(seed=0):  # a fixed rendering of each held-out compose concept
        if base_key(ex.key) in DEV_KEYS and base_key(ex.key) not in seen and (tests := ex.tests()) is not None:
            seen.add(base_key(ex.key))
            out.append(Problem(ex.name, f"Write a Python function `{ex.name}({ex.params})` that {ex.task}", tests))
    return tuple(out)


def dev_bugs() -> tuple[tuple[str, str, str], ...]:
    """(signature, buggy code with docstring, tests) for each dev concept, like evaluate.BUGGY."""
    from ycode.lm.basics import _rename_tests, _with_doc

    out = []
    for c in _concepts():
        body = "\n".join("    " + line for line in _DEV_BUGS[c.name].splitlines())
        out.append((f"{c.name}({c.params})", _with_doc(c, c.name, body), _rename_tests(c.tests, c.name)))
    return tuple(out)


def verify_devset() -> None:
    """Every dev problem's reference passes its tests, every dev bug fails them, and none is trained on."""
    from ycode.lm.basics import _passes, held_out_names
    from ycode.lm.compose import HELD_OUT_KEYS

    assert not DEV_KEYS & HELD_OUT_KEYS
    problems, bugs = dev_problems(), dev_bugs()
    names = {p.name for p in problems} | {sig.split("(")[0] for sig, _, _ in bugs}
    assert not names & held_out_names(), names & held_out_names()
    assert len(problems) == len(DEV_CONCEPT_NAMES) + len(DEV_KEYS), len(problems)  # every key renders
    for c in _concepts():
        assert _passes(c.source(), next(p.tests for p in problems if p.name == c.name)), c.name
    for sig, code, tests in bugs:
        assert not _passes(code, tests), f"dev bug in {sig} does not fail its tests"
