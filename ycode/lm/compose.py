"""Compositional programming exercises: thousands of small tasks built from parts.

``ycode.lm.basics`` has ~185 hand-written concepts, each phrased one way. A small model trained on
them learns "this concept -> this answer" instead of what the words mean ("three times" -> ``3 *``,
"no items" -> ``len(items) == 0``). This generator combines operations, constants, filters,
transforms and aggregations, each with several English phrasings, so the words themselves must be
learned. Every exercise's tests are computed by running its reference solution on sample inputs.

Held out: each exercise has a concept key, and the keys of the evaluation problems (the standard 30
in ``ycode.lm.evaluate`` and the 20 fresh ones in ``ycode.lm.basics``) are never generated. The two
binary operations the benchmark asks for directly (sum and difference of two numbers) are only
generated with three operands or with a constant.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

NUMBER_WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
                8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}

# Concepts asked for by the evaluation problems. Never generated.
HELD_OUT_KEYS = {
    # standard 30
    ("bin", "add"), ("bin", "sub"), ("pred", "mod", 2, 0), ("unary", "pow", 2), ("unary", "mul_self"),
    ("agg", "max", "all", "id"), ("agg", "min", "all", "id"), ("agg", "sum", "all", "id"),
    ("agg", "avg", "all", "id"), ("list", "even", "id"),
    # fresh 20
    ("unary", "mul", 3), ("pred", "eq", 0), ("str_count", " "), ("str_ends", "."), ("index", 1),
    ("agg", "sum", "even", "id"), ("agg", "count", "zero", "id"), ("list", "all", "div", 2),
    ("list", "all", "add", 1), ("agg", "sum", "all", "cube"), ("len_eq", 0), ("index", "middle"),
    ("str_count", "a"), ("unary", "div", 60),
}


@dataclass(frozen=True)
class Exercise:
    key: tuple
    name: str
    params: str
    task: str
    body: str  # one or more lines, unindented
    samples: tuple  # argument tuples for computing tests

    def source(self) -> str:
        return f"def {self.name}({self.params}):\n" + "\n".join("    " + line for line in self.body.splitlines())

    def tests(self) -> str | None:
        """Assertions computed by running the reference solution on the samples."""
        namespace: dict = {}
        exec(self.source(), namespace)
        fn = namespace[self.name]
        lines = []
        for args in self.samples:
            try:
                result = fn(*args)
            except Exception:  # noqa: BLE001 - this sample is outside the function's domain
                continue
            call = f"{self.name}({', '.join(repr(a) for a in args)})"
            if isinstance(result, float):
                lines.append(f"assert abs({call} - {result!r}) < 1e-9")
            else:
                lines.append(f"assert {call} == {result!r}")
        return "\n".join(lines) if len(lines) >= 2 else None


def _w(k: int) -> str:
    return NUMBER_WORDS.get(k, str(k))


def _num(k: int, rng: random.Random) -> str:
    """k in task text: as a word ("three") or as digits, at random, so both readings are learned."""
    return _w(k) if k in NUMBER_WORDS and rng.random() < 0.5 else str(k)


def _items(k: int, rng: random.Random) -> str:
    return f"{_num(k, rng)} item" + ("" if k == 1 else "s")


# ------------------------------------------------------------------ families

_INTS = [(-3,), (0,), (1,), (4,), (7,), (12,)]
_PAIRS = [(3, 4), (10, 2), (-5, 3), (0, 7), (9, 9)]


def _unary(rng: random.Random) -> list[Exercise]:
    out = []
    ops = {
        "add": ("x + {k}", ["returns x plus {k}", "returns {k} more than x", "returns x increased by {k}"],
                ["add_{w}", "plus_{w}", "increase_by_{w}"]),
        "sub": ("x - {k}", ["returns x minus {k}", "returns {k} less than x", "returns x decreased by {k}"],
                ["minus_{w}", "decrease_by_{w}", "subtract_{w}"]),
        "mul": ("{k} * x", ["returns {k} times x", "returns x multiplied by {k}", "returns the product of x and {k}"],
                ["times_{w}", "multiply_by_{w}", "scale_by_{w}"]),
        "div": ("x / {k}", ["returns x divided by {k}", "returns x split into {k} equal parts"],
                ["divide_by_{w}", "over_{w}"]),
        "floordiv": ("x // {k}", ["returns the whole number of times {k} fits into x", "returns x integer-divided by {k}"],
                     ["floor_div_{w}", "whole_{w}s"]),
        "mod": ("x % {k}", ["returns the remainder when x is divided by {k}", "returns x modulo {k}"],
                ["mod_{w}", "remainder_{w}"]),
        "pow": ("x ** {k}", ["returns x raised to the power {k}", "returns x to the power of {k}"],
                ["power_{w}", "pow_{w}"]),
    }
    for op, (expr, tasks, names) in ops.items():
        for k in range(2, 11):
            if (op == "pow" and k > 4) or ("unary", op, k) in HELD_OUT_KEYS:
                continue
            for task, name in zip(tasks, rng.sample(names, len(names))):
                out.append(Exercise(("unary", op, k), name.format(w=_w(k)), "x", task.format(k=_num(k, rng)) + ".",
                                    f"return {expr.format(k=k)}", tuple(_INTS)))
    return out


def _ternary() -> list[Exercise]:
    out = []
    for params in ("a, b, c", "x, y, z"):
        p = params.split(", ")
        out += [
            Exercise(("tern", "add"), "sum_three_numbers", params, f"returns the sum of {p[0]}, {p[1]} and {p[2]}.",
                     f"return {p[0]} + {p[1]} + {p[2]}", ((1, 2, 3), (-1, 5, 0), (10, 10, 10))),
            Exercise(("tern", "add"), "add_all_three", params, f"returns {p[0]} plus {p[1]} plus {p[2]}.",
                     f"return {p[0]} + {p[1]} + {p[2]}", ((1, 2, 3), (4, -4, 9))),
            Exercise(("tern", "sub"), "subtract_two_from_first", params, f"returns {p[0]} minus {p[1]} minus {p[2]}.",
                     f"return {p[0]} - {p[1]} - {p[2]}", ((10, 2, 3), (0, 1, 1))),
            Exercise(("tern", "mul"), "multiply_three_numbers", params, f"returns the product of {p[0]}, {p[1]} and {p[2]}.",
                     f"return {p[0]} * {p[1]} * {p[2]}", ((1, 2, 3), (2, 2, 2), (-1, 4, 5))),
            Exercise(("tern", "min"), "min_of_three", params, f"returns the smallest of {p[0]}, {p[1]} and {p[2]}.",
                     f"return min({p[0]}, {p[1]}, {p[2]})", ((1, 2, 3), (9, -1, 4))),
            Exercise(("tern", "avg"), "mean_of_three", params, f"returns the average of {p[0]}, {p[1]} and {p[2]}.",
                     f"return ({p[0]} + {p[1]} + {p[2]}) / 3", ((1, 2, 3), (3, 3, 3), (0, 0, 9))),
        ]
    return out


def _binary() -> list[Exercise]:
    """Two-argument operations other than the benchmark's sum and difference."""
    out = []
    specs = [
        ("mul", "{a} * {b}", ["returns {a} multiplied by {b}", "returns the product of {a} and {b}"], ["multiply", "product"]),
        ("div", "{a} / {b}", ["returns {a} divided by {b}"], ["divide", "ratio"]),
        ("mod", "{a} % {b}", ["returns the remainder of {a} divided by {b}"], ["remainder_of", "mod"]),
        ("pow", "{a} ** {b}", ["returns {a} raised to the power {b}"], ["raise_power", "power"]),
        ("max", "max({a}, {b})", ["returns the larger of {a} and {b}", "returns the bigger number of {a} and {b}"],
         ["larger", "bigger"]),
        ("min", "min({a}, {b})", ["returns the smaller of {a} and {b}", "returns the lesser of {a} and {b}"],
         ["smaller", "lesser"]),
        ("absdiff", "abs({a} - {b})", ["returns the distance between {a} and {b}"], ["distance", "gap"]),
        ("eq", "{a} == {b}", ["returns True if {a} equals {b}", "returns True if {a} and {b} are equal"],
         ["are_equal", "same_value"]),
        ("gt", "{a} > {b}", ["returns True if {a} is greater than {b}"], ["is_greater", "greater_than"]),
        ("lt", "{a} < {b}", ["returns True if {a} is less than {b}"], ["is_less", "less_than"]),
        ("sumsq", "{a} * {a} + {b} * {b}", ["returns the sum of the squares of {a} and {b}"], ["sum_squares", "square_sum"]),
        ("double_sum", "2 * ({a} + {b})", ["returns twice the sum of {a} and {b}"], ["double_total", "twice_sum"]),
    ]
    for op, expr, tasks, names in specs:
        for (pa, pb), task, name in zip([("a", "b"), ("x", "y"), ("m", "n")], tasks * 3, names * 3):
            samples = ((3, 4), (10, 2), (-5, 3), (9, 9), (7, 1)) if op != "pow" else ((2, 3), (5, 0), (3, 2))
            out.append(Exercise(("bin", op), name, f"{pa}, {pb}", task.format(a=pa, b=pb) + ".",
                                f"return {expr.format(a=pa, b=pb)}", samples))
    return out


def _predicates(rng: random.Random) -> list[Exercise]:
    out = []
    specs = {
        "gt": ("n > {k}", ["returns True if n is greater than {k}", "returns True if n is more than {k}"],
               ["above_{w}", "greater_than_{w}"]),
        "lt": ("n < {k}", ["returns True if n is less than {k}", "returns True if n is smaller than {k}"],
               ["below_{w}", "less_than_{w}"]),
        "ge": ("n >= {k}", ["returns True if n is at least {k}"], ["at_least_{w}"]),
        "le": ("n <= {k}", ["returns True if n is at most {k}"], ["at_most_{w}"]),
        "eq": ("n == {k}", ["returns True if n equals {k}", "returns True if n is exactly {k}"], ["is_{w}", "equals_{w}"]),
        "ne": ("n != {k}", ["returns True if n is not equal to {k}"], ["not_{w}"]),
        "mod": ("n % {k} == 0", ["returns True if n is divisible by {k}", "returns True if n is a multiple of {k}"],
                ["divisible_by_{w}", "multiple_of_{w}"]),
    }
    for op, (expr, tasks, names) in specs.items():
        for k in range(0, 11):
            if op == "mod" and k < 2:
                continue
            key = ("pred", op, k) if op != "mod" else ("pred", "mod", k, 0)
            if key in HELD_OUT_KEYS:
                continue
            for task, name in zip(tasks, names):
                ints = ((k - 1,), (k,), (k + 1,), (k * 3 + 1,), (-k,), (k * 2,))
                out.append(Exercise(key, name.format(w=_w(k)), "n", task.format(k=_num(k, rng)) + ".",
                                    f"return {expr.format(k=k)}", ints))
    return out


_FILTERS = {
    "all": (None, "the numbers"),
    "positive": ("x > 0", "the positive numbers"),
    "negative": ("x < 0", "the negative numbers"),
    "odd": ("x % 2 == 1", "the odd numbers"),
    "even": ("x % 2 == 0", "the even numbers"),
    "zero": ("x == 0", "the zeros"),
    "nonzero": ("x != 0", "the non-zero numbers"),
    "gt5": ("x > 5", "the numbers greater than 5"),
    "lt3": ("x < 3", "the numbers less than 3"),
    "mult3": ("x % 3 == 0", "the multiples of 3"),
}
_TRANSFORMS = {
    "id": ("x", ""),
    "square": ("x * x", "the squares of "),
    "cube": ("x ** 3", "the cubes of "),
    "double": ("2 * x", "twice "),
    "abs": ("abs(x)", "the absolute values of "),
}
_LISTS = ((1, 2, 3, 4, 5, 6),), ((-3, 0, 7, 2, 9),), ((0, 0, 4),), ((10, -1, 6, 3),)


def _aggregations() -> list[Exercise]:
    out = []
    aggs = {
        "sum": ("sum({gen})", "returns the sum of {what} in the list", "sum"),
        "count": ("sum(1 for x in numbers{cond})", "returns how many numbers in the list are {adj}", "count"),
        "max": ("max({gen})", "returns the largest of {what} in the list", "largest"),
        "min": ("min({gen})", "returns the smallest of {what} in the list", "smallest"),
        "any": ("any({cond_expr} for x in numbers)", "returns True if any number in the list is {adj}", "any"),
        "all_": ("all({cond_expr} for x in numbers)", "returns True if every number in the list is {adj}", "all"),
    }
    adjectives = {"positive": "positive", "negative": "negative", "odd": "odd", "even": "even", "zero": "zero",
                  "nonzero": "non-zero", "gt5": "greater than 5", "lt3": "less than 3", "mult3": "a multiple of 3"}
    for agg, (tmpl, task, word) in aggs.items():
        for filt, (cond, what) in _FILTERS.items():
            transforms = _TRANSFORMS if agg in ("sum", "max", "min") else {"id": _TRANSFORMS["id"]}
            for tr, (texpr, tphrase) in transforms.items():
                key = ("agg", agg, filt, tr)
                if key in HELD_OUT_KEYS or (agg in ("count", "any", "all_") and filt == "all"):
                    continue
                cond_part = f" if {cond}" if cond else ""
                gen = f"{texpr} for x in numbers{cond_part}"
                if agg in ("max", "min"):
                    body = f"return {agg}({gen})"
                elif agg == "sum":
                    body = f"return sum({gen})"
                elif agg == "count":
                    body = f"return sum(1 for x in numbers if {cond})"
                else:
                    body = f"return {agg.rstrip('_')}({cond} for x in numbers)"
                what_full = f"{tphrase}{what}" if tphrase else what
                text = task.format(what=what_full, adj=adjectives.get(filt, ""))
                name = f"{word}_{filt}" + (f"_{tr}" if tr != "id" else "")
                out.append(Exercise(key, name, "numbers", text + ".", body, _LISTS))
    return out


def _list_ops(rng: random.Random) -> list[Exercise]:
    out = []
    for filt, (cond, what) in _FILTERS.items():
        if filt == "all" or ("list", filt, "id") in HELD_OUT_KEYS:
            continue
        out.append(Exercise(("list", filt, "id"), f"keep_{filt}", "numbers", f"returns a list of {what} in the list.",
                            f"return [x for x in numbers if {cond}]", _LISTS))
    maps = {("add", k): (f"x + {k}", f"with {_num(k, rng)} added to every number") for k in (1, 2, 5, 10)}
    maps.update({("sub", k): (f"x - {k}", f"with {_num(k, rng)} subtracted from every number") for k in (1, 3)})
    maps.update({("mul", k): (f"{k} * x", f"with every number multiplied by {_num(k, rng)}") for k in (2, 3, 10)})
    maps.update({("div", k): (f"x / {k}", f"with every number divided by {_num(k, rng)}") for k in (2, 4)})
    maps.update({("square", 0): ("x * x", "of the squares of the numbers"),
                 ("neg", 0): ("-x", "with the sign of every number flipped"),
                 ("abs", 0): ("abs(x)", "of the absolute values of the numbers")})
    for (op, k), (expr, phrase) in maps.items():
        if ("list", "all", op, k) in HELD_OUT_KEYS:
            continue
        name = {"add": f"add_{_w(k)}_each" if k > 1 else "increment_each", "sub": f"subtract_{_w(k)}_each",
                "mul": f"scale_each_by_{_w(k)}", "div": f"divide_each_by_{_w(k)}", "square": "square_each",
                "neg": "negate_each", "abs": "abs_each"}[op]
        out.append(Exercise(("list", "all", op, k), name, "numbers", f"returns a new list {phrase}.",
                            f"return [{expr} for x in numbers]", _LISTS))
    return out


def _sequences(rng: random.Random) -> list[Exercise]:
    out = []
    for k in (0, 2, 3, 4):
        if ("index", k) in HELD_OUT_KEYS:
            continue
        nth = {0: "first", 2: "third", 3: "fourth", 4: "fifth"}[k]
        out.append(Exercise(("index", k), f"{nth}_item", "items", f"returns the {nth} item of the list.",
                            f"return items[{k}]", (((5, 6, 7, 8, 9),), ((1, 2, 3, 4),), ((3, 1, 4, 1, 5, 9),))))
    for k in (1, 2):
        nth = {1: "last", 2: "second to last"}[k]
        out.append(Exercise(("index", -k), nth.replace(" ", "_") + "_item", "items", f"returns the {nth} item of the list.",
                            f"return items[-{k}]", (((5, 6, 7, 8, 9),), ((1, 2, 3),))))
    for k in (1, 2, 3, 5):
        out.append(Exercise(("len_gt", k), f"longer_than_{_w(k)}", "items",
                            f"returns True if the list has more than {_items(k, rng)}.", f"return len(items) > {k}",
                            (((1,) * k,), ((1,) * (k + 1),), ((),))))
        if ("len_eq", k) not in HELD_OUT_KEYS:
            out.append(Exercise(("len_eq", k), f"has_{_w(k)}_item" + ("" if k == 1 else "s"), "items",
                                f"returns True if the list has exactly {_items(k, rng)}.", f"return len(items) == {k}",
                                (((1,) * k,), ((1,) * (k + 1),), ((),))))
    out.append(Exercise(("len",), "size", "items", "returns the number of items in the list.", "return len(items)",
                        (((1, 2, 3),), ((),))))
    return out


_STRINGS = (("hello world",), ("banana",), ("",), ("Mississippi",), ("a.b.c",))


def _strings(rng: random.Random) -> list[Exercise]:
    out = []
    for ch in ("e", "o", "s", "i", "x", ",", "-"):
        if ("str_count", ch) in HELD_OUT_KEYS:
            continue
        word = {"e": "e", "o": "o", "s": "s", "i": "i", "x": "x", ",": "comma", "-": "dash"}[ch]
        out.append(Exercise(("str_count", ch), f"count_{word}", "s", f"returns how many times {ch!r} occurs in s.",
                            f"return s.count({ch!r})", _STRINGS))
        out.append(Exercise(("str_has", ch), f"has_{word}", "s", f"returns True if s contains {ch!r}.",
                            f"return {ch!r} in s", _STRINGS))
    for suffix in ("!", "?", "ing", "s", "ed"):
        if ("str_ends", suffix) in HELD_OUT_KEYS:
            continue
        name = {"!": "ends_with_bang", "?": "is_question", "ing": "ends_with_ing", "s": "ends_with_s", "ed": "ends_with_ed"}[suffix]
        out.append(Exercise(("str_ends", suffix), name, "s", f"returns True if the string s ends with {suffix!r}.",
                            f"return s.endswith({suffix!r})", (("going",), ("why?",), ("cats",), ("hi!",), ("",))))
    for prefix in ("a", "the", "#", "http"):
        name = {"a": "starts_with_a", "the": "starts_with_the", "#": "is_comment", "http": "is_url"}[prefix]
        out.append(Exercise(("str_starts", prefix), name, "s", f"returns True if the string s starts with {prefix!r}.",
                            f"return s.startswith({prefix!r})", (("apple",), ("the end",), ("# note",), ("http://x",), ("",))))
    for k in (2, 3, 5):
        out.append(Exercise(("str_first", k), f"first_{_w(k)}_chars", "s", f"returns the first {_num(k, rng)} characters of s.",
                            f"return s[:{k}]", _STRINGS))
        out.append(Exercise(("str_longer", k), f"longer_than_{_w(k)}_chars", "s",
                            f"returns True if s has more than {_num(k, rng)} characters.", f"return len(s) > {k}", _STRINGS))
    out += [
        Exercise(("str", "strip"), "trim", "s", "returns s without leading and trailing whitespace.", "return s.strip()",
                 (("  hi ",), ("x",))),
        Exercise(("str", "dots"), "dots_to_dashes", "s", "returns s with every '.' replaced by '-'.",
                 "return s.replace('.', '-')", _STRINGS),
        Exercise(("str", "words"), "word_list", "text", "returns a list of the words in text.", "return text.split()",
                 _STRINGS),
        Exercise(("str", "lower_eq"), "same_ignoring_case", "a, b", "returns True if a and b are equal ignoring case.",
                 "return a.lower() == b.lower()", (("Hi", "hI"), ("a", "b"))),
        Exercise(("str", "concat"), "concatenate", "a, b", "returns the two strings joined together.", "return a + b",
                 (("ab", "cd"), ("", "x"))),
        Exercise(("str", "is_lower"), "is_all_lower", "s", "returns True if every letter in s is lower case.",
                 "return s.islower()", (("abc",), ("aBc",))),
    ]
    return out


# Loops, unit conversions and case changes. The same concepts as above written as explicit loops
# (seven of the fifteen benchmark bug fixes are loops), plus families whose slot words must be read.
# Excluded on purpose, as semantic equivalents of held-out problems: unfiltered max/min/sum/average,
# the sum of the even numbers, counting zeros, counting spaces, vowels, the letter 'a' or one given
# character, any division by 60, halving, adding 1 to each item, "no items", the middle item, the
# second item, the sum of cubes, upper case, the first word, evenness.
_LOOP_ADJ = {"positive": "positive", "negative": "negative", "odd": "odd", "gt5": "greater than 5",
             "lt3": "less than 3", "mult3": "a multiple of 3", "nonzero": "non-zero"}


def _loops(rng: random.Random) -> list[Exercise]:
    out = []
    for filt, adj in _LOOP_ADJ.items():
        cond, what = _FILTERS[filt]
        specs = [
            ("sum", f"returns the sum of {what} in the list, using a loop.", f"loop_sum_{filt}",
             f"total = 0\nfor x in numbers:\n    if {cond}:\n        total += x\nreturn total"),
            ("count", f"counts the numbers in the list that are {adj}, using a loop.", f"loop_count_{filt}",
             f"count = 0\nfor x in numbers:\n    if {cond}:\n        count += 1\nreturn count"),
            ("max", f"returns the largest of {what} in the list, or None if there are none.", f"loop_largest_{filt}",
             f"best = None\nfor x in numbers:\n    if {cond} and (best is None or x > best):\n        best = x\nreturn best"),
            ("min", f"returns the smallest of {what} in the list, or None if there are none.", f"loop_smallest_{filt}",
             f"best = None\nfor x in numbers:\n    if {cond} and (best is None or x < best):\n        best = x\nreturn best"),
            ("any", f"returns True as soon as it finds a number in the list that is {adj}, otherwise False.",
             f"loop_any_{filt}", f"for x in numbers:\n    if {cond}:\n        return True\nreturn False"),
            ("all_", f"returns False as soon as it finds a number in the list that is not {adj}, otherwise True.",
             f"loop_all_{filt}", f"for x in numbers:\n    if not ({cond}):\n        return False\nreturn True"),
        ]
        for agg, task, name, body in specs:
            key = ("loop", "agg", agg, filt, "id")
            if key[1:] not in HELD_OUT_KEYS:
                out.append(Exercise(key, name, "numbers", task, body, _LISTS))
    for k in (3, 4, 5, 6, 7, 9):
        out.append(Exercise(("loop", "multiples", k), f"sum_multiples_of_{_w(k)}", "n",
                            f"returns the sum of the multiples of {_num(k, rng)} below n.",
                            f"total = 0\nfor i in range(n):\n    if i % {k} == 0:\n        total += i\nreturn total",
                            ((10,), (20,), (1,), (31,))))
        out.append(Exercise(("loop", "count_multiples", k), f"count_multiples_of_{_w(k)}", "n",
                            f"returns how many of the numbers from 1 to n are multiples of {_num(k, rng)}.",
                            f"count = 0\nfor i in range(1, n + 1):\n    if i % {k} == 0:\n        count += 1\nreturn count",
                            ((10,), (20,), (1,), (31,))))
    classes = {"digit": ("c.isdigit()", "digits"), "upper": ("c.isupper()", "upper-case letters"),
               "lower": ("c.islower()", "lower-case letters"), "alpha": ("c.isalpha()", "letters"),
               "punct": ("c in '.,!?'", "punctuation marks (. , ! ?)")}
    texts = (("Hello World 42!",), ("abc",), ("",), ("A1b2C3, ok?",))
    for cls, (cond, what) in classes.items():
        out.append(Exercise(("loop", "str_count", cls), f"count_{cls}_chars", "s", f"returns how many {what} s contains.",
                            f"count = 0\nfor c in s:\n    if {cond}:\n        count += 1\nreturn count", texts))
        out.append(Exercise(("loop", "str_keep", cls), f"keep_{cls}_chars", "s", f"returns only the {what} of s, in order.",
                            f"result = ''\nfor c in s:\n    if {cond}:\n        result += c\nreturn result", texts))
    return out


def _conversions(rng: random.Random) -> list[Exercise]:
    units = [("days", "hours", 24), ("weeks", "days", 7), ("kilograms", "grams", 1000), ("metres", "centimetres", 100),
             ("feet", "inches", 12), ("dozens", "items", 12), ("kilometres", "metres", 1000)]
    out = []
    for big, small, k in units:
        out.append(Exercise(("conv", big, small), f"{big}_to_{small}", big, f"converts {big} to {small}.",
                            f"return {big} * {k}", ((1,), (3,), (0,), (2.5,))))
        out.append(Exercise(("conv", small, big), f"{small}_to_{big}", small, f"converts {small} to {big}.",
                            f"return {small} / {k}", ((k,), (k * 3,), (0,), (k // 2,))))
    return out


def _cases() -> list[Exercise]:
    specs = [("lower", "s.lower()", "returns s in lower case."), ("title", "s.title()", "returns s in title case."),
             ("swap", "s.swapcase()", "returns s with the case of every letter swapped."),
             ("cap", "s.capitalize()", "returns s with only its first letter in upper case.")]
    out = [Exercise(("case", c), f"make_{c}", "s", task, f"return {expr}", (("Hello World",), ("abc",), ("",)))
           for c, expr, task in specs]
    for k, nth in ((2, "third"), (3, "fourth"), (-1, "last"), (-2, "second to last")):
        out.append(Exercise(("word", k), f"{nth.replace(' ', '_')}_word", "text", f"returns the {nth} word of text.",
                            f"return text.split()[{k}]", (("a b c d e",), ("one two three four",), ("x y",))))
    return out


def base_key(key: tuple) -> tuple:
    """The concept behind a key: a loop rendering shares its concept with the one-liner."""
    return key[1:] if key and key[0] == "loop" else key


def _raw_exercises(seed: int = 7) -> list[Exercise]:
    rng = random.Random(seed)
    return (_unary(rng) + _ternary() + _binary() + _predicates(rng) + _aggregations() + _list_ops(rng)
            + _sequences(rng) + _strings(rng) + _loops(rng) + _conversions(rng) + _cases())


def all_exercises(seed: int = 7) -> list[Exercise]:
    """Training exercises: never a benchmark concept or name, a basics concept's name, or a dev item."""
    from ycode.lm.basics import CONCEPTS, held_out_names
    from ycode.lm.devset import DEV_KEYS

    basics_names = {n for c in CONCEPTS for n in (c.name, *c.aliases)}
    exercises = [e for e in _raw_exercises(seed)
                 if base_key(e.key) not in DEV_KEYS and e.name not in basics_names]
    clash = [e for e in exercises if base_key(e.key) in HELD_OUT_KEYS]
    if clash:
        raise AssertionError(f"held-out concepts generated: {sorted({e.key for e in clash})}")
    names = {e.name for e in exercises} & held_out_names()
    if names:
        raise AssertionError(f"benchmark function names generated: {sorted(names)}")
    return exercises


_PROMPTS = (
    "Write a Python function `{sig}` that {task}",
    "Write a Python function `{sig}` that {task}",
    "Write a Python function `{sig}` that {task}",
    "Write a function {sig} that {task}",
    "Implement `{sig}`, which {task}",
)


def _documented(ex: Exercise, body: str | None = None) -> str:
    doc = ex.task[0].upper() + ex.task[1:]
    if doc.startswith("Returns "):
        doc = "Return " + doc[len("Returns "):]
    return (f'def {ex.name}({ex.params}):\n    """{doc}"""\n'
            + "\n".join("    " + line for line in (ex.body if body is None else body).splitlines()))


def compose_examples(seed: int = 7, per_exercise: int = 1):
    """Write-a-function examples (prompt, code-only answer) with verified, auto-generated tests."""
    from ycode.lm.basics import neutralize
    from ycode.lm.data import InstructionExample

    rng = random.Random(seed)
    out = []
    for ex in all_exercises(seed):
        if ex.tests() is None:
            continue
        for _ in range(per_exercise):
            prompt = rng.choice(_PROMPTS).format(sig=f"{ex.name}({ex.params})", task=ex.task)
            out.append(neutralize(InstructionExample(prompt, f"```python\n{ex.source()}\n```"), ex.name, rng))
    rng.shuffle(out)
    return out


def compose_bugfix_examples(seed: int = 7, per_exercise: int = 1):
    """Find-and-fix-the-bug examples from the exercises: realistic single bugs, answer = fixed code."""
    from ycode.lm.basics import BUGFIX_PROMPT, _parses, _passes, _text_bugs, neutralize
    from ycode.lm.data import InstructionExample, inject_bug

    rng = random.Random(seed + 1)
    out = []
    for ex in all_exercises(seed):
        tests = ex.tests()
        if tests is None:
            continue
        fixed = _documented(ex)
        candidates = _text_bugs(fixed)
        for _ in range(3):
            ast_bug = inject_bug(fixed, rng)
            if ast_bug and ast_bug not in candidates:
                candidates.append(ast_bug)
        rng.shuffle(candidates)
        made = 0
        for buggy, _wrong, _right in candidates:
            if made >= per_exercise:
                break
            if not _parses(buggy) or _passes(buggy, tests):
                continue
            ex_out = InstructionExample(BUGFIX_PROMPT.format(code=buggy), f"```python\n{fixed}\n```")
            out.append(neutralize(ex_out, ex.name, rng))
            made += 1
    rng.shuffle(out)
    return out


def contrast_examples(seed: int = 7, per_exercise: int = 1):
    """Bug fixes whose buggy body is a neighbouring exercise's correct code (largest vs smallest, positive
    vs negative, plus vs minus k, any vs all, ...): only the docstring says what is wrong, so copying the
    code back can never be right."""
    from collections import defaultdict

    from ycode.lm.basics import BUGFIX_PROMPT, _parses, _passes, neutralize
    from ycode.lm.data import InstructionExample

    rng = random.Random(seed + 2)
    exercises = [(ex, t) for ex in all_exercises(seed) if (t := ex.tests()) is not None]
    groups = defaultdict(list)
    for ex, _ in exercises:
        groups[(ex.key[0], len(ex.key), ex.params)].append(ex)
    out = []
    for ex, tests in exercises:
        partners = [o for o in groups[(ex.key[0], len(ex.key), ex.params)]
                    if sum(a != b for a, b in zip(o.key, ex.key)) == 1 and o.body != ex.body]
        rng.shuffle(partners)
        made = 0
        for other in partners:
            if made >= per_exercise:
                break
            buggy = _documented(ex, other.body)
            if not _parses(buggy) or _passes(buggy, tests):
                continue
            answer = f"```python\n{_documented(ex)}\n```"
            out.append(neutralize(InstructionExample(BUGFIX_PROMPT.format(code=buggy), answer), ex.name, rng))
            made += 1
    rng.shuffle(out)
    return out
