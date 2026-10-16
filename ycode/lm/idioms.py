"""Idiom curriculum (V6): verified programs built from general Python idioms, for instruction tuning.

v5's fine-tuning data re-emits ~100 hand-written concepts every round (one round: 1,768 write
examples with 717 distinct bodies, mostly one-line arithmetic such as ``return x + 10``). After 54
rounds the model's loss on them is ~0.01: a fine-tuning run spends its tokens on what the model
already knows, while on new tasks it writes the body of the nearest trained function (66% of
unseen-task failures). This module adds programs the model has not seen, built from idioms that
general tasks are made of: slices inside aggregates, neighbouring pairs, positions, accumulation,
distinct values, counting with dictionaries, digits, words, two-list operations, small loops with
state.

* **Families.** Each family has several constants or conditions (``variants``), several task
  phrasings and several implementations (comprehension, loop, builtin), so the same behaviour is
  seen written in different ways and the same idiom in different tasks.
* **Verified.** Every implementation of every program is executed on generated inputs from the
  family's domain: all must run without error, agree with each other, and not return a constant.
* **Held out more strictly than v5's data.** A program is dropped if its name is a held-out name,
  if it computes a held-out item (``leakage.behaviour_leaks``), or if any of its implementations
  has the AST skeleton of a held-out reference (``leakage.skeleton``: the same code with other
  names, constants, operators, min/max, ...). So no near-twin of a held-out item is trained on:
  what the held-out sets measure is transfer of idioms to new tasks.
"""

from __future__ import annotations

import copy
import random
import signal
from dataclasses import dataclass, field

from ycode.lm.data import InstructionExample, inject_bug

IDIOMS_VERSION = "idioms-v3"  # v1: the first 76 families (FAMILIES_V1); v2 adds 19 multi-step algorithm families,
                               # plausible-but-wrong repairs and worked input-output examples (opt-in options);
                               # v3 adds 120 families from ycode.lm.idioms_extra (written without reading any
                               # evaluation set; filtered by the same held-out check)

NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
       10: "ten", 12: "twelve", 13: "thirteen", 100: "a hundred"}
ORD = {2: "second", 3: "third", 4: "fourth", 5: "fifth"}


@dataclass(frozen=True)
class Family:
    key: str
    params: str
    kinds: tuple[str, ...]  # input kind of each parameter (see _value)
    names: tuple[str, ...]  # name templates
    tasks: tuple[str, ...]  # complete "Write a Python function `sig` that ..."
    bodies: tuple[str, ...]  # implementations, body only, 4-space nesting, no leading indentation
    variants: tuple[dict, ...] = ({},)
    level: int = 1  # 1: one idiom; 2: several steps or state


@dataclass(frozen=True)
class Program:
    family: str
    name: str
    params: str
    tasks: tuple[str, ...]
    bodies: tuple[str, ...]
    level: int
    inputs: tuple = field(default=(), compare=False, repr=False)

    def source(self, body: int = 0, name: str | None = None) -> str:
        lines = self.bodies[body].splitlines()
        return f"def {name or self.name}({self.params}):\n" + "\n".join("    " + line for line in lines)


def _k(*values):
    return tuple({"k": v, "kw": NUM.get(v, str(v))} for v in values)


_PRED = (  # (condition on x, word, name word)
    ("x % 2 == 0", "even", "even"), ("x % 2 == 1", "odd", "odd"), ("x < 0", "negative", "negative"),
    ("x == 0", "zero", "zero"), ("x > 10", "greater than 10", "big"),
)
_MINMAX = ({"fn": "max", "big": "largest", "nm": "max", "cmp": ">"},
           {"fn": "min", "big": "smallest", "nm": "min", "cmp": "<"})
_CHARCLASS = ({"m": "isdigit", "desc": "digits", "nm": "digits"}, {"m": "isupper", "desc": "upper-case letters", "nm": "upper"},
              {"m": "isalpha", "desc": "letters", "nm": "letters"}, {"m": "isspace", "desc": "spaces", "nm": "spaces"})

FAMILIES: tuple[Family, ...] = (
    # ---- slices inside aggregates
    Family("sum_after_first", "numbers", ("ints",), ("sum_after_first_{kw}", "sum_skipping_{kw}"),
           ("returns the sum of the numbers that come after the first {kw} in the list (0 if there are none).",
            "returns the total of the list without its first {kw} numbers."),
           ("return sum(numbers[{k}:])", "total = 0\nfor x in numbers[{k}:]:\n    total += x\nreturn total",
            "return sum(x for i, x in enumerate(numbers) if i >= {k})"), _k(1, 2, 3, 4)),
    Family("sum_of_first", "numbers", ("ints",), ("sum_of_first_{kw}", "first_{kw}_total"),
           ("returns the sum of the first {kw} numbers of the list (of all of them if the list is shorter).",
            "adds up the first {kw} numbers in the list and returns the result."),
           ("return sum(numbers[:{k}])", "total = 0\nfor x in numbers[:{k}]:\n    total += x\nreturn total",
            "return sum(x for i, x in enumerate(numbers) if i < {k})"), _k(2, 3, 4, 5)),
    Family("extreme_of_first", "numbers", ("ints1",), ("{nm}_of_first_{kw}", "{nm}_in_first_{kw}"),
           ("returns the {big} of the first {kw} numbers in the non-empty list (of all of them if there are fewer).",),
           ("return {fn}(numbers[:{k}])",
            "best = numbers[0]\nfor x in numbers[1:{k}]:\n    if x {cmp} best:\n        best = x\nreturn best"),
           tuple({**m, **k} for m in _MINMAX for k in _k(2, 3, 4))),
    Family("count_positive_in_first", "numbers", ("ints",), ("positives_in_first_{kw}",),
           ("returns how many of the first {kw} numbers in the list are greater than zero.",),
           ("return sum(1 for x in numbers[:{k}] if x > 0)", "return len([x for x in numbers[:{k}] if x > 0])",
            "count = 0\nfor x in numbers[:{k}]:\n    if x > 0:\n        count += 1\nreturn count"), _k(3, 4, 5)),
    Family("slice_between", "items", ("ints",), ("items_{a}_to_{b}", "slice_{a}_{b}"),
           ("returns a new list with the items from position {a} up to, but not including, position {b} "
            "(positions start at 0; fewer items if the list is short).",
            "returns the items at positions {a} to {bm} inclusive (positions start at 0), as a new list."),
           ("return items[{a}:{b}]", "return [x for i, x in enumerate(items) if {a} <= i < {b}]"),
           tuple({"a": a, "b": b, "bm": b - 1} for a, b in ((1, 3), (2, 5), (1, 4), (3, 6)))),
    Family("every_kth", "items", ("ints",), ("every_{kw}_from_{o}",),
           ("returns the items at positions {o}, {o} + {k}, {o} + 2 * {k}, ... of the list (positions start at 0).",),
           ("return items[{o}::{k}]", "return [x for i, x in enumerate(items) if i >= {o} and (i - {o}) % {k} == 0]"),
           tuple({"o": o, **k} for o, k in ((0, _k(3)[0]), (1, _k(3)[0]), (2, _k(4)[0]), (0, _k(4)[0])))),
    Family("last_reversed", "items", ("ints",), ("last_{kw}_reversed",),
           ("returns the last {kw} items of the list in reverse order, the last item first (all of them if the list is shorter).",),
           ("return items[::-1][:{k}]", "return list(reversed(items))[:{k}]"), _k(2, 3, 4)),
    # ---- neighbouring pairs
    Family("neighbour_op", "numbers", ("ints",), ("neighbour_{nm}", "pairwise_{nm}"),
           ("returns a list with the {word} of each number and the number after it (one item fewer than the list).",),
           ("return [{expr} for a, b in zip(numbers, numbers[1:])]",
            "return [numbers[i] {op} numbers[i + 1] for i in range(len(numbers) - 1)]"),
           ({"expr": "a + b", "op": "+", "word": "sum", "nm": "sums"},
            {"expr": "a * b", "op": "*", "word": "product", "nm": "products"})),
    Family("neighbour_steps", "numbers", ("ints",), ("steps_between", "differences"),
           ("returns a list of how much each number differs from the one before it: numbers[1] - numbers[0], "
            "numbers[2] - numbers[1], and so on.",),
           ("return [b - a for a, b in zip(numbers, numbers[1:])]",
            "result = []\nfor i in range(1, len(numbers)):\n    result.append(numbers[i] - numbers[i - 1])\nreturn result")),
    Family("largest_jump", "numbers", ("ints2",), ("largest_jump", "biggest_step"),
           ("returns the largest absolute difference between two neighbouring numbers in a list of at least two numbers.",),
           ("return max(abs(b - a) for a, b in zip(numbers, numbers[1:]))",
            "best = 0\nfor i in range(1, len(numbers)):\n    best = max(best, abs(numbers[i] - numbers[i - 1]))\nreturn best")),
    Family("neighbours_close", "numbers", ("ints",), ("steps_at_most_{kw}", "smooth_by_{kw}"),
           ("returns True if every two neighbouring numbers in the list differ by at most {k} (True for lists shorter than two).",),
           ("return all(abs(b - a) <= {k} for a, b in zip(numbers, numbers[1:]))",
            "for i in range(len(numbers) - 1):\n    if abs(numbers[i + 1] - numbers[i]) > {k}:\n        return False\nreturn True"),
           _k(1, 2, 3, 5)),
    Family("equal_neighbours", "items", ("ints",), ("has_equal_neighbours", "repeats_next_to_each_other"),
           ("returns True if two neighbouring items of the list are equal.",
            "returns True if some item is immediately followed by an equal item."),
           ("return any(a == b for a, b in zip(items, items[1:]))",
            "for i in range(len(items) - 1):\n    if items[i] == items[i + 1]:\n        return True\nreturn False")),
    Family("pairs_summing_above", "numbers", ("ints",), ("neighbour_pairs_above_{k}",),
           ("returns how many pairs of neighbouring numbers add up to more than {k}.",),
           ("return sum(1 for a, b in zip(numbers, numbers[1:]) if a + b > {k})",
            "count = 0\nfor i in range(len(numbers) - 1):\n    if numbers[i] + numbers[i + 1] > {k}:\n        count += 1\nreturn count"),
           _k(0, 5, 10)),
    Family("count_peaks", "numbers", ("ints",), ("count_peaks", "local_maxima"),
           ("returns how many numbers are larger than both of their neighbours (the first and the last number never count).",),
           ("return sum(1 for a, b, c in zip(numbers, numbers[1:], numbers[2:]) if b > a and b > c)",
            "count = 0\nfor i in range(1, len(numbers) - 1):\n    if numbers[i] > numbers[i - 1] and numbers[i] > numbers[i + 1]:\n"
            "        count += 1\nreturn count"), level=2),
    Family("sign_changes", "numbers", ("ints",), ("sign_changes", "count_sign_flips"),
           ("returns how many times a positive number is next to a negative number in the list (zeros never count).",),
           ("return sum(1 for a, b in zip(numbers, numbers[1:]) if a * b < 0)",
            "count = 0\nfor i in range(len(numbers) - 1):\n    if numbers[i] * numbers[i + 1] < 0:\n        count += 1\nreturn count"),
           level=2),
    Family("longest_run", "items", ("ints",), ("longest_run", "longest_streak"),
           ("returns the length of the longest stretch of equal neighbouring items (0 for an empty list).",),
           ("best = 0\nrun = 0\nfor i, x in enumerate(items):\n    run = run + 1 if i > 0 and x == items[i - 1] else 1\n"
            "    best = max(best, run)\nreturn best",
            "if not items:\n    return 0\nbest = run = 1\nfor a, b in zip(items, items[1:]):\n    run = run + 1 if a == b else 1\n"
            "    if run > best:\n        best = run\nreturn best"), level=2),
    # ---- positions
    Family("sum_every_kth_position", "numbers", ("ints",), ("sum_every_{kw}",),
           ("returns the sum of the numbers at positions 0, {k}, 2 * {k}, ... of the list.",),
           ("return sum(numbers[::{k}])", "return sum(x for i, x in enumerate(numbers) if i % {k} == 0)"), _k(3, 4)),
    Family("alternating_sum", "numbers", ("ints",), ("alternating_sum", "plus_minus_sum"),
           ("returns numbers[0] - numbers[1] + numbers[2] - numbers[3] + ... (0 for an empty list).",),
           ("return sum(x if i % 2 == 0 else -x for i, x in enumerate(numbers))",
            "total = 0\nsign = 1\nfor x in numbers:\n    total += sign * x\n    sign = -sign\nreturn total"), level=2),
    Family("positions_compared", "numbers, limit", ("ints", "int"), ("positions_{nm}", "indexes_{nm}"),
           ("returns the positions (starting at 0) of the numbers that are {word} limit.",),
           ("return [i for i, x in enumerate(numbers) if x {op} limit]",
            "result = []\nfor i in range(len(numbers)):\n    if numbers[i] {op} limit:\n        result.append(i)\nreturn result"),
           ({"op": ">", "word": "greater than", "nm": "above"}, {"op": "<", "word": "less than", "nm": "below"},
            {"op": ">=", "word": "at least", "nm": "at_least"})),
    Family("first_position", "numbers", ("ints",), ("first_{nm}_position", "index_of_first_{nm}"),
           ("returns the position of the first {word} number in the list, or -1 if there is none.",),
           ("for i, x in enumerate(numbers):\n    if {cond}:\n        return i\nreturn -1",
            "return next((i for i, x in enumerate(numbers) if {cond}), -1)"),
           tuple({"cond": c, "word": w, "nm": n} for c, w, n in _PRED)),
    Family("last_position", "numbers", ("ints",), ("last_{nm}_position", "index_of_last_{nm}"),
           ("returns the position of the last {word} number in the list, or -1 if there is none.",),
           ("result = -1\nfor i, x in enumerate(numbers):\n    if {cond}:\n        result = i\nreturn result",
            "for i in range(len(numbers) - 1, -1, -1):\n    x = numbers[i]\n    if {cond}:\n        return i\nreturn -1"),
           tuple({"cond": c, "word": w, "nm": n} for c, w, n in _PRED[:4])),
    Family("position_of_extreme", "numbers", ("ints1",), ("position_of_{nm}", "where_is_{nm}"),
           ("returns the position of the {big} number in the non-empty list (the first one if it appears more than once).",),
           ("return numbers.index({fn}(numbers))",
            "best = 0\nfor i in range(1, len(numbers)):\n    if numbers[i] {cmp} numbers[best]:\n        best = i\nreturn best"),
           _MINMAX),
    # ---- building lists
    Family("wrap_with", "items, x", ("ints", "int"), ("wrap_with", "add_at_both_ends"),
           ("returns a new list with x added before the first item and after the last item.",),
           ("return [x] + items + [x]", "result = [x]\nresult.extend(items)\nresult.append(x)\nreturn result")),
    Family("interleave", "a, b", ("ints_eq", "ints_eq"), ("interleave", "zip_together"),
           ("returns a list that alternates the items of the equally long lists a and b: a[0], b[0], a[1], b[1], ...",),
           ("result = []\nfor x, y in zip(a, b):\n    result.append(x)\n    result.append(y)\nreturn result",
            "return [v for pair in zip(a, b) for v in pair]")),
    Family("clip", "numbers, limit", ("ints", "int"), ("clip_{nm}", "{nm}_capped"),
           ("returns a new list in which every number {word} limit is replaced by limit.",),
           ("return [{fn}(x, limit) for x in numbers]",
            "result = []\nfor x in numbers:\n    result.append(limit if x {cmp} limit else x)\nreturn result"),
           ({"fn": "min", "word": "greater than", "nm": "high", "cmp": ">"},
            {"fn": "max", "word": "smaller than", "nm": "low", "cmp": "<"})),
    Family("flip_odd_positions", "numbers", ("ints",), ("flip_odd_positions",),
           ("returns a new list in which the numbers at odd positions (1, 3, 5, ...) have their sign flipped.",),
           ("return [-x if i % 2 == 1 else x for i, x in enumerate(numbers)]",
            "result = list(numbers)\nfor i in range(1, len(result), 2):\n    result[i] = -result[i]\nreturn result")),
    Family("running_total", "numbers", ("ints",), ("running_total", "prefix_sums"),
           ("returns a list whose i-th item is the sum of the first i + 1 numbers of the list.",),
           ("result = []\ntotal = 0\nfor x in numbers:\n    total += x\n    result.append(total)\nreturn result",
            "return [sum(numbers[:i + 1]) for i in range(len(numbers))]"), level=2),
    Family("map_filtered", "numbers", ("ints",), ("{nm}_of_{pw}", "{pw}_{nm}"),
           ("returns the {word} of the {pred} numbers in the list, in order.",),
           ("return [{expr} for x in numbers if {cond}]",
            "result = []\nfor x in numbers:\n    if {cond}:\n        result.append({expr})\nreturn result"),
           tuple({"expr": e, "word": w, "nm": n, "cond": c, "pred": p, "pw": pn}
                 for e, w, n in (("x * x", "squares", "squares"), ("x + 1", "successors", "plus_one"))
                 for c, p, pn in _PRED[:3])),
    Family("pad_list", "items, size", ("ints", "small"), ("pad_with_zeros", "fill_to_size"),
           ("returns the list followed by enough zeros to make it size items long (unchanged if it is already that long).",),
           ("return items + [0] * (size - len(items))",
            "result = list(items)\nwhile len(result) < size:\n    result.append(0)\nreturn result")),
    Family("rotate_left", "items", ("ints",), ("rotate_left_{kw}", "shift_left_{kw}"),
           ("returns the list rotated left by {kw} step(s): each step moves the first item to the end (an empty list stays empty).",),
           ("if not items:\n    return []\nk = {k} % len(items)\nreturn items[k:] + items[:k]",
            "result = list(items)\nfor _ in range({k}):\n    if result:\n        result.append(result.pop(0))\nreturn result"),
           _k(1, 2, 3), level=2),
    Family("chunk_sums", "numbers", ("ints",), ("sums_of_groups_of_{kw}", "chunk_totals_{k}"),
           ("returns the sums of consecutive groups of {kw} numbers of the list (the last group may be shorter).",),
           ("return [sum(numbers[i:i + {k}]) for i in range(0, len(numbers), {k})]",
            "result = []\nfor i in range(0, len(numbers), {k}):\n    result.append(sum(numbers[i:i + {k}]))\nreturn result"),
           _k(2, 3), level=2),
    Family("merge_sorted", "a, b", ("sorted_ints", "sorted_ints"), ("merge_sorted", "combine_sorted"),
           ("returns one sorted list with all the numbers of the two sorted lists a and b.",),
           ("result = []\ni = j = 0\nwhile i < len(a) and j < len(b):\n    if a[i] <= b[j]:\n        result.append(a[i])\n"
            "        i += 1\n    else:\n        result.append(b[j])\n        j += 1\nreturn result + a[i:] + b[j:]",
            "return sorted(a + b)"), level=2),
    # ---- distinct values
    Family("distinct_in_order", "items", ("ints",), ("unique_in_order", "without_repeats"),
           ("returns the items of the list without repeats, keeping the order in which they first appear.",),
           ("result = []\nfor x in items:\n    if x not in result:\n        result.append(x)\nreturn result",
            "return list(dict.fromkeys(items))")),
    Family("sorted_distinct", "items", ("ints",), ("sorted_unique", "distinct_sorted"),
           ("returns the different values in the list, in increasing order.",),
           ("return sorted(set(items))", "result = []\nfor x in sorted(items):\n    if not result or result[-1] != x:\n"
            "        result.append(x)\nreturn result")),
    Family("count_singletons", "items", ("ints",), ("count_singletons", "appear_once"),
           ("returns how many items appear exactly once in the list.",),
           ("return sum(1 for x in items if items.count(x) == 1)", "return len([x for x in items if items.count(x) == 1])")),
    Family("has_duplicates", "items", ("ints",), ("has_duplicates", "any_repeated"),
           ("returns True if some value appears more than once in the list.",),
           ("return len(set(items)) != len(items)",
            "seen = set()\nfor x in items:\n    if x in seen:\n        return True\n    seen.add(x)\nreturn False")),
    Family("common_items", "a, b", ("ints", "ints"), ("in_both", "shared_items"),
           ("returns the items of list a that also appear in list b, in the order of a (repeats in a are kept).",),
           ("return [x for x in a if x in b]", "result = []\nfor x in a:\n    if x in b:\n        result.append(x)\nreturn result")),
    # ---- dictionaries
    Family("frequencies", "items", ("ints",), ("frequencies", "count_each"),
           ("returns a dictionary that maps each item of the list to how many times it appears.",),
           ("counts = {{}}\nfor x in items:\n    counts[x] = counts.get(x, 0) + 1\nreturn counts",
            "return {{x: items.count(x) for x in items}}")),
    Family("most_common", "items", ("ints1",), ("most_common", "mode_of"),
           ("returns the item that appears most often in the non-empty list (of the most frequent ones, the one that comes first).",),
           ("best = items[0]\nfor x in items:\n    if items.count(x) > items.count(best):\n        best = x\nreturn best",
            "return max(items, key=items.count)"), level=2),
    Family("pair_up", "keys, values", ("strs_eq", "ints_eq"), ("pair_up", "to_dict"),
           ("returns a dictionary that maps each key to the value at the same position in values (the lists are equally long; "
            "for a repeated key the last value wins).",),
           ("return dict(zip(keys, values))", "result = {{}}\nfor k, v in zip(keys, values):\n    result[k] = v\nreturn result")),
    Family("sum_values_over", "d, limit", ("dict", "int"), ("sum_values_over", "total_above"),
           ("returns the sum of the values in the dictionary d that are greater than limit.",),
           ("return sum(v for v in d.values() if v > limit)",
            "total = 0\nfor key in d:\n    if d[key] > limit:\n        total += d[key]\nreturn total")),
    Family("key_of_extreme", "d", ("dict1",), ("key_with_{nm}", "{nm}_key"),
           ("returns the key of the non-empty dictionary d whose value is the {big} (the first such key if there is a tie).",),
           ("return {fn}(d, key=d.get)",
            "best = None\nfor k in d:\n    if best is None or d[k] {cmp} d[best]:\n        best = k\nreturn best"),
           _MINMAX),
    Family("invert", "d", ("dict",), ("invert", "swap_keys_and_values"),
           ("returns a new dictionary with the keys and values of d swapped (if values repeat, the last key wins).",),
           ("return {{v: k for k, v in d.items()}}", "result = {{}}\nfor k in d:\n    result[d[k]] = k\nreturn result")),
    Family("word_lengths", "text", ("text",), ("word_lengths", "length_of_each_word"),
           ("returns a dictionary that maps each word of text to its number of characters.",),
           ("return {{w: len(w) for w in text.split()}}", "result = {{}}\nfor w in text.split():\n    result[w] = len(w)\nreturn result")),
    # ---- digits and integers
    Family("digits_above", "n", ("nonneg",), ("digits_above_{k}", "count_big_digits_{k}"),
           ("returns how many digits of the non-negative integer n are greater than {k}.",),
           ("return sum(1 for d in str(n) if int(d) > {k})",
            "count = 0\nwhile True:\n    if n % 10 > {k}:\n        count += 1\n    n //= 10\n    if n == 0:\n        return count"),
           _k(4, 5, 7)),
    Family("extreme_digit", "n", ("nonneg",), ("{nm}_digit", "{big}_digit_of"),
           ("returns the {big} digit of the non-negative integer n.",),
           ("return {fn}(int(d) for d in str(n))", "best = n % 10\nwhile n > 0:\n    best = {fn}(best, n % 10)\n    n //= 10\nreturn best"),
           _MINMAX),
    Family("reverse_digits", "n", ("nonneg",), ("reverse_digits", "digits_backwards"),
           ("returns the number whose digits are those of the non-negative integer n in reverse order (120 gives 21).",),
           ("return int(str(n)[::-1])", "result = 0\nwhile n > 0:\n    result = result * 10 + n % 10\n    n //= 10\nreturn result")),
    Family("digit_product", "n", ("nonneg",), ("digit_product", "product_of_digits"),
           ("returns the product of the digits of the non-negative integer n.",),
           ("result = 1\nfor d in str(n):\n    result *= int(d)\nreturn result",
            "result = n % 10\nn //= 10\nwhile n > 0:\n    result *= n % 10\n    n //= 10\nreturn result")),
    Family("digital_root", "n", ("nonneg",), ("digital_root", "repeated_digit_sum"),
           ("returns the digital root of the non-negative integer n: replace n by the sum of its digits until a single digit is left.",),
           ("while n >= 10:\n    n = sum(int(d) for d in str(n))\nreturn n", "return 0 if n == 0 else 1 + (n - 1) % 9"), level=2),
    Family("round_up_to", "n", ("nonneg",), ("round_up_to_{k}", "next_multiple_of_{k}"),
           ("returns the smallest multiple of {k} that is greater than or equal to the non-negative integer n.",),
           ("return (n + {k} - 1) // {k} * {k}", "while n % {k} != 0:\n    n += 1\nreturn n"), _k(5, 100, 3)),
    Family("divisible_by_both", "n", ("int",), ("divisible_by_{a}_and_{b}",),
           ("returns True if n is divisible by both {a} and {b}.",),
           ("return n % {a} == 0 and n % {b} == 0", "return n % {ab} == 0"),
           ({"a": 2, "b": 3, "ab": 6}, {"a": 3, "b": 5, "ab": 15}, {"a": 4, "b": 5, "ab": 20})),
    Family("count_multiples_below", "n", ("pos",), ("multiples_of_{k}_below",),
           ("returns how many positive integers smaller than the positive integer n are divisible by {k}.",),
           ("return sum(1 for i in range(1, n) if i % {k} == 0)", "return (n - 1) // {k}"), _k(3, 4, 7)),
    Family("clamp", "x, low, high", ("int", "lo", "hi"), ("clamp", "limit_to_range"),
           ("returns x limited to the range from low to high (low <= high): low if x is below it, high if x is above it, otherwise x.",),
           ("return max(low, min(x, high))", "if x < low:\n    return low\nif x > high:\n    return high\nreturn x")),
    Family("total_in_units", "{a}, {b}", ("small", "small"), ("total_{unit}", "{unit}_in_total"),
           ("returns the total number of {unit} in the given number of {a} and {b}.",),
           ("return {a} * {f} + {b}", "total = {a} * {f}\ntotal += {b}\nreturn total"),
           ({"a": "minutes", "b": "seconds", "f": 60, "unit": "seconds"}, {"a": "days", "b": "hours", "f": 24, "unit": "hours"},
            {"a": "weeks", "b": "days", "f": 7, "unit": "days"}, {"a": "feet", "b": "inches", "f": 12, "unit": "inches"})),
    Family("total_three_units", "hours, minutes, seconds", ("small", "small", "small"), ("total_seconds", "to_seconds"),
           ("returns the total number of seconds in the given hours, minutes and seconds.",),
           ("return hours * 3600 + minutes * 60 + seconds", "return (hours * 60 + minutes) * 60 + seconds")),
    # ---- strings and words
    Family("count_char_class", "s", ("str",), ("count_{nm}", "how_many_{nm}"),
           ("returns how many characters of s are {desc}.",),
           ("return sum(1 for c in s if c.{m}())", "count = 0\nfor c in s:\n    if c.{m}():\n        count += 1\nreturn count"),
           _CHARCLASS),
    Family("keep_char_class", "s", ("str",), ("only_{nm}", "keep_{nm}"),
           ("returns a string with only the {desc} of s, in their original order.",),
           ("return ''.join(c for c in s if c.{m}())", "result = ''\nfor c in s:\n    if c.{m}():\n        result += c\nreturn result"),
           _CHARCLASS[:3]),
    Family("initials", "text", ("text",), ("initials", "first_letters"),
           ("returns the first letter of every word in text, in upper case, joined together ('' for no words).",),
           ("return ''.join(w[0].upper() for w in text.split())",
            "result = ''\nfor word in text.split():\n    result += word[0].upper()\nreturn result")),
    Family("long_words", "text", ("text",), ("words_longer_than_{k}", "long_words_{k}"),
           ("returns a list of the words of text that have more than {k} characters, in order.",),
           ("return [w for w in text.split() if len(w) > {k}]",
            "result = []\nfor w in text.split():\n    if len(w) > {k}:\n        result.append(w)\nreturn result"), _k(3, 4)),
    Family("capitalize_words", "text", ("text",), ("capitalize_words", "title_case_words"),
           ("returns the words of text separated by single spaces, each with its first letter in upper case and the rest in lower case.",),
           ("return ' '.join(w.capitalize() for w in text.split())",
            "return ' '.join(w[0].upper() + w[1:].lower() for w in text.split())")),
    Family("join_with", "words", ("strs",), ("join_with_{nm}",),
           ("returns the strings of the list joined into one string with '{sep}' between each two of them.",),
           ("return '{sep}'.join(words)",
            "result = ''\nfor i, w in enumerate(words):\n    if i > 0:\n        result += '{sep}'\n    result += w\nreturn result"),
           ({"sep": "-", "nm": "dashes"}, {"sep": ", ", "nm": "commas"}, {"sep": " / ", "nm": "slashes"})),
    Family("half_of_string", "s", ("str",), ("{nm}_half", "{nm}_half_of"),
           ("returns the {nm} half of s ({rule}).",),
           ("return {expr}", "n = len(s) // 2\nreturn {expr2}"),
           ({"nm": "first", "rule": "for an odd length the middle character is not included", "expr": "s[:len(s) // 2]", "expr2": "s[:n]"},
            {"nm": "second", "rule": "for an odd length the middle character is included", "expr": "s[len(s) // 2:]", "expr2": "s[n:]"})),
    Family("all_same_char", "s", ("str",), ("all_same_character", "one_repeated_character"),
           ("returns True if every character of s is the same character (True for the empty string).",),
           ("return len(set(s)) <= 1", "return all(c == s[0] for c in s)")),
    Family("char_positions", "s, ch", ("str", "char"), ("positions_of_char", "where_char"),
           ("returns the positions (starting at 0) at which the character ch appears in s.",),
           ("return [i for i, c in enumerate(s) if c == ch]",
            "result = []\nfor i in range(len(s)):\n    if s[i] == ch:\n        result.append(i)\nreturn result")),
    Family("mismatches", "a, b", ("str_eq", "str_eq"), ("mismatches", "positions_differing"),
           ("returns at how many positions the equally long strings a and b have different characters.",),
           ("return sum(1 for x, y in zip(a, b) if x != y)",
            "count = 0\nfor i in range(len(a)):\n    if a[i] != b[i]:\n        count += 1\nreturn count")),
    Family("shortest_word_length", "text", ("text1",), ("shortest_word_length", "length_of_shortest_word"),
           ("returns the number of characters of the shortest word in text (text has at least one word).",),
           ("return min(len(w) for w in text.split())", "return len(min(text.split(), key=len))")),
    Family("mask_vowels", "s", ("str",), ("mask_vowels", "hide_vowels"),
           ("returns s with every lower-case vowel (a, e, i, o, u) replaced by '{ch}'.",),
           ("return ''.join('{ch}' if c in 'aeiou' else c for c in s)",
            "result = ''\nfor c in s:\n    result += '{ch}' if c in 'aeiou' else c\nreturn result"),
           ({"ch": "*"}, {"ch": "_"})),
    Family("chain_words", "a, b", ("word0", "word0"), ("chains", "links_up"),
           ("returns True if a and b are non-empty and the last character of a is the first character of b.",),
           ("return a != '' and b != '' and a[-1] == b[0]",
            "if not a or not b:\n    return False\nreturn a[-1] == b[0]")),
    Family("words_containing", "text, letter", ("text", "char"), ("words_containing", "count_words_with"),
           ("returns how many words of text contain the character letter.",),
           ("return sum(1 for w in text.split() if letter in w)",
            "count = 0\nfor w in text.split():\n    if letter in w:\n        count += 1\nreturn count")),
    Family("shift_letters", "s", ("lower_word",), ("shift_by_{k}", "rotate_letters_{k}"),
           ("returns s (lower-case letters only) with every letter moved {k} places forward in the alphabet, wrapping around from z to a.",),
           ("return ''.join(chr((ord(c) - ord('a') + {k}) % 26 + ord('a')) for c in s)",
            "letters = 'abcdefghijklmnopqrstuvwxyz'\nreturn ''.join(letters[(letters.index(c) + {k}) % 26] for c in s)"),
           _k(1, 3, 13), level=2),
    # ---- booleans over ranges and lists
    Family("in_range", "x, low, high", ("int", "lo", "hi"), ("in_range_{nm}", "within_{nm}"),
           ("returns True if x is {word} low and high.",),
           ("return {expr}", "if x {o1} low:\n    return False\nreturn x {o2} high"),
           ({"expr": "low <= x <= high", "word": "between (or equal to)", "nm": "inclusive", "o1": "<", "o2": "<="},
            {"expr": "low < x < high", "word": "strictly between", "nm": "exclusive", "o1": "<=", "o2": "<"})),
    Family("exactly_one_positive", "a, b", ("int", "int"), ("exactly_one_positive",),
           ("returns True if exactly one of a and b is greater than zero.",),
           ("return (a > 0) != (b > 0)", "count = 0\nif a > 0:\n    count += 1\nif b > 0:\n    count += 1\nreturn count == 1")),
    Family("all_in_range", "numbers, low, high", ("ints", "lo", "hi"), ("all_in_range", "every_between"),
           ("returns True if every number of the list is between low and high, inclusive (True for an empty list).",),
           ("return all(low <= x <= high for x in numbers)",
            "for x in numbers:\n    if x < low or x > high:\n        return False\nreturn True")),
    # ---- two lists
    Family("pairwise_extreme", "a, b", ("ints_eq", "ints_eq"), ("pairwise_{nm}", "elementwise_{nm}"),
           ("returns a list with the {big} of a[i] and b[i] at each position i (a and b are equally long).",),
           ("return [{fn}(x, y) for x, y in zip(a, b)]",
            "result = []\nfor i in range(len(a)):\n    result.append(a[i] if a[i] {cmp} b[i] else b[i])\nreturn result"),
           _MINMAX),
    Family("dot_product", "a, b", ("ints_eq", "ints_eq"), ("dot_product", "sum_of_products"),
           ("returns the sum of a[i] * b[i] over all positions i of the equally long lists a and b.",),
           ("return sum(x * y for x, y in zip(a, b))", "total = 0\nfor i in range(len(a)):\n    total += a[i] * b[i]\nreturn total")),
    Family("count_matches", "a, b", ("ints_eq", "ints_eq"), ("count_matches", "same_position_count"),
           ("returns at how many positions the equally long lists a and b hold equal values.",),
           ("return sum(1 for x, y in zip(a, b) if x == y)",
            "count = 0\nfor i in range(len(a)):\n    if a[i] == b[i]:\n        count += 1\nreturn count")),
) + (
    # ---- multi-step algorithms (curriculum v2): state, nested loops, parsing, grids
    Family("balanced_parens", "s", ("brackets",), ("parens_balanced", "brackets_match"),
           ("returns True if every '(' in s is closed by a later ')' and every ')' closes an earlier '(' "
            "(other characters are ignored).",),
           ("depth = 0\nfor c in s:\n    if c == '(':\n        depth += 1\n    elif c == ')':\n        depth -= 1\n"
            "        if depth < 0:\n            return False\nreturn depth == 0",
            "stack = []\nfor c in s:\n    if c == '(':\n        stack.append(c)\n    elif c == ')':\n        if not stack:\n"
            "            return False\n        stack.pop()\nreturn not stack"), level=3),
    Family("run_lengths", "s", ("runs",), ("run_lengths", "encode_runs"),
           ("returns a list of [character, count] pairs, one for each run of equal neighbouring characters in s, in order.",),
           ("result = []\nfor c in s:\n    if result and result[-1][0] == c:\n        result[-1][1] += 1\n    else:\n"
            "        result.append([c, 1])\nreturn result",
            "result = []\ni = 0\nwhile i < len(s):\n    j = i\n    while j < len(s) and s[j] == s[i]:\n        j += 1\n"
            "    result.append([s[i], j - i])\n    i = j\nreturn result"), level=3),
    Family("group_by_length", "words", ("strs",), ("group_by_length", "words_by_length"),
           ("returns a dictionary that maps each word length to the list of words of that length, in their original order.",),
           ("groups = {{}}\nfor w in words:\n    groups.setdefault(len(w), []).append(w)\nreturn groups",
            "groups = {{}}\nfor w in words:\n    if len(w) not in groups:\n        groups[len(w)] = []\n    groups[len(w)].append(w)\n"
            "return groups"), level=2),
    Family("transpose", "rows", ("matrix",), ("transpose", "columns_as_rows"),
           ("returns the columns of the rectangular grid rows (a list of equally long lists) as a list of lists.",),
           ("return [list(col) for col in zip(*rows)]",
            "if not rows:\n    return []\nreturn [[row[j] for row in rows] for j in range(len(rows[0]))]"), level=2),
    Family("row_extremes", "rows", ("matrix1",), ("row_{nm}s", "{nm}_of_each_row"),
           ("returns a list with the {big} number of each row of the grid rows (every row is non-empty).",),
           ("return [{fn}(row) for row in rows]",
            "result = []\nfor row in rows:\n    best = row[0]\n    for x in row:\n        if x {cmp} best:\n            best = x\n"
            "    result.append(best)\nreturn result"), _MINMAX, level=2),
    Family("diagonal_sum", "grid", ("square",), ("diagonal_sum", "trace"),
           ("returns the sum of the numbers on the main diagonal of the square grid (grid[0][0] + grid[1][1] + ...).",),
           ("return sum(grid[i][i] for i in range(len(grid)))",
            "total = 0\nfor i, row in enumerate(grid):\n    total += row[i]\nreturn total"), level=2),
    Family("best_run_sum", "numbers", ("ints1",), ("best_run_sum", "max_consecutive_sum"),
           ("returns the largest sum of one or more consecutive numbers of the non-empty list.",),
           ("best = current = numbers[0]\nfor x in numbers[1:]:\n    current = max(x, current + x)\n    best = max(best, current)\n"
            "return best",
            "best = numbers[0]\nfor i in range(len(numbers)):\n    for j in range(i + 1, len(numbers) + 1):\n"
            "        best = max(best, sum(numbers[i:j]))\nreturn best"), level=3),
    Family("pair_with_sum", "numbers, target", ("ints", "int_small"), ("has_pair_with_sum", "two_add_up_to"),
           ("returns True if two numbers at different positions in the list add up to target.",),
           ("for i in range(len(numbers)):\n    for j in range(i + 1, len(numbers)):\n        if numbers[i] + numbers[j] == target:\n"
            "            return True\nreturn False",
            "seen = set()\nfor x in numbers:\n    if target - x in seen:\n        return True\n    seen.add(x)\nreturn False"),
           level=3),
    Family("search_sorted", "numbers, x", ("sorted_distinct", "int_small"), ("search_sorted", "find_in_sorted"),
           ("returns the position of x in the sorted list numbers (no number appears twice), or -1 if x is not in it.",),
           ("low, high = 0, len(numbers) - 1\nwhile low <= high:\n    mid = (low + high) // 2\n    if numbers[mid] == x:\n"
            "        return mid\n    if numbers[mid] < x:\n        low = mid + 1\n    else:\n        high = mid - 1\nreturn -1",
            "return numbers.index(x) if x in numbers else -1"), level=3),
    Family("sum_numbers_in_text", "text", ("mixed_text",), ("sum_numbers_in_text", "add_up_numbers_in"),
           ("returns the sum of the whole numbers written in text, where a number is a run of digits (0 if there is none).",),
           ("total = 0\ncurrent = ''\nfor c in text + ' ':\n    if c.isdigit():\n        current += c\n    elif current:\n"
            "        total += int(current)\n        current = ''\nreturn total",
            "import re\nreturn sum(int(n) for n in re.findall(r'[0-9]+', text))"), level=3),
    Family("count_inversions", "numbers", ("ints",), ("count_inversions", "out_of_order_pairs"),
           ("returns how many pairs of positions i < j have numbers[i] > numbers[j].",),
           ("count = 0\nfor i in range(len(numbers)):\n    for j in range(i + 1, len(numbers)):\n        if numbers[i] > numbers[j]:\n"
            "            count += 1\nreturn count",
            "return sum(1 for i, a in enumerate(numbers) for b in numbers[i + 1:] if a > b)"), level=3),
    Family("before_first_negative", "numbers", ("ints",), ("before_first_negative", "until_negative"),
           ("returns the numbers that come before the first negative number of the list (all of them if there is none).",),
           ("result = []\nfor x in numbers:\n    if x < 0:\n        break\n    result.append(x)\nreturn result",
            "for i, x in enumerate(numbers):\n    if x < 0:\n        return numbers[:i]\nreturn list(numbers)"), level=2),
    Family("first_over_running", "numbers, limit", ("ints", "int"), ("when_total_exceeds", "first_position_over_total"),
           ("returns the first position at which the running total of the numbers (from the start up to and including that "
            "position) is greater than limit, or -1 if it never is.",),
           ("total = 0\nfor i, x in enumerate(numbers):\n    total += x\n    if total > limit:\n        return i\nreturn -1",
            "for i in range(len(numbers)):\n    if sum(numbers[:i + 1]) > limit:\n        return i\nreturn -1"), level=3),
    Family("digit_names", "n", ("nonneg",), ("digit_names", "spell_digits"),
           ("returns the English names of the digits of the non-negative integer n separated by single spaces (405 gives 'four zero five').",),
           ("names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']\n"
            "return ' '.join(names[int(d)] for d in str(n))",
            "names = {{'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven', "
            "'8': 'eight', '9': 'nine'}}\nwords = []\nfor d in str(n):\n    words.append(names[d])\nreturn ' '.join(words)"), level=2),
    Family("common_prefix", "a, b", ("word0", "word0"), ("common_prefix", "shared_start"),
           ("returns the longest string that both a and b start with.",),
           ("i = 0\nwhile i < len(a) and i < len(b) and a[i] == b[i]:\n    i += 1\nreturn a[:i]",
            "result = ''\nfor x, y in zip(a, b):\n    if x != y:\n        break\n    result += x\nreturn result"), level=2),
    Family("collapse_runs", "items", ("runs_list",), ("collapse_runs", "squeeze_repeats"),
           ("returns the list with every run of equal neighbouring items replaced by a single item.",),
           ("result = []\nfor x in items:\n    if not result or result[-1] != x:\n        result.append(x)\nreturn result",
            "return [x for i, x in enumerate(items) if i == 0 or items[i - 1] != x]"), level=2),
    Family("star_bars", "numbers", ("small_list",), ("star_bars", "histogram_lines"),
           ("returns a string with one line per number of the list, each line made of that many '{ch}' characters, "
            "lines separated by newlines.",),
           ("return '\\n'.join('{ch}' * n for n in numbers)",
            "lines = []\nfor n in numbers:\n    lines.append('{ch}' * n)\nreturn '\\n'.join(lines)"), ({"ch": "*"}, {"ch": "#"}), level=2),
    Family("evens_then_odds", "numbers", ("ints",), ("evens_then_odds", "split_by_parity"),
           ("returns the even numbers of the list followed by the odd ones, each group in its original order.",),
           ("return [x for x in numbers if x % 2 == 0] + [x for x in numbers if x % 2 != 0]",
            "evens, odds = [], []\nfor x in numbers:\n    (evens if x % 2 == 0 else odds).append(x)\nreturn evens + odds"), level=2),
    Family("letter_counts", "text", ("text",), ("first_letter_counts", "count_by_first_letter"),
           ("returns a dictionary that maps each lower-cased first letter of the words of text to how many words start with it.",),
           ("counts = {{}}\nfor w in text.split():\n    k = w[0].lower()\n    counts[k] = counts.get(k, 0) + 1\nreturn counts",
            "counts = {{}}\nfor w in text.split():\n    if w[0].lower() in counts:\n        counts[w[0].lower()] += 1\n    else:\n"
            "        counts[w[0].lower()] = 1\nreturn counts"), level=2),
)

FAMILIES_V1 = frozenset(f.key for f in FAMILIES[:76])  # idioms-v1 (V6 iterations 1-2)
FAMILIES_V2 = frozenset(f.key for f in FAMILIES)  # idioms-v2 (V6 iteration 3)


def _with_extra() -> tuple[Family, ...]:
    from ycode.lm.idioms_extra import EXTRA_FAMILIES

    return FAMILIES + EXTRA_FAMILIES


FAMILIES = _with_extra()

# ------------------------------------------------------------------ inputs

_WORDS = ("apple", "Banana", "cat", "dog", "Echo", "fig", "grape", "Hat", "ice", "jam", "kiwi", "Lemon", "moon", "nut",
          "owl", "Pear", "queen", "rose", "sun", "tree", "Umbrella", "vine", "wolf", "x", "yes", "zoo", "a", "I", "sky")


def _value(kind: str, rng: random.Random, size: int | None = None):
    if kind in ("ints", "ints_eq", "ints1", "ints2", "sorted_ints"):
        lo = {"ints1": 1, "ints2": 2}.get(kind, 0)
        n = size if size is not None else rng.randint(lo, 7)
        values = [rng.randint(-9, 20) for _ in range(n)]
        return sorted(values) if kind == "sorted_ints" else values
    if kind in ("int", "lo", "hi"):
        return rng.randint(-30, 60)
    if kind == "small":
        return rng.randint(0, 12)
    if kind == "pos":
        return rng.randint(1, 60)
    if kind == "nonneg":
        return int("".join(rng.choice("0123456789") for _ in range(rng.randint(1, 5))))
    if kind in ("str", "str_eq"):
        n = size if size is not None else rng.randint(0, 10)
        return "".join(rng.choice("abceiouXYZ 019!") for _ in range(n))
    if kind == "lower_word":
        return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(0, 8)))
    if kind == "word0":
        return "".join(rng.choice("abcde") for _ in range(rng.randint(0, 4)))
    if kind in ("text", "text1"):
        return " ".join(rng.choice(_WORDS) for _ in range(rng.randint(1 if kind == "text1" else 0, 5)))
    if kind in ("strs", "strs_eq"):
        return [rng.choice(_WORDS) for _ in range(size if size is not None else rng.randint(0, 5))]
    if kind == "char":
        return rng.choice("aeiozX 1")
    if kind == "int_small":
        return rng.randint(-5, 25)
    if kind == "brackets":
        return "".join(rng.choice("(()) ab") for _ in range(rng.randint(0, 9)))
    if kind == "runs":
        return "".join(rng.choice("aab") * rng.randint(1, 3) for _ in range(rng.randint(0, 4)))
    if kind == "runs_list":
        return [x for _ in range(rng.randint(0, 4)) for x in [rng.randint(0, 3)] * rng.randint(1, 3)]
    if kind in ("matrix", "matrix1", "square"):
        n = rng.randint(1 if kind != "matrix" else 0, 4)
        m = n if kind == "square" else rng.randint(1, 4)
        return [[rng.randint(-5, 9) for _ in range(m)] for _ in range(n)]
    if kind == "sorted_distinct":
        return sorted(rng.sample(range(-5, 25), rng.randint(0, 7)))
    if kind == "mixed_text":
        return "".join(rng.choice(["ab", " ", "7", "12", "x", "305", ".", "0"]) for _ in range(rng.randint(0, 7)))
    if kind == "small_list":
        return [rng.randint(0, 5) for _ in range(rng.randint(0, 4))]
    if kind in ("dict", "dict1"):
        keys = rng.sample(_WORDS, rng.randint(1 if kind == "dict1" else 0, 5))
        return {k: rng.randint(-5, 20) for k in keys}
    raise ValueError(kind)


def make_inputs(kinds: tuple[str, ...], rng: random.Random, n: int = 40) -> list[tuple]:
    out = []
    for _ in range(n * 3):
        if len(out) >= n:
            break
        size = rng.randint(0, 6)
        args = [_value(k, rng, size if k.endswith("_eq") else None) for k in kinds]
        if "lo" in kinds and "hi" in kinds:
            i, j = kinds.index("lo"), kinds.index("hi")
            if args[i] > args[j]:
                args[i], args[j] = args[j], args[i]
        if tuple(args) not in out:
            out.append(tuple(args))
    return out


# ------------------------------------------------------------------ programs

class _Timeout(Exception):
    pass


def _run(source: str, name: str, inputs: list[tuple]) -> list | None:
    """Outputs of the function on every input, or None if it fails to compile, raises or is slow."""
    ns: dict = {}

    def alarm(*_):
        raise _Timeout

    old = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, 2.0)
    try:
        exec(source, ns)
        return [ns[name](*copy.deepcopy(args)) for args in inputs]
    except Exception:  # noqa: BLE001 - any failure disqualifies the program
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _fill(template: str, v: dict) -> str:
    return template.format(**v)


FAMILY_SETS = ("v1", "v2", "v3")  # each a prefix of FAMILIES, so a pinned set reproduces its programs exactly


def families(family_set: str = "v3") -> tuple[Family, ...]:
    keys = {"v1": FAMILIES_V1, "v2": FAMILIES_V2}.get(family_set)
    if family_set not in FAMILY_SETS:
        raise ValueError(f"unknown family set {family_set!r}")
    return FAMILIES if keys is None else tuple(f for f in FAMILIES if f.key in keys)


def candidate_programs(seed: int = 0, family_list: tuple[Family, ...] | None = None) -> list[Program]:
    """Every program of every family, before the held-out checks; each verified by execution."""
    rng = random.Random(seed)
    out = []
    for fam in FAMILIES if family_list is None else family_list:
        for v in fam.variants:
            params = _fill(fam.params, v)
            names = tuple(dict.fromkeys(_fill(n, v) for n in fam.names))
            bodies = tuple(_fill(b, v) for b in fam.bodies)
            tasks = tuple(_fill(t, v) for t in fam.tasks)
            prog = Program(fam.key, names[0], params, tasks, bodies, fam.level)
            inputs = make_inputs(fam.kinds, rng)
            results = [_run(prog.source(i), prog.name, inputs) for i in range(len(bodies))]
            if any(r is None for r in results):
                raise AssertionError(f"{fam.key} {v}: an implementation fails on its own domain")
            if any(repr(r) != repr(results[0]) for r in results[1:]):
                raise AssertionError(f"{fam.key} {v}: implementations disagree")
            if len({repr(o) for o in results[0]}) < 2:
                raise AssertionError(f"{fam.key} {v}: constant output")
            out.append(Program(fam.key, names[0], params, tasks, bodies, fam.level, tuple(inputs)))
            for alias in names[1:]:  # aliases are the same program under another name
                out.append(Program(fam.key, alias, params, tasks, bodies, fam.level, tuple(inputs)))
    return out


def held_out_check(programs: list[Program]) -> tuple[list[Program], dict[str, str]]:
    """(kept programs, {name: reason} for dropped ones): held-out name, skeleton or behaviour."""
    from ycode.lm.basics import held_out_names
    from ycode.lm.leakage import behaviour_leaks, held_out_items, held_out_skeletons, skeleton

    items = held_out_items()
    names = held_out_names() | {i.name for i in items}
    skeletons = held_out_skeletons(items)
    kept, dropped = [], {}
    behaviour: dict[str, list] = {}
    for p in programs:
        if p.name in names:
            dropped[p.name] = "held-out name"
            continue
        twins = [t for i in range(len(p.bodies)) for t in skeletons.get(skeleton(p.source(i)) or "", [])]
        if twins:
            dropped[p.name] = f"skeleton of {twins[0]}"
            continue
        key = p.source(0, "f")
        if key not in behaviour:
            arity = len([x for x in p.params.split(",") if x.strip()])
            behaviour[key] = behaviour_leaks({p.source(0): (p.name, arity)}, items, near=1.0)
        if behaviour[key]:
            dropped[p.name] = f"computes {behaviour[key][0].split}:{behaviour[key][0].item}"
            continue
        kept.append(p)
    return kept, dropped


_TEMPLATES = (
    (0.6, "Write a Python function `{sig}` that {task}"),
    (0.15, "Implement `{sig}`: it {task}"),
    (0.15, "Write `{sig}`, a function that {task}"),
    (0.1, "Complete this function so that it {task}\n```python\ndef {sig}:\n    ...\n```"),
)


def _prompt(sig: str, task: str, rng: random.Random) -> str:
    roll, acc = rng.random(), 0.0
    for p, template in _TEMPLATES:
        acc += p
        if roll < acc:
            return template.format(sig=sig, task=task)
    return _TEMPLATES[0][1].format(sig=sig, task=task)


def _doc(task: str) -> str:
    """The task as a docstring sentence: 'returns the sum ...' -> 'Return the sum ...'."""
    first, _, rest = task.partition(" ")
    verb = first[:-1] if first.endswith("s") and first not in ("is", "has") else first
    return f"{verb[0].upper()}{verb[1:]} {rest}".strip()


def _documented(p: Program, body: str) -> str:
    lines = body.splitlines()
    return (f"def {p.name}({p.params}):\n    \"\"\"{_doc(p.tasks[0])}\"\"\"\n"
            + "\n".join("    " + line for line in lines))


def _examples_sentence(p: Program, rng: random.Random) -> str | None:
    """'For example, `f([])` returns `0` and `f([3, 1])` returns `4`.': the smallest input (a boundary
    case: empty, zero, single item) and a typical one, with outputs from the reference."""
    inputs = sorted(p.inputs, key=lambda a: len(repr(a)))
    if len(inputs) < 4:
        return None
    chosen = [inputs[0], inputs[rng.randrange(len(inputs) // 2, len(inputs))]]
    outputs = _run(p.source(0), p.name, chosen)
    if outputs is None or any(len(repr(o)) > 60 for o in outputs) or any(len(repr(a)) > 60 for a in chosen):
        return None
    parts = [f"`{p.name}({', '.join(repr(x) for x in args)})` returns `{out!r}`" for args, out in zip(chosen, outputs)]
    return "For example, " + " and ".join(parts) + "."


def _sibling_repairs(programs: list[Program], rng: random.Random, per_program: int) -> list[tuple[Program, str, int]]:
    """(program, wrong source, implementation of the fix) triples: the program's docstring and
    signature with the body of another program that takes the same parameters (a sibling of the same
    family first: another constant or condition, written with the same implementation, so the fix
    changes exactly what the docstring decides; else another family's idiom). The wrong code must
    run on the program's inputs and give a different result on at least one: plausible, executable
    and wrong."""
    by_params: dict[str, list[Program]] = {}
    for q in programs:
        by_params.setdefault(q.params, []).append(q)
    out = []
    for p in programs:
        expected = _run(p.source(0), p.name, list(p.inputs))
        others = [q for q in by_params[p.params] if q.source(0, "f") != p.source(0, "f")]
        same = [q for q in others if q.family == p.family]
        rng.shuffle(same)
        rest = [q for q in others if q.family != p.family]
        rng.shuffle(rest)
        made = 0
        for q in same + rest:
            if made >= per_program:
                break
            impl = rng.randrange(len(q.bodies))
            fix = impl if q.family == p.family else rng.randrange(len(p.bodies))
            wrong = _documented(p, q.bodies[impl])
            got = _run(wrong, p.name, list(p.inputs))
            if got is None or repr(got) == repr(expected):
                continue
            out.append((p, wrong, fix))
            made += 1
    return out


def idiom_examples(seed: int, *, write_per_program: int = 2, bugfix_per_program: int = 1, contrast_per_program: int = 0,
                   io_fraction: float = 0.0, programs: list[Program] | None = None,
                   family_set: str = "v3") -> list[InstructionExample]:
    """Write-a-function examples (each with a random phrasing, implementation and prompt template;
    with probability ``io_fraction`` followed by worked input-output examples, one of them a
    boundary case), find-the-bug examples (a realistic bug injected into one implementation, kept
    only if it changes the output on the program's inputs) and, with ``contrast_per_program``,
    plausible-but-wrong repairs (the task's docstring over a sibling program's body; the answer is
    the program) for every held-out-clean program."""
    from ycode.lm.basics import BUGFIX_PROMPT, neutralize

    rng = random.Random(seed)
    if programs is None:
        programs, _ = held_out_check(candidate_programs(family_list=families(family_set)))
    out = []
    for p, wrong, fix in _sibling_repairs(programs, rng, contrast_per_program) if contrast_per_program else []:
        fixed = _documented(p, p.bodies[fix])
        out.append(neutralize(InstructionExample(BUGFIX_PROMPT.format(code=wrong), f"```python\n{fixed}\n```"),
                              p.name, rng, 0.3))
    for p in programs:
        sig = f"{p.name}({p.params})"
        for _ in range(write_per_program):
            body = rng.randrange(len(p.bodies))
            task = rng.choice(p.tasks)
            if io_fraction and rng.random() < io_fraction and (example := _examples_sentence(p, rng)):
                task = f"{task} {example}"
            ex = InstructionExample(_prompt(sig, task, rng), f"```python\n{p.source(body)}\n```")
            out.append(neutralize(ex, p.name, rng, 0.3))
        made = 0
        for _ in range(bugfix_per_program * 6):
            if made >= bugfix_per_program:
                break
            body = rng.randrange(len(p.bodies))
            fixed = p.source(body)
            bug = inject_bug(fixed, rng)
            if bug is None:
                continue
            buggy = bug[0]
            expected = _run(fixed, p.name, list(p.inputs))
            got = _run(buggy, p.name, list(p.inputs))
            if got is not None and repr(got) == repr(expected):
                continue  # the mutation does not change behaviour on the domain: not a bug
            ex = InstructionExample(BUGFIX_PROMPT.format(code=buggy), f"```python\n{fixed}\n```")
            out.append(neutralize(ex, p.name, rng, 0.3))
            made += 1
    rng.shuffle(out)
    return out
