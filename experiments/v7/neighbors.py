"""V7 diagnostic development set: tasks next to functions the model was trained on.

Two balanced groups (written for V7 from the training catalogue only; no held-out set was read):

* ``same_wording``: the prompt is worded like a trained function's prompt but one detail changes the
  implementation (the sixth item instead of the fifth, ``>=`` instead of ``>``, the last tie instead
  of the first). ``neighbour`` is that trained function; ``wrong`` is its behaviour under the task's
  signature. Retrieving the familiar function fails the tests.
* ``new_wording``: the implementation is a trained function's (``same_as``), but the name and the
  wording are new. ``neighbour`` is a trained function whose description is plausible next to the
  task but which computes something else (``wrong``).

Every task has a reference, test inputs (the expected values are the reference's outputs, so the
tests check behaviour), two example inputs that are not test inputs, and a one-sentence behavioural
constraint. ``prompts`` renders the four V7 prompt variants:

* ``original``: "Write a Python function `name(params)` that <task>", the dev set's format;
* ``io``: + "For example, `call` returns `value` and `call` returns `value`.";
* ``constraint``: + the constraint sentence;
* ``distractor``: + "The same module also has `neighbour(params)`, which <its task>" (the trained
  neighbour's own training description: irrelevant to the task, plausible next to it).

``checked_tasks`` verifies all of this by execution and drops any task that is a trained function
(same_wording), is not its trained twin (new_wording), or matches a benchmark or unseen-v2 item by
name, behaviour or code skeleton. The holdout check reports counts only.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

VERSION = "neighbours-v1"
VARIANTS = ("original", "io", "constraint", "distractor")


@dataclass(frozen=True)
class Task:
    group: str
    name: str
    params: str
    task: str
    body: str
    wrong_body: str  # the neighbour's behaviour under this task's signature
    neighbour: str  # a trained function (ycode training catalogue)
    tests: tuple  # argument tuples
    examples: tuple  # two argument tuples, not among the tests
    constraint: str
    same_as: str | None = None  # new_wording: the trained function with this implementation

    def source(self, body: str | None = None) -> str:
        lines = (body or self.body).strip("\n").splitlines()
        return f"def {self.name}({self.params}):\n" + "\n".join("    " + line for line in lines)

    @property
    def wrong_source(self) -> str:
        return self.source(self.wrong_body)


def _t(group, name, params, task, body, wrong_body, neighbour, tests, examples, constraint, same_as=None):
    return Task(group, name, params, task, body, wrong_body, neighbour, tuple(tests), tuple(examples),
                constraint, same_as)


L7 = [3, 8, 1, 9, 4, 7, 2]
SW = "same_wording"
NW = "new_wording"

TASKS: tuple[Task, ...] = (
    # ---- same wording, different implementation (neighbour: the trained function it resembles)
    _t(SW, "sixth_item", "items", "returns the sixth item of the list.", "return items[5]", "return items[4]",
       "fifth_item", [(L7,), (list("abcdefgh"),), ([0, 0, 0, 0, 1, 2],), ([10, 20, 30, 40, 50, 60, 70, 80],),
                      ([True, False, True, False, True, False],)],
       [([5, 6, 7, 8, 9, 10, 11],), (["a", "b", "c", "d", "e", "f"],)],
       "The list has at least six items, and the first item counts as item one."),
    _t(SW, "third_to_last_item", "items", "returns the third to last item of the list.", "return items[-3]",
       "return items[-1]", "last_item", [(L7,), (list("abcde"),), ([1, 2, 3],), ([9, 8, 7, 6],), ([0, 5, 0, 5, 0],)],
       [([4, 5, 6, 7, 8],), (["x", "y", "z"],)],
       "The list has at least three items, and the last item is the first from the end."),
    _t(SW, "fifth_word", "text", "returns the fifth word of text.", "return text.split()[4]",
       "return text.split()[3]", "fourth_word",
       [("one two three four five six",), ("a b c d e",), ("the quick brown fox jumps over",),
        ("x  y  z  w  v  u",), ("red green blue cyan pink gray",)],
       [("we will meet again at noon",), ("p q r s t",)],
       "Words are separated by spaces, and text has at least five words."),
    _t(SW, "first_four_chars", "s", "returns the first four characters of s.", "return s[:4]", "return s[:5]",
       "first_five_chars", [("abcdefg",), ("python",), ("ab",), ("12345",), ("four",)],
       [("keyboard",), ("hello world",)],
       "If s has fewer than four characters, the whole of s is returned."),
    _t(SW, "last_two_chars", "s", "returns the last two characters of s.", "return s[-2:]", "return s[:2]",
       "first_two_chars", [("abcdef",), ("python",), ("xy",), ("hello",), ("a",)],
       [("garden",), ("moon",)],
       "The characters come from the end of s; if s is shorter than two characters, all of s is returned."),
    _t(SW, "has_four_items", "items", "returns True if the list has exactly 4 items.", "return len(items) == 4",
       "return len(items) == 5", "has_five_items",
       [([1, 2, 3, 4],), ([1, 2, 3, 4, 5],), ([],), (list("abc"),), (list("wxyz"),), ([0] * 6,)],
       [([7, 8, 9, 10],), ([1, 2],)],
       "Lists of three or five items give False."),
    _t(SW, "longer_than_four", "items", "returns True if the list has more than 4 items.", "return len(items) > 4",
       "return len(items) > 5", "longer_than_five",
       [([1, 2, 3, 4, 5],), ([1, 2, 3, 4],), ([0] * 6,), ([],), (list("abcde"),)],
       [(list("abcdefgh"),), ([1, 2, 3],)],
       "A list of exactly four items gives False and one of five items gives True."),
    _t(SW, "count_at_least", "numbers, limit", "returns how many numbers are greater than or equal to limit.",
       "return sum(1 for x in numbers if x >= limit)", "return sum(1 for x in numbers if x > limit)",
       "count_greater", [([1, 5, 3, 5, 7], 5), ([], 3), ([2, 2, 2], 2), ([-1, 0, 1], 0), ([10, 20, 30], 25)],
       [([4, 6, 8, 6], 6), ([1, 2, 3], 4)],
       "Numbers equal to limit are counted too."),
    _t(SW, "product_of_positives", "numbers", "returns the product of the positive numbers in the list.",
       "result = 1\nfor x in numbers:\n    if x > 0:\n        result *= x\nreturn result",
       "return sum(x for x in numbers if x > 0)", "sum_positive",
       [([2, -3, 4],), ([],), ([-1, -2],), ([1, 5, 0, 3],), ([7],)],
       [([3, 3, -5],), ([2, 10],)],
       "The result is 1 when no number in the list is positive."),
    _t(SW, "last_index_of_max", "numbers", "returns the index of the last occurrence of the largest number in a non-empty list.",
       "return len(numbers) - 1 - numbers[::-1].index(max(numbers))", "return numbers.index(max(numbers))",
       "index_of_max", [([1, 5, 2, 5],), ([3],), ([9, 1, 9, 9, 0],), ([4, 3, 2],), ([-2, -2],)],
       [([7, 2, 7],), ([1, 8, 3, 8, 8],)],
       "When the largest number occurs more than once, the index of its last occurrence is returned."),
    _t(SW, "shortest_word_last", "text", "returns the shortest word in text (the last one if there is a tie).",
       "best = None\nfor w in text.split():\n    if best is None or len(w) <= len(best):\n        best = w\nreturn best",
       "return min(text.split(), key=len)", "shortest_word",
       [("an apple is a fruit",), ("cat dog",), ("hello",), ("to be or not",), ("big red box",)],
       [("we go up",), ("one of my",)],
       "If several words share the shortest length, the one that comes later in text wins."),
    _t(SW, "remove_dashes", "s", "returns s with all dashes removed.", "return s.replace('-', '')",
       "return s.replace(' ', '')", "remove_spaces",
       [("a-b-c",), ("no dashes here",), ("--x--",), ("well - known",), ("",)],
       [("2024-01-31",), ("up - down",)],
       "Only '-' characters are removed; spaces stay."),
    _t(SW, "ends_with_ly", "s", "returns True if the string s ends with 'ly'.", "return s.endswith('ly')",
       "return s.endswith('ed')", "ends_with_ed",
       [("quickly",), ("played",), ("ly",), ("lyric",), ("",), ("fly",)],
       [("slowly",), ("jumped",)],
       "Only a final 'ly' counts; 'ly' elsewhere in s does not."),
    _t(SW, "drop_last", "items, n", "returns the list without its last n items.", "return items[:len(items) - n]",
       "return items[n:]", "drop", [([1, 2, 3, 4, 5], 2), ([1, 2, 3], 0), (list("abcd"), 1), ([7, 8], 2), ([9, 8, 7, 6], 3)],
       [([10, 20, 30, 40], 1), (list("xyz"), 2)],
       "n is between 0 and the length of the list; with n = 0 the whole list is returned."),
    _t(SW, "every_third", "items", "returns every third item of the list, starting with the first.", "return items[::3]",
       "return items[::2]", "every_other", [(list(range(10)),), ([1, 2],), ([],), (list("abcdefg"),), ([5, 6, 7],)],
       [(list(range(20, 27)),), (list("pqrs"),)],
       "The kept items are those at positions 0, 3, 6 and so on."),
    _t(SW, "rotate_right", "items, k", "returns the list rotated k positions to the right.",
       "if not items:\n    return []\nk %= len(items)\nreturn items[-k:] + items[:-k] if k else list(items)",
       "if not items:\n    return []\nk %= len(items)\nreturn items[k:] + items[:k]", "rotate_left",
       [([1, 2, 3, 4, 5], 1), ([1, 2, 3, 4, 5], 2), (list("abc"), 0), ([7, 8, 9], 3), ([1, 2, 3, 4], 3)],
       [([10, 20, 30, 40], 1), (list("wxyz"), 2)],
       "The last k items move to the front, in order."),
    _t(SW, "is_strictly_between", "x, low, high", "returns True if x is strictly between low and high.",
       "return low < x < high", "return low <= x <= high", "is_between",
       [(5, 1, 10), (1, 1, 10), (10, 1, 10), (0, 1, 10), (3, 2, 4), (-1, -5, 0)],
       [(7, 7, 9), (8, 7, 9)],
       "x equal to low or to high gives False."),
    _t(SW, "keep_at_most", "numbers, limit", "returns a list of the numbers less than or equal to limit.",
       "return [x for x in numbers if x <= limit]", "return [x for x in numbers if x > limit]", "greater_than",
       [([1, 5, 3, 7], 4), ([], 2), ([2, 2, 3], 2), ([-1, 0, 1], 0), ([10, 20], 5)],
       [([6, 1, 9, 4], 5), ([3, 3, 8], 3)],
       "Numbers equal to limit are kept."),
    _t(SW, "hours_to_weeks", "hours", "converts hours to weeks.", "return hours / 168", "return hours / 24",
       "hours_to_days", [(168,), (0,), (84,), (336,), (504,)],
       [(1680,), (42,)],
       "A week has 168 hours."),
    _t(SW, "capitalize_last", "s", "returns s with its last character in upper case.", "return s[:-1] + s[-1:].upper()",
       "return s[:1].upper() + s[1:]", "capitalize_first", [("hello",), ("a",), ("",), ("abc def",), ("Xy",)],
       [("python",), ("go",)],
       "Only the final character changes; an empty string stays empty."),
    _t(SW, "fourth_power", "x", "returns x raised to the fourth power.", "return x ** 4", "return x * x * x", "cube",
       [(2,), (3,), (-2,), (0,), (1,), (10,)], [(5,), (-1,)],
       "The result is x multiplied by itself four times."),
    _t(SW, "sum_of_cubes", "n", "returns the sum of the cubes of the integers from 1 to n.",
       "return sum(i ** 3 for i in range(1, n + 1))", "return sum(i * i for i in range(1, n + 1))", "sum_of_squares",
       [(3,), (1,), (0,), (5,), (10,)], [(4,), (2,)],
       "For n = 0 the sum is 0."),
    # ---- new wording, same implementation as a trained function (neighbour: a plausible distractor)
    _t(NW, "tally_items", "items", "tells how many elements the list holds.", "return len(items)",
       "return sum(1 for x in items if x)", "count_true",
       [([1, 2, 3],), ([],), ([True, False, True],), (list("abcdef"),), ([None],)],
       [([5, 5, 5, 5],), ([False],)],
       "An empty list gives 0.", "count_items"),
    _t(NW, "final_element", "items", "gives back the element at the very end of a non-empty list.", "return items[-1]",
       "return items[0]", "first_item", [(L7,), ([5],), (list("abc"),), ([[1], [2]],), ([0, 1],)],
       [([10, 20, 30],), (["p", "q"],)],
       "The list is never empty; for a one-item list that item is returned.", "last_item"),
    _t(NW, "front_element", "items", "gives back the element at the very front of a non-empty list.", "return items[0]",
       "return items[-1]", "last_item", [(L7,), ([5],), (list("abc"),), ([[1], [2]],), ([0, 1],)],
       [([10, 20, 30],), (["p", "q"],)],
       "The list is never empty; for a one-item list that item is returned.", "first_item"),
    _t(NW, "opposite", "x", "gives the additive inverse of x.", "return -x", "return abs(x)", "absolute",
       [(3,), (-4,), (0,), (2.5,), (-100,)], [(7,), (-9,)],
       "Positive numbers become negative and negative numbers become positive.", "negate"),
    _t(NW, "glue_with_commas", "words", "glues the strings in words together, putting a comma and a space between neighbours.",
       "return ', '.join(words)", "return [len(w) for w in words]", "lengths",
       [(["a", "b", "c"],), ([],), (["solo"],), (["x y", "z"],), (["1", "2"],)],
       [(["red", "green"],), (["up", "down", "left"],)],
       "There is no comma before the first string or after the last.", "join_with_commas"),
    _t(NW, "spread", "numbers", "measures how far apart the extremes of a non-empty list are: its maximum minus its minimum.",
       "return max(numbers) - min(numbers)", "return numbers.index(max(numbers))", "index_of_max",
       [([3, 9, 1],), ([5],), ([-2, 2],), ([7, 7, 7],), ([1, 2, 3, 10],)],
       [([4, 8, 6],), ([-5, 0, 5],)],
       "A list with one number gives 0.", "range_of_values"),
    _t(NW, "back_to_front", "items", "produces a fresh list holding the same elements back to front.",
       "return items[::-1]", "return sorted(items, reverse=True)", "sort_descending",
       [([1, 3, 2],), ([],), (list("abc"),), ([5, 5, 1],), ([2, 9, 4, 7],)],
       [([10, 30, 20],), (["q", "p"],)],
       "The items keep their values; only their order is reversed.", "reverse_list"),
    _t(NW, "how_often", "items, value", "tells how often value appears among items.", "return items.count(value)",
       "return items.index(value) if value in items else -1", "index_of",
       [([1, 2, 1, 1], 1), ([], 3), (list("banana"), "a"), ([0, 0], 1), ([2, 3, 2], 2)],
       [([5, 6, 5], 5), (list("hello"), "l")],
       "A value that does not occur gives 0.", "count_occurrences"),
    _t(NW, "seconds_in", "minutes", "tells how many seconds there are in the given number of minutes.",
       "return minutes * 60", "return minutes * 24", "days_to_hours",
       [(1,), (0,), (2.5,), (10,), (90,)], [(3,), (45,)],
       "A minute has 60 seconds.", "minutes_to_seconds"),
    _t(NW, "strip_numerals", "s", "takes every character from 0 to 9 out of s.", "return ''.join(c for c in s if not c.isdigit())",
       "return ''.join(c for c in s if c.isalpha())", "letters_only",
       [("a1b2c3",), ("no digits!",), ("2024",), ("",), ("x 9 y",)],
       [("room 101",), ("4ever young",)],
       "Every other character, including spaces and punctuation, stays.", "remove_digits"),
    _t(NW, "none_at_or_below_zero", "numbers", "says whether no number in the list is zero or below.",
       "return all(x > 0 for x in numbers)", "return any(x < 0 for x in numbers)", "any_negative",
       [([1, 2, 3],), ([1, 0, 2],), ([],), ([-1],), ([5, 6, -7],)],
       [([4, 9],), ([3, -3],)],
       "An empty list gives True.", "all_positive"),
    _t(NW, "dips_below_zero", "numbers", "says whether the list holds at least one value under zero.",
       "return any(x < 0 for x in numbers)", "return all(x > 0 for x in numbers)", "all_positive",
       [([1, -2, 3],), ([0, 1],), ([],), ([-5],), ([2, 4, 6],)],
       [([3, -1],), ([7, 8],)],
       "Zero is not under zero; an empty list gives False.", "any_negative"),
    _t(NW, "bracketed", "s", "puts s between a '[' and a ']'.", "return '[' + s + ']'", "return s.strip()", "trim",
       [("abc",), ("",), (" x ",), ("[]",), ("hello world",)],
       [("note",), (" y",)],
       "The opening bracket comes first and the closing bracket last; s is not changed.", "wrap_in_brackets"),
    _t(NW, "word_tally", "text", "tallies the words of text into a dictionary from each word to its count.",
       "counts = {}\nfor w in text.split():\n    counts[w] = counts.get(w, 0) + 1\nreturn counts",
       "counts = {}\nfor c in text:\n    counts[c] = counts.get(c, 0) + 1\nreturn counts", "char_frequencies",
       [("a b a",), ("",), ("hi",), ("x y z x",), ("one one one",)],
       [("to be or not to be",), ("cat dog cat",)],
       "Words are separated by whitespace; each distinct word is a key.", "word_frequencies"),
    _t(NW, "only_whitespace", "s", "says whether s holds nothing but whitespace, or nothing at all.",
       "return s.strip() == ''", "return s.isupper()", "is_uppercase",
       [("   ",), ("",), (" a ",), ("\t\n",), ("ABC",)],
       [("  \t ",), ("x",)],
       "The empty string gives True.", "is_blank"),
    _t(NW, "keep_even_positions", "items", "keeps every item whose position in the list is an even number, counting from zero.",
       "return items[::2]", "return [x for x in items if x % 2 == 1]", "filter_odd",
       [(list(range(7)),), ([5],), ([],), (list("abcdef"),), ([9, 8, 7, 6],)],
       [(list(range(10, 15)),), (list("pqrs"),)],
       "The first item is at position 0 and is kept.", "every_other"),
    _t(NW, "total_stored", "d", "adds up everything stored as a value in the dictionary d.", "return sum(d.values())",
       "return list(d.keys())", "dict_keys",
       [({"a": 1, "b": 2},), ({},), ({"x": -5},), ({1: 10, 2: 20, 3: 30},), ({"p": 0, "q": 0},)],
       [({"m": 3, "n": 4},), ({5: 1},)],
       "An empty dictionary gives 0.", "dict_values_sum"),
    _t(NW, "goes_into", "a, b", "says whether b goes into a with nothing left over.", "return a % b == 0",
       "return a % b", "remainder", [(10, 5), (10, 3), (0, 7), (9, 3), (14, 4)], [(12, 4), (7, 2)],
       "b is never zero.", "is_divisible"),
    _t(NW, "mirror_digits", "n", "gives the integer whose digits are those of the non-negative integer n read from right to left.",
       "return int(str(n)[::-1])", "return len(str(n))", "count_digits",
       [(123,), (0,), (100,), (5,), (9081,)], [(4567,), (30,)],
       "Leading zeros of the result disappear, so 100 gives 1.", "reverse_number"),
    _t(NW, "number_switched_on", "flags", "counts the switches that are on in a list of booleans.",
       "return sum(1 for f in flags if f)", "return len(flags)", "count_items",
       [([True, False, True],), ([],), ([False, False],), ([True],), ([True] * 4 + [False],)],
       [([False, True, True, True],), ([True, False],)],
       "True means on and False means off.", "count_true"),
    _t(NW, "squares_total", "numbers", "squares each number in the list and adds the results.",
       "return sum(x * x for x in numbers)", "return [x * x for x in numbers]", "squares",
       [([1, 2, 3],), ([],), ([-2],), ([0, 5],), ([1, 1, 1, 1],)],
       [([3, 4],), ([-1, 2],)],
       "An empty list gives 0.", "sum_all_square"),
    # ---- second batch (same design), written after the first batch's checks dropped 20 tasks
    _t(SW, "second_char", "s", "returns the second character of the string s, which has at least two characters.",
       "return s[1]", "return s[0]", "first_char", [("abc",), ("xy",), ("hello",), ("12",), ("!?a",)],
       [("python",), ("ok",)], "The first character counts as character one."),
    _t(SW, "second_to_last_char", "s", "returns the second to last character of the string s, which has at least two characters.",
       "return s[-2]", "return s[-1]", "last_char", [("abc",), ("xy",), ("hello",), ("12",), ("!?a",)],
       [("python",), ("ok",)], "The last character is the first from the end, so this is the one just before it."),
    _t(SW, "minutes_to_hours", "minutes", "converts minutes to hours.", "return minutes / 60", "return minutes * 60",
       "hours_to_minutes", [(60,), (0,), (90,), (30,), (600,)], [(120,), (15,)], "An hour has 60 minutes, so 90 minutes is 1.5 hours."),
    _t(SW, "seconds_to_minutes", "seconds", "converts seconds to minutes.", "return seconds / 60", "return seconds * 60",
       "minutes_to_seconds", [(60,), (0,), (90,), (30,), (600,)], [(120,), (15,)], "A minute has 60 seconds, so 90 seconds is 1.5 minutes."),
    _t(SW, "kelvin_to_celsius", "k", "converts a temperature from Kelvin to Celsius.", "return k - 273.15", "return k + 273.15",
       "celsius_to_kelvin", [(273.15,), (0,), (300,), (373.15,), (100,)], [(310.15,), (200,)],
       "0 degrees Celsius is 273.15 Kelvin."),
    _t(SW, "count_non_negative", "numbers", "returns how many numbers in the list are not negative.",
       "return sum(1 for x in numbers if x >= 0)", "return sum(1 for x in numbers if x < 0)", "count_negative",
       [([1, -2, 0, 3],), ([],), ([-1, -5],), ([0, 0],), ([4, 5, -6],)], [([2, -3, 7],), ([-1, 0],)],
       "Zero is not negative, so it is counted."),
    _t(SW, "remove_first", "items, value", "returns a new list without the first occurrence of value.",
       "result = list(items)\nif value in result:\n    result.remove(value)\nreturn result",
       "return [x for x in items if x != value]", "remove_value",
       [([1, 2, 1, 3], 1), ([], 5), ([4, 4, 4], 4), (list("abca"), "a"), ([7, 8], 9)],
       [([5, 6, 5, 6], 6), (list("xyzx"), "x")],
       "Later occurrences of value stay; if value does not occur, the list is unchanged."),
    _t(SW, "first_words", "text, n", "returns the first n words of text, joined by single spaces.",
       "return ' '.join(text.split()[:n])", "return text[:n]", "truncate",
       [("the quick brown fox", 2), ("hello world", 1), ("a b c", 0), ("one two", 5), ("red green blue", 3)],
       [("we will meet at noon", 3), ("x y z", 2)],
       "n counts words, not characters; if text has fewer than n words, all of them are returned."),
    _t(SW, "repeat_with_spaces", "s, n", "returns s repeated n times, separated by single spaces.",
       "return ' '.join([s] * n)", "return s * n", "repeat_string",
       [("ab", 3), ("x", 1), ("hi", 0), ("go", 2), ("a b", 2)], [("yes", 3), ("no", 2)],
       "There is no space at the start or at the end."),
    _t(SW, "count_nonempty_lines", "text", "returns the number of non-empty lines in text.",
       "return sum(1 for line in text.splitlines() if line.strip())", "return len(text.splitlines())", "count_lines",
       [("a\n\nb",), ("",), ("one\ntwo\nthree",), ("\n\n",), ("x\n \ny\n",)], [("first\n\nsecond\n",), ("solo",)],
       "A line holding only spaces counts as empty."),
    _t(SW, "odd_positions", "items", "returns every second item of the list, starting with the second.",
       "return items[1::2]", "return items[::2]", "every_other",
       [(list(range(7)),), ([5],), ([],), (list("abcdef"),), ([9, 8, 7, 6],)], [(list(range(10, 15)),), (list("pqrs"),)],
       "The kept items are those at positions 1, 3, 5 and so on."),
    _t(SW, "sum_first_n", "numbers, n", "returns the sum of the first n numbers of the list.",
       "return sum(numbers[:n])", "return numbers[:n]", "take",
       [([1, 2, 3, 4], 2), ([5], 1), ([], 3), ([1, 1, 1], 0), ([2, 4, 6], 5)], [([10, 20, 30], 2), ([7, 8, 9], 3)],
       "The result is a number; with n = 0 it is 0."),
    _t(SW, "closer_to_zero", "a, b", "returns whichever of a and b is closer to zero (a if they are equally close).",
       "return a if abs(a) <= abs(b) else b", "if a < b:\n    return a\nreturn b", "smaller",
       [(3, -5), (-2, 1), (4, -4), (0, 7), (-9, -3)], [(6, -2), (-1, 8)],
       "Distance from zero ignores the sign."),
    _t(SW, "join_with_dashes", "words", "returns the strings in words joined by '-'.", "return '-'.join(words)",
       "return ', '.join(words)", "join_with_commas", [(["a", "b", "c"],), ([],), (["solo"],), (["x y", "z"],), (["1", "2"],)],
       [(["red", "green"],), (["up", "down", "left"],)], "There is no dash before the first string or after the last."),
    _t(NW, "how_long", "s", "measures the length of s in characters.", "return len(s)",
       "return sum(1 for c in s.lower() if c.isalpha() and c not in 'aeiou')", "count_consonants",
       [("hello",), ("",), ("a b c",), ("12345",), ("xyz!",)], [("python",), ("go",)],
       "Spaces and punctuation are characters too.", "string_length"),
    _t(NW, "words_backwards", "text", "says the words of text in the opposite order, with one space between each.",
       "return ' '.join(text.split()[::-1])", "return ' '.join(word[::-1] for word in text.split())", "reverse_each_word",
       [("one two three",), ("hello",), ("",), ("a  b",), ("red green",)], [("we go home",), ("x y",)],
       "The letters inside each word keep their order.", "reverse_words"),
    _t(NW, "capital_initials", "phrase", "builds an abbreviation from the first letter of every word of phrase, in capitals.",
       "return ''.join(word[0] for word in phrase.split()).upper()",
       "return ' '.join(word.capitalize() for word in phrase.split())", "capitalize_words",
       [("portable network graphics",), ("hello",), ("as soon as possible",), ("Big Data",), ("x y z",)],
       [("graphics processing unit",), ("away from keyboard",)],
       "The result has one letter per word and no spaces.", "acronym"),
    _t(NW, "repeats_somewhere", "items", "reports whether some value appears two or more times in the list.",
       "return len(set(items)) != len(items)", "return len(set(items)) <= 1", "all_equal",
       [([1, 2, 1],), ([],), ([1, 2, 3],), ([5, 5],), (list("abcb"),)], [([3, 4, 3, 5],), ([9],)],
       "An empty list or a list of distinct values gives False.", "has_duplicates"),
    _t(NW, "shared_values", "a, b", "lists, in ascending order, the values that both lists contain.",
       "return sorted(set(a) & set(b))",
       "result = []\ni = j = 0\nwhile i < len(a) and j < len(b):\n    if a[i] <= b[j]:\n        result.append(a[i])\n        i += 1\n    else:\n        result.append(b[j])\n        j += 1\nreturn result + a[i:] + b[j:]",
       "merge_sorted", [([1, 2, 3], [2, 3, 4]), ([], [1]), ([5, 1], [1, 5]), ([1, 1, 2], [1]), ([7], [8])],
       [([3, 6, 9], [6, 9, 12]), ([2, 4], [4])], "Each shared value appears once.", "common_items"),
    _t(NW, "partial_sums", "numbers", "gives the list of totals of numbers up to and including each position.",
       "result = []\ntotal = 0\nfor x in numbers:\n    total += x\n    result.append(total)\nreturn result",
       "result = []\nbest = None\nfor x in numbers:\n    if best is None or x > best:\n        best = x\n    result.append(best)\nreturn result",
       "running_max", [([1, 2, 3],), ([],), ([5],), ([2, -1, 4],), ([0, 0, 1],)], [([3, 3, 3],), ([10, -5],)],
       "The last total is the sum of the whole list.", "cumulative_sum"),
    _t(NW, "swap_keys_values", "d", "builds a dictionary in which each value of d points back to its key.",
       "return {v: k for k, v in d.items()}", "return sorted(d)", "sorted_keys",
       [({"a": 1, "b": 2},), ({},), ({1: "x"},), ({"p": 0, "q": 5},), ({3: 4, 5: 6},)], [({"m": 7},), ({"u": 1, "v": 2},)],
       "The values of d are all different.", "invert_dict"),
    _t(NW, "has_366_days", "year", "says whether year has 366 days in the Gregorian calendar.",
       "return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)", "return year % 2 == 1", "is_odd",
       [(2024,), (1900,), (2000,), (2023,), (1996,), (2100,)], [(2020,), (1800,)],
       "Century years count only when divisible by 400.", "is_leap_year"),
    _t(NW, "groups_of", "items, size", "cuts the list into consecutive groups of size items each (the last group may be smaller).",
       "return [items[i:i + size] for i in range(0, len(items), size)]", "return items[:size]", "take",
       [([1, 2, 3, 4, 5], 2), ([], 3), (list("abcdef"), 3), ([1], 1), ([1, 2, 3], 5)], [([9, 8, 7, 6], 2), (list("xyz"), 2)],
       "Every group except possibly the last has exactly size items.", "chunk"),
    _t(NW, "floor_at_zero", "numbers", "gives a copy of the list in which every value below zero becomes 0.",
       "return [x if x > 0 else 0 for x in numbers]", "return [x for x in numbers if x > 0]", "filter_positive",
       [([1, -2, 3],), ([],), ([-5],), ([0, 4],), ([-1, -1, 2],)], [([3, -3, 6],), ([-7, 8],)],
       "The result has as many values as the list.", "clip_negatives"),
    _t(NW, "kilograms_from_pounds", "pounds", "turns a weight in pounds into kilograms (1 lb = 0.45359237 kg).",
       "return pounds * 0.45359237", "return pounds * 1000", "kilograms_to_grams",
       [(1,), (0,), (10,), (2.5,), (100,)], [(5,), (20,)], "The result is smaller than the number of pounds.", "pounds_to_kg"),
    _t(NW, "n_copies", "s, n", "concatenates n copies of s.", "return s * n", "return s[:n]", "truncate",
       [("ab", 3), ("x", 1), ("hi", 0), ("go", 2), ("abc", 1)], [("yes", 2), ("z", 4)],
       "n = 0 gives the empty string.", "repeat_string"),
    _t(NW, "pairs_into_dict", "pairs", "turns a list of (key, value) tuples into a dictionary.", "return dict(pairs)",
       "return list(pairs)", "dict_keys", [([("a", 1), ("b", 2)],), ([],), ([(1, "x")],), ([("k", 0), ("j", 9)],), ([(2, 3), (4, 5)],)],
       [([("m", 7)],), ([("u", 1), ("v", 2)],)], "Each tuple's first element is the key.", "pairs_to_dict"),
    _t(NW, "totals_per_column", "matrix", "adds up each column of a rectangular grid stored as a list of rows.",
       "return [sum(col) for col in zip(*matrix)]", "return [sum(row) for row in matrix]", "row_sums",
       [([[1, 2], [3, 4]],), ([[5, 6, 7]],), ([[1], [2], [3]],), ([[0, 0], [1, 1], [2, 2]],), ([[9, 8, 7], [1, 2, 3]],)],
       [([[2, 4], [6, 8]],), ([[1, 1, 1], [2, 2, 2]],)], "The result has one total per column.", "column_sums"),
)


def _call(t: Task, args: tuple) -> str:
    return f"{t.name}({', '.join(repr(a) for a in args)})"


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout


def run(source: str, name: str, args_list, *, seconds: float = 1.0) -> list:
    """Outputs (or the exception's type name) of ``name`` from ``source`` on each argument tuple; a
    call that runs longer than ``seconds`` (e.g. a training function's loop on 0) gives ("err", "Timeout")."""
    import signal

    ns: dict = {}
    exec(source, ns)
    out = []
    previous = signal.signal(signal.SIGALRM, _alarm)
    try:
        for args in args_list:
            signal.setitimer(signal.ITIMER_REAL, seconds)
            try:
                out.append(("ok", ns[name](*copy.deepcopy(args))))
            except _Timeout:
                out.append(("err", "Timeout"))
            except Exception as exc:  # noqa: BLE001 - the error's type is the outcome
                out.append(("err", type(exc).__name__))
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        signal.signal(signal.SIGALRM, previous)
    return out


def expected(t: Task) -> list:
    return [v for _, v in run(t.source(), t.name, t.tests)]


def test_source(t: Task) -> str:
    return "\n".join(f"assert {_call(t, a)} == {v!r}" for a, v in zip(t.tests, expected(t)))


def prompts(t: Task, train: dict[str, dict]) -> dict[str, str]:
    base = f"Write a Python function `{t.name}({t.params})` that {t.task}"
    shown = [f"`{_call(t, a)}` returns `{v!r}`" for a, (_, v) in zip(t.examples, run(t.source(), t.name, t.examples))]
    n = train[t.neighbour]
    return {"original": base,
            "io": f"{base} For example, {shown[0]} and {shown[1]}.",
            "constraint": f"{base} {t.constraint}",
            "distractor": f"{base} The same module also has `{n['name']}({n['params']})`, which {n['task']}"}


def checked_tasks(train: dict[str, dict], log=print) -> list[Task]:
    """The tasks that pass every check (see the module docstring)."""
    from ycode.lm.basics import dev_names, held_out_names
    from ycode.lm.leakage import behaviour_leaks, held_out_items, held_out_skeletons, skeleton

    holdout = [i for i in held_out_items() if i.split in ("bench", "unseen2")]
    holdout_names = held_out_names() | {i.name for i in holdout}
    holdout_skeletons = held_out_skeletons(holdout)
    dev_sets = [i for i in held_out_items() if i.split in ("dev", "unseen")]
    dev_items = {i.name for i in dev_sets}
    kept, dropped = [], {"holdout": 0}
    by_arity: dict[int, list[dict]] = {}
    for c in train.values():
        by_arity.setdefault(c["arity"], []).append(c)
    for t in TASKS:
        arity = len([p for p in t.params.split(",") if p.strip()])
        inputs = list(t.tests) + list(t.examples)
        mine = run(t.source(), t.name, inputs)
        assert all(k == "ok" for k, _ in mine), (t.name, mine)
        assert len({repr(v) for _, v in mine}) > 1, t.name
        assert t.neighbour in train and len(t.examples) == 2, t.name
        assert not {repr(a) for a in t.examples} & {repr(a) for a in t.tests}, (t.name, "an example is a test")
        wrong = run(t.wrong_source, t.name, t.tests)
        assert sum(repr(a) != repr(b) for a, b in zip(wrong, mine)) >= 2, (t.name, "wrong code passes")
        nb = train[t.neighbour]
        ns = run(nb["source"], nb["name"], t.tests)
        assert [repr(x) for x in ns] == [repr(x) for x in wrong], (t.name, "wrong_body is not the neighbour")
        twins = [c["name"] for c in by_arity.get(arity, []) if _same(c, inputs, mine)]
        if t.group == SW and twins:
            dropped[f"{t.name}: a trained function ({twins[0]})"] = 1
            continue
        if t.group == NW and t.same_as not in twins:
            dropped[f"{t.name}: not its trained twin"] = 1
            continue
        if (t.name in holdout_names or t.name in dev_names() or skeleton(t.source()) in holdout_skeletons
                or behaviour_leaks({t.source(): (t.name, arity)}, holdout, near=1.0)):
            dropped["holdout"] += 1  # counts only: never name a holdout item
            continue
        if t.name in train:
            dropped[f"{t.name}: name is trained"] = 1
            continue
        if t.name in dev_items or behaviour_leaks({t.source(): (t.name, arity)}, dev_sets, near=1.0):
            dropped[f"{t.name}: a dev or unseen-v1 item"] = 1
            continue
        kept.append(t)
    # balance the groups: the first n of each, in file order
    n = min(sum(t.group == SW for t in kept), sum(t.group == NW for t in kept))
    balanced = [t for t in kept if t.group == SW][:n] + [t for t in kept if t.group == NW][:n]
    log(f"{len(kept)} of {len(TASKS)} tasks pass; dropped {dropped}; {n} per group kept")
    return balanced


def _same(c: dict, inputs, mine) -> bool:
    try:
        theirs = run(c["source"], c["name"], inputs)
    except Exception:  # noqa: BLE001
        return False
    return [repr(x) for x in theirs] == [repr(x) for x in mine]


def freeze(path) -> dict:
    """Write the checked, balanced set with its four prompts per task, tests, reference and the
    neighbour's code, so evaluation never depends on regenerating it."""
    import json
    from pathlib import Path

    from experiments.v7.mechanism import catalogue

    train = catalogue()
    tasks = checked_tasks(train)
    data = {"version": VERSION, "variants": list(VARIANTS), "tasks": [
        {"group": t.group, "name": t.name, "params": t.params, "neighbour": t.neighbour, "same_as": t.same_as,
         "prompts": prompts(t, train), "tests": test_source(t), "test_inputs": [list(a) for a in t.tests],
         "reference": t.source(), "wrong": t.wrong_source,
         "expected": [repr(v) for v in expected(t)],
         "wrong_outputs": [repr(v) if k == "ok" else f"<{v}>" for k, v in run(t.wrong_source, t.name, t.tests)]}
        for t in tasks]}
    Path(path).write_text(json.dumps(data, indent=1))
    return data


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    d = freeze(Path(__file__).resolve().parents[2] / "ycode" / "lm" / "neighbours_v1.json")
    print(len(d["tasks"]), "tasks written")
