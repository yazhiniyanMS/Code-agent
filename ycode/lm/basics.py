"""Basic programming exercises for instruction tuning, with verified solutions.

YCode-LM's first instruction data came from real library code, which is long and complex. Small,
everyday functions ("write `cube(x)`", "fix this off-by-one") were rare. This module generates them:

* ``CONCEPTS``: ~100 small function exercises (arithmetic, strings, lists, dicts, number theory,
  searching). Each has a reference solution and tests, and every solution is executed against
  its tests before it is used.
* Write-a-function examples in several phrasings, and find-and-fix-the-bug examples made by
  injecting a realistic bug into a reference solution (kept only if the bug makes a test fail).

**Held out:** none of the evaluation problems in ``ycode.lm.evaluate`` (``PROBLEMS``, ``BUGGY``)
or ``FRESH_PROBLEMS`` below is a training concept; ``assert_no_overlap`` checks the names. Related
exercises do appear (``cube`` next to the benchmark's ``square``), which is why ``FRESH_PROBLEMS``
exists: 20 problems whose concepts appear nowhere in this file's training set.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ycode.lm.data import InstructionExample, inject_bug


@dataclass(frozen=True)
class Concept:
    name: str
    params: str
    task: str  # completes "Write a function `name(params)` that ..."
    body: str  # indented with 4 spaces
    tests: str
    aliases: tuple[str, ...] = ()

    @property
    def signature(self) -> str:
        return f"{self.name}({self.params})"

    def source(self, name: str | None = None) -> str:
        return f"def {name or self.name}({self.params}):\n{self.body}"


def _c(name, params, task, body, tests, *aliases) -> Concept:
    lines = [line for line in body.strip("\n").splitlines()]
    return Concept(name, params, task, "\n".join("    " + line for line in lines), tests.strip(), aliases)


CONCEPTS: tuple[Concept, ...] = (
    # ---- arithmetic
    _c("multiply", "a, b", "returns the product of a and b.", "return a * b",
       "assert f(3, 4) == 12\nassert f(-2, 5) == -10", "product_of_two", "times"),
    _c("divide", "a, b", "returns a divided by b.", "return a / b", "assert f(7, 2) == 3.5", "quotient"),
    _c("floor_divide", "a, b", "returns the integer division of a by b.", "return a // b",
       "assert f(7, 2) == 3\nassert f(9, 3) == 3", "int_divide"),
    _c("remainder", "a, b", "returns the remainder when a is divided by b.", "return a % b",
       "assert f(7, 3) == 1\nassert f(9, 3) == 0", "modulo"),
    _c("power", "base, exponent", "returns base raised to the power exponent.", "return base ** exponent",
       "assert f(2, 10) == 1024\nassert f(5, 0) == 1", "raise_to_power"),
    _c("cube", "x", "returns x cubed.", "return x * x * x", "assert f(3) == 27\nassert f(-2) == -8", "cubed"),
    _c("double", "x", "returns twice x.", "return 2 * x", "assert f(4) == 8\nassert f(-1) == -2", "times_two"),
    _c("half", "x", "returns half of x.", "return x / 2", "assert f(9) == 4.5", "halve"),
    _c("negate", "x", "returns x with its sign flipped.", "return -x", "assert f(3) == -3\nassert f(-4) == 4"),
    _c("absolute", "x", "returns the absolute value of x.", "if x < 0:\n    return -x\nreturn x",
       "assert f(-5) == 5\nassert f(3) == 3", "abs_value"),
    _c("distance", "a, b", "returns the absolute difference between a and b.", "return abs(a - b)",
       "assert f(3, 10) == 7\nassert f(10, 3) == 7", "abs_difference"),
    _c("mean_of_two", "a, b", "returns the average of a and b.", "return (a + b) / 2",
       "assert f(2, 4) == 3\nassert f(1, 2) == 1.5", "midpoint"),
    _c("larger", "a, b", "returns the larger of a and b.", "if a > b:\n    return a\nreturn b",
       "assert f(3, 7) == 7\nassert f(9, 2) == 9", "max_of_two", "bigger"),
    _c("smaller", "a, b", "returns the smaller of a and b.", "if a < b:\n    return a\nreturn b",
       "assert f(3, 7) == 3\nassert f(9, 2) == 2", "min_of_two"),
    _c("largest_of_three", "a, b, c", "returns the largest of a, b and c.", "return max(a, b, c)",
       "assert f(1, 9, 4) == 9\nassert f(-1, -5, -2) == -1", "max_of_three"),
    _c("sign", "x", "returns 1 if x is positive, -1 if it is negative, and 0 if it is zero.",
       "if x > 0:\n    return 1\nif x < 0:\n    return -1\nreturn 0", "assert f(5) == 1\nassert f(-2) == -1\nassert f(0) == 0"),
    _c("is_odd", "n", "returns True if n is odd, otherwise False.", "return n % 2 == 1",
       "assert f(3) is True\nassert f(4) is False", "odd"),
    _c("is_positive", "x", "returns True if x is greater than zero.", "return x > 0",
       "assert f(2)\nassert not f(0)\nassert not f(-3)"),
    _c("is_negative", "x", "returns True if x is less than zero.", "return x < 0", "assert f(-2)\nassert not f(0)"),
    _c("is_divisible", "a, b", "returns True if a is divisible by b.", "return a % b == 0",
       "assert f(10, 5)\nassert not f(10, 3)", "divides_evenly"),
    _c("is_multiple_of_three", "n", "returns True if n is a multiple of 3.", "return n % 3 == 0",
       "assert f(9)\nassert not f(10)"),
    _c("cube_root_is_integer", "n", "returns True if n is a perfect cube of a non-negative integer.",
       "k = 0\nwhile k * k * k < n:\n    k += 1\nreturn k * k * k == n", "assert f(27)\nassert f(0)\nassert not f(20)",
       "is_perfect_cube"),
    _c("triangular_number", "n", "returns the sum of the integers from 1 to n.", "return n * (n + 1) // 2",
       "assert f(4) == 10\nassert f(0) == 0", "sum_to_n"),
    _c("sum_of_squares", "n", "returns the sum of the squares of the integers from 1 to n.",
       "total = 0\nfor i in range(1, n + 1):\n    total += i * i\nreturn total", "assert f(3) == 14\nassert f(0) == 0"),
    _c("count_digits", "n", "returns how many decimal digits the non-negative integer n has.",
       "return len(str(n))", "assert f(0) == 1\nassert f(1234) == 4", "num_digits"),
    _c("last_digit", "n", "returns the last decimal digit of the non-negative integer n.", "return n % 10",
       "assert f(1234) == 4\nassert f(7) == 7"),
    _c("reverse_number", "n", "returns the digits of the non-negative integer n in reverse order, as an integer.",
       "return int(str(n)[::-1])", "assert f(123) == 321\nassert f(100) == 1", "reverse_digits"),
    _c("power_of_two", "n", "returns True if n is a positive power of two.",
       "if n < 1:\n    return False\nwhile n % 2 == 0:\n    n //= 2\nreturn n == 1",
       "assert f(1)\nassert f(16)\nassert not f(12)\nassert not f(0)", "is_power_of_two"),
    _c("lcm", "a, b", "returns the least common multiple of the positive integers a and b.",
       "m = max(a, b)\nwhile m % a != 0 or m % b != 0:\n    m += 1\nreturn m", "assert f(4, 6) == 12\nassert f(3, 5) == 15",
       "least_common_multiple"),
    _c("collatz_steps", "n", "returns how many steps the Collatz process takes to reach 1 from n.",
       "steps = 0\nwhile n != 1:\n    if n % 2 == 0:\n        n //= 2\n    else:\n        n = 3 * n + 1\n    steps += 1\nreturn steps",
       "assert f(1) == 0\nassert f(6) == 8"),
    _c("divisors", "n", "returns a list of the positive divisors of n in ascending order.",
       "return [d for d in range(1, n + 1) if n % d == 0]", "assert f(12) == [1, 2, 3, 4, 6, 12]\nassert f(1) == [1]",
       "factors"),
    # ---- conversions
    _c("km_to_miles", "km", "converts a distance from kilometres to miles (1 km = 0.621371 miles).",
       "return km * 0.621371", "assert abs(f(10) - 6.21371) < 1e-9", "kilometres_to_miles"),
    _c("minutes_to_seconds", "minutes", "converts minutes to seconds.", "return minutes * 60",
       "assert f(2) == 120\nassert f(0) == 0"),
    _c("hours_to_minutes", "hours", "converts hours to minutes.", "return hours * 60", "assert f(3) == 180"),
    _c("fahrenheit_to_kelvin", "f_deg", "converts a temperature from Fahrenheit to Kelvin.",
       "return (f_deg - 32) * 5 / 9 + 273.15", "assert abs(f(32) - 273.15) < 1e-9"),
    _c("percentage", "part, whole", "returns part as a percentage of whole.", "return part / whole * 100",
       "assert f(1, 4) == 25\nassert f(3, 3) == 100", "percent"),
    _c("bool_to_int", "flag", "returns 1 if flag is True and 0 otherwise.", "return 1 if flag else 0",
       "assert f(True) == 1\nassert f(False) == 0"),
    # ---- strings
    _c("to_lower", "s", "returns s converted to lower case.", "return s.lower()", "assert f('AbC') == 'abc'",
       "lowercase"),
    _c("capitalize_words", "text", "returns text with the first letter of every word in upper case.",
       "return ' '.join(word.capitalize() for word in text.split())", "assert f('hello big world') == 'Hello Big World'",
       "title_case"),
    _c("last_word", "text", "returns the last word of text.", "return text.split()[-1]",
       "assert f('hello big world') == 'world'"),
    _c("first_char", "s", "returns the first character of the non-empty string s.", "return s[0]",
       "assert f('python') == 'p'"),
    _c("last_char", "s", "returns the last character of the non-empty string s.", "return s[-1]",
       "assert f('python') == 'n'"),
    _c("string_length", "s", "returns the number of characters in s.", "return len(s)",
       "assert f('abc') == 3\nassert f('') == 0", "length_of"),
    _c("repeat_string", "s, n", "returns s repeated n times.", "return s * n",
       "assert f('ab', 3) == 'ababab'\nassert f('x', 0) == ''"),
    _c("count_consonants", "s", "returns the number of consonant letters in s.",
       "return sum(1 for c in s.lower() if c.isalpha() and c not in 'aeiou')",
       "assert f('hello') == 3\nassert f('aei') == 0"),
    _c("remove_spaces", "s", "returns s with all spaces removed.", "return s.replace(' ', '')",
       "assert f('a b  c') == 'abc'", "strip_spaces"),
    _c("starts_with", "s, prefix", "returns True if s starts with prefix.", "return s.startswith(prefix)",
       "assert f('python', 'py')\nassert not f('python', 'th')"),
    _c("ends_with", "s, suffix", "returns True if s ends with suffix.", "return s.endswith(suffix)",
       "assert f('python', 'on')\nassert not f('python', 'py')"),
    _c("contains_substring", "s, sub", "returns True if sub occurs in s.", "return sub in s",
       "assert f('banana', 'nan')\nassert not f('banana', 'xyz')", "has_substring"),
    _c("replace_word", "text, old, new", "returns text with every occurrence of old replaced by new.",
       "return text.replace(old, new)", "assert f('a cat and a cat', 'cat', 'dog') == 'a dog and a dog'"),
    _c("reverse_words", "text", "returns the words of text in reverse order, separated by single spaces.",
       "return ' '.join(text.split()[::-1])", "assert f('one two three') == 'three two one'"),
    _c("longest_word", "text", "returns the longest word in text (the first one if there is a tie).",
       "best = ''\nfor word in text.split():\n    if len(word) > len(best):\n        best = word\nreturn best",
       "assert f('a bbb cc') == 'bbb'\nassert f('ab cd') == 'ab'"),
    _c("shortest_word", "text", "returns the shortest word in text (the first one if there is a tie).",
       "return min(text.split(), key=len)", "assert f('aaa b cc') == 'b'"),
    _c("initials", "name", "returns the initials of a full name, in upper case.",
       "return ''.join(part[0].upper() for part in name.split())", "assert f('ada lovelace') == 'AL'"),
    _c("is_digit_string", "s", "returns True if s is non-empty and contains only digits.", "return s.isdigit()",
       "assert f('123')\nassert not f('12a')\nassert not f('')", "all_digits"),
    _c("count_uppercase", "s", "returns how many upper-case letters s contains.",
       "return sum(1 for c in s if c.isupper())", "assert f('HeLLo') == 3\nassert f('abc') == 0"),
    _c("swap_case", "s", "returns s with upper-case letters made lower case and lower-case letters made upper case.",
       "return s.swapcase()", "assert f('aBc') == 'AbC'"),
    _c("join_with_commas", "words", "returns the strings in words joined by ', '.", "return ', '.join(words)",
       "assert f(['a', 'b', 'c']) == 'a, b, c'\nassert f([]) == ''"),
    _c("split_lines", "text", "returns a list of the lines in text.", "return text.splitlines()",
       "assert f('a\\nb\\nc') == ['a', 'b', 'c']"),
    _c("pad_left", "s, width", "returns s padded on the left with spaces to at least width characters.",
       "return s.rjust(width)", "assert f('ab', 4) == '  ab'\nassert f('abcde', 2) == 'abcde'"),
    _c("acronym", "phrase", "returns the acronym of phrase: the first letter of each word, in upper case.",
       "return ''.join(word[0] for word in phrase.split()).upper()", "assert f('portable network graphics') == 'PNG'"),
    _c("vowel_positions", "s", "returns a list of the indexes in s that hold a vowel.",
       "return [i for i, c in enumerate(s) if c in 'aeiou']", "assert f('banana') == [1, 3, 5]"),
    # ---- lists
    _c("product", "numbers", "returns the product of a list of numbers.",
       "result = 1\nfor x in numbers:\n    result *= x\nreturn result", "assert f([2, 3, 4]) == 24\nassert f([]) == 1",
       "product_of_list", "multiply_all"),
    _c("count_items", "items", "returns how many items the list contains.", "return len(items)",
       "assert f([1, 2, 3]) == 3\nassert f([]) == 0", "list_length"),
    _c("first_item", "items", "returns the first item of a non-empty list.", "return items[0]", "assert f([5, 6]) == 5",
       "head"),
    _c("last_item", "items", "returns the last item of a non-empty list.", "return items[-1]", "assert f([5, 6]) == 6",
       "tail_item"),
    _c("sum_positive", "numbers", "returns the sum of the positive numbers in the list.",
       "return sum(x for x in numbers if x > 0)", "assert f([1, -2, 3]) == 4\nassert f([-1]) == 0"),
    _c("count_negative", "numbers", "returns how many numbers in the list are negative.",
       "return sum(1 for x in numbers if x < 0)", "assert f([1, -2, -3]) == 2\nassert f([]) == 0"),
    _c("filter_odd", "numbers", "returns a list of the odd numbers in numbers.", "return [n for n in numbers if n % 2 == 1]",
       "assert f([1, 2, 3, 4]) == [1, 3]\nassert f([]) == []", "odd_numbers"),
    _c("filter_positive", "numbers", "returns a list of the numbers greater than zero.",
       "return [n for n in numbers if n > 0]", "assert f([3, -1, 0, 2]) == [3, 2]", "positives"),
    _c("greater_than", "numbers, limit", "returns a list of the numbers greater than limit.",
       "return [n for n in numbers if n > limit]", "assert f([1, 5, 3, 8], 3) == [5, 8]"),
    _c("double_all", "numbers", "returns a new list with every number doubled.", "return [2 * n for n in numbers]",
       "assert f([1, 2, 3]) == [2, 4, 6]", "doubled"),
    _c("squares", "numbers", "returns a list of the squares of the numbers.", "return [n * n for n in numbers]",
       "assert f([1, 2, 3]) == [1, 4, 9]", "square_all"),
    _c("lengths", "words", "returns a list of the lengths of the strings in words.", "return [len(w) for w in words]",
       "assert f(['a', 'bcd', '']) == [1, 3, 0]", "word_lengths"),
    _c("contains", "items, value", "returns True if value is in the list items.", "return value in items",
       "assert f([1, 2, 3], 2)\nassert not f([1, 2, 3], 5)", "includes"),
    _c("index_of", "items, value", "returns the index of the first occurrence of value in items, or -1 if it is absent.",
       "for i, item in enumerate(items):\n    if item == value:\n        return i\nreturn -1",
       "assert f([4, 5, 6], 5) == 1\nassert f([4, 5], 9) == -1", "find_index", "linear_search"),
    _c("count_occurrences", "items, value", "returns how many times value occurs in items.",
       "return sum(1 for item in items if item == value)", "assert f([1, 2, 1, 1], 1) == 3\nassert f([], 1) == 0"),
    _c("reverse_list", "items", "returns a new list with the items in reverse order.", "return items[::-1]",
       "assert f([1, 2, 3]) == [3, 2, 1]\nassert f([]) == []", "reversed_list"),
    _c("sort_descending", "numbers", "returns a new list of the numbers sorted from largest to smallest.",
       "return sorted(numbers, reverse=True)", "assert f([3, 1, 2]) == [3, 2, 1]"),
    _c("take", "items, n", "returns the first n items of the list.", "return items[:n]",
       "assert f([1, 2, 3, 4], 2) == [1, 2]\nassert f([1], 5) == [1]", "first_n"),
    _c("drop", "items, n", "returns the list without its first n items.", "return items[n:]",
       "assert f([1, 2, 3, 4], 2) == [3, 4]", "skip_n"),
    _c("range_list", "n", "returns a list of the integers from 0 to n - 1.", "return list(range(n))",
       "assert f(4) == [0, 1, 2, 3]\nassert f(0) == []"),
    _c("all_positive", "numbers", "returns True if every number in the list is greater than zero.",
       "return all(n > 0 for n in numbers)", "assert f([1, 2])\nassert not f([1, -2])\nassert f([])"),
    _c("any_negative", "numbers", "returns True if at least one number in the list is negative.",
       "return any(n < 0 for n in numbers)", "assert f([1, -2])\nassert not f([1, 2])"),
    _c("range_of_values", "numbers", "returns the difference between the largest and smallest number in a non-empty list.",
       "return max(numbers) - min(numbers)", "assert f([3, 9, 1]) == 8\nassert f([5]) == 0", "spread"),
    _c("pairwise_sum", "a, b", "returns a list of the sums of the numbers at the same positions in a and b.",
       "return [x + y for x, y in zip(a, b)]", "assert f([1, 2], [10, 20]) == [11, 22]", "add_lists"),
    _c("chunk", "items, size", "splits the list into consecutive pieces of length size (the last may be shorter).",
       "return [items[i:i + size] for i in range(0, len(items), size)]",
       "assert f([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]", "chunks"),
    _c("rotate_left", "items, k", "returns the list rotated k positions to the left.",
       "if not items:\n    return []\nk = k % len(items)\nreturn items[k:] + items[:k]",
       "assert f([1, 2, 3, 4], 1) == [2, 3, 4, 1]\nassert f([], 3) == []"),
    _c("cumulative_sum", "numbers", "returns the running totals of the list of numbers.",
       "result = []\ntotal = 0\nfor n in numbers:\n    total += n\n    result.append(total)\nreturn result",
       "assert f([1, 2, 3]) == [1, 3, 6]\nassert f([]) == []", "running_total"),
    _c("most_common", "items", "returns the item that occurs most often in a non-empty list.",
       "return max(items, key=items.count)", "assert f([1, 2, 2, 3]) == 2"),
    _c("remove_value", "items, value", "returns a new list without any occurrence of value.",
       "return [item for item in items if item != value]", "assert f([1, 2, 1, 3], 1) == [2, 3]"),
    _c("interleave", "a, b", "returns a list alternating the items of two lists of equal length.",
       "result = []\nfor x, y in zip(a, b):\n    result.append(x)\n    result.append(y)\nreturn result",
       "assert f([1, 2], ['a', 'b']) == [1, 'a', 2, 'b']"),
    _c("all_equal", "items", "returns True if every item in the list is equal.", "return len(set(items)) <= 1",
       "assert f([3, 3, 3])\nassert not f([3, 4])\nassert f([])"),
    _c("count_greater", "numbers, limit", "returns how many numbers are greater than limit.",
       "count = 0\nfor n in numbers:\n    if n > limit:\n        count += 1\nreturn count", "assert f([1, 5, 7], 4) == 2"),
    _c("index_of_max", "numbers", "returns the index of the largest number in a non-empty list.",
       "best = 0\nfor i in range(1, len(numbers)):\n    if numbers[i] > numbers[best]:\n        best = i\nreturn best",
       "assert f([3, 9, 2]) == 1\nassert f([5]) == 0", "argmax"),
    # ---- dicts and sets
    _c("dict_keys", "d", "returns a list of the keys of the dictionary d.", "return list(d)",
       "assert f({'a': 1, 'b': 2}) == ['a', 'b']", "keys_of"),
    _c("dict_values_sum", "d", "returns the sum of the values of the dictionary d.", "return sum(d.values())",
       "assert f({'a': 1, 'b': 2}) == 3\nassert f({}) == 0", "sum_values"),
    _c("invert_dict", "d", "returns a new dictionary with the keys and values of d swapped.",
       "return {value: key for key, value in d.items()}", "assert f({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}",
       "swap_keys_values"),
    _c("get_or_default", "d, key, default", "returns d[key] if key is in d, otherwise default.",
       "if key in d:\n    return d[key]\nreturn default", "assert f({'a': 1}, 'a', 0) == 1\nassert f({}, 'a', 0) == 0"),
    _c("word_frequencies", "text", "returns a dictionary mapping each word in text to how many times it occurs.",
       "counts = {}\nfor word in text.split():\n    counts[word] = counts.get(word, 0) + 1\nreturn counts",
       "assert f('a b a') == {'a': 2, 'b': 1}\nassert f('') == {}", "count_each_word"),
    _c("common_items", "a, b", "returns a sorted list of the values that appear in both lists.",
       "return sorted(set(a) & set(b))", "assert f([1, 2, 3], [2, 3, 4]) == [2, 3]", "intersection"),
    _c("has_duplicates", "items", "returns True if any value occurs more than once in the list.",
       "return len(set(items)) != len(items)", "assert f([1, 2, 1])\nassert not f([1, 2, 3])"),
    _c("key_with_max_value", "d", "returns the key of the non-empty dictionary d with the largest value.",
       "return max(d, key=d.get)", "assert f({'a': 1, 'b': 5, 'c': 3}) == 'b'"),
    _c("pairs_to_dict", "pairs", "builds a dictionary from a list of (key, value) pairs.", "return dict(pairs)",
       "assert f([('a', 1), ('b', 2)]) == {'a': 1, 'b': 2}"),
    # ---- classic small algorithms
    _c("count_down", "n", "returns a list counting down from n to 1.", "return list(range(n, 0, -1))",
       "assert f(3) == [3, 2, 1]\nassert f(0) == []"),
    _c("fizzbuzz", "n", "returns 'Fizz' if n is divisible by 3, 'Buzz' if by 5, 'FizzBuzz' if by both, otherwise str(n).",
       "if n % 15 == 0:\n    return 'FizzBuzz'\nif n % 3 == 0:\n    return 'Fizz'\nif n % 5 == 0:\n    return 'Buzz'\nreturn str(n)",
       "assert f(9) == 'Fizz'\nassert f(10) == 'Buzz'\nassert f(30) == 'FizzBuzz'\nassert f(7) == '7'"),
    _c("is_leap_year", "year", "returns True if year is a leap year in the Gregorian calendar.",
       "return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)",
       "assert f(2024)\nassert not f(1900)\nassert f(2000)\nassert not f(2023)", "leap_year"),
    _c("grade", "score", "returns 'A' for scores of 90 or more, 'B' for 80 or more, 'C' for 70 or more, otherwise 'F'.",
       "if score >= 90:\n    return 'A'\nif score >= 80:\n    return 'B'\nif score >= 70:\n    return 'C'\nreturn 'F'",
       "assert f(95) == 'A'\nassert f(80) == 'B'\nassert f(72) == 'C'\nassert f(10) == 'F'", "letter_grade"),
    _c("bubble_sort", "items", "returns a new list with the items sorted in ascending order, using bubble sort.",
       "result = list(items)\nfor i in range(len(result)):\n    for j in range(len(result) - 1 - i):\n"
       "        if result[j] > result[j + 1]:\n            result[j], result[j + 1] = result[j + 1], result[j]\nreturn result",
       "assert f([3, 1, 2]) == [1, 2, 3]\nassert f([]) == []"),
    _c("merge_sorted", "a, b", "merges two sorted lists into one sorted list.",
       "result = []\ni = j = 0\nwhile i < len(a) and j < len(b):\n    if a[i] <= b[j]:\n        result.append(a[i])\n"
       "        i += 1\n    else:\n        result.append(b[j])\n        j += 1\nreturn result + a[i:] + b[j:]",
       "assert f([1, 4], [2, 3, 5]) == [1, 2, 3, 4, 5]\nassert f([], [1]) == [1]"),
    _c("count_islands_1d", "bits", "returns how many separate runs of 1s the list of 0s and 1s contains.",
       "count = 0\nprevious = 0\nfor b in bits:\n    if b == 1 and previous == 0:\n        count += 1\n    previous = b\nreturn count",
       "assert f([1, 1, 0, 1, 0, 0, 1]) == 3\nassert f([0, 0]) == 0", "count_runs"),
    _c("power_set_size", "items", "returns how many subsets the list has.", "return 2 ** len(items)",
       "assert f([1, 2, 3]) == 8\nassert f([]) == 1"),
    _c("matrix_sum", "matrix", "returns the sum of all numbers in a list of lists.",
       "return sum(sum(row) for row in matrix)", "assert f([[1, 2], [3, 4]]) == 10\nassert f([]) == 0"),
    _c("transpose", "matrix", "returns the transpose of a rectangular list of lists.",
       "return [list(row) for row in zip(*matrix)]", "assert f([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]"),
    _c("triangle_area", "base, height", "returns the area of a triangle.", "return base * height / 2",
       "assert f(4, 3) == 6"),
    _c("rectangle_perimeter", "width, height", "returns the perimeter of a rectangle.", "return 2 * (width + height)",
       "assert f(3, 4) == 14", "perimeter"),
    _c("circle_area", "radius", "returns the area of a circle with the given radius.",
       "import math\nreturn math.pi * radius ** 2", "import math\nassert abs(f(2) - 4 * math.pi) < 1e-9"),
)


# 20 problems whose concepts appear nowhere in CONCEPTS: a fresh check that the skills generalize.
FRESH_PROBLEMS: tuple[tuple[str, str, str, str], ...] = (
    ("triple", "triple(x)", "returns three times x.", "assert triple(4) == 12\nassert triple(-1) == -3"),
    ("is_zero", "is_zero(x)", "returns True if x equals zero.", "assert is_zero(0)\nassert not is_zero(3)"),
    ("area_of_square", "area_of_square(side)", "returns the area of a square with the given side length.",
     "assert area_of_square(3) == 9"),
    ("count_spaces", "count_spaces(s)", "returns how many space characters s contains.",
     "assert count_spaces('a b c') == 2\nassert count_spaces('abc') == 0"),
    ("ends_with_dot", "ends_with_dot(s)", "returns True if the string s ends with a period.",
     "assert ends_with_dot('end.')\nassert not ends_with_dot('end')"),
    ("second_item", "second_item(items)", "returns the second item of a list with at least two items.",
     "assert second_item([7, 8, 9]) == 8"),
    ("sum_of_evens", "sum_of_evens(numbers)", "returns the sum of the even numbers in the list.",
     "assert sum_of_evens([1, 2, 3, 4]) == 6\nassert sum_of_evens([]) == 0"),
    ("count_zeros", "count_zeros(numbers)", "returns how many zeros the list contains.",
     "assert count_zeros([0, 1, 0]) == 2\nassert count_zeros([]) == 0"),
    ("halve_all", "halve_all(numbers)", "returns a new list with every number divided by two.",
     "assert halve_all([2, 4, 5]) == [1, 2, 2.5]"),
    ("max_length", "max_length(words)", "returns the length of the longest string in a non-empty list.",
     "assert max_length(['a', 'abcd', 'ab']) == 4"),
    ("seconds_to_minutes", "seconds_to_minutes(seconds)", "converts seconds to minutes.",
     "assert seconds_to_minutes(120) == 2\nassert seconds_to_minutes(30) == 0.5"),
    ("is_empty", "is_empty(items)", "returns True if the list has no items.", "assert is_empty([])\nassert not is_empty([1])"),
    ("add_one", "add_one(numbers)", "returns a new list with 1 added to every number.",
     "assert add_one([1, 2]) == [2, 3]\nassert add_one([]) == []"),
    ("starts_with_vowel", "starts_with_vowel(word)", "returns True if the non-empty string word starts with a vowel.",
     "assert starts_with_vowel('apple')\nassert not starts_with_vowel('pear')"),
    ("middle_item", "middle_item(items)", "returns the middle item of a list with an odd number of items.",
     "assert middle_item([1, 2, 3]) == 2\nassert middle_item([5]) == 5"),
    ("count_letter_a", "count_letter_a(s)", "returns how many times the letter 'a' occurs in s.",
     "assert count_letter_a('banana') == 3"),
    ("sum_of_cubes", "sum_of_cubes(numbers)", "returns the sum of the cubes of the numbers.",
     "assert sum_of_cubes([1, 2]) == 9\nassert sum_of_cubes([]) == 0"),
    ("all_strings_short", "all_strings_short(words, limit)",
     "returns True if every string in words has fewer than limit characters.",
     "assert all_strings_short(['a', 'bb'], 3)\nassert not all_strings_short(['abcd'], 3)"),
    ("dict_size", "dict_size(d)", "returns how many keys the dictionary d has.", "assert dict_size({'a': 1, 'b': 2}) == 2"),
    ("join_words", "join_words(words)", "returns the strings in words joined with single spaces.",
     "assert join_words(['a', 'b']) == 'a b'\nassert join_words([]) == ''"),
)


_WRITE_TEMPLATES = (
    "Write a Python function `{sig}` that {task}",
    "Write a Python function `{sig}` that {task}",  # the most common phrasing, weighted double
    "Write a function {sig} that {task}",
    "Implement `{sig}`, which {task}",
    "Create a Python function named `{name}` that takes {params} and {task}",
    "Python: define `{sig}`. It {task}",
)
BUGFIX_PROMPT = "This function has a bug. Find and fix it:\n```python\n{code}\n```"


def _rename_tests(tests: str, name: str) -> str:
    return tests.replace("f(", f"{name}(")


def _parses(source: str) -> bool:
    try:
        compile(source, "<bug>", "exec")
    except SyntaxError:
        return False
    return True


class _Timeout(Exception):
    pass


def _passes(source: str, tests: str, timeout: float = 1.0) -> bool:
    """Run our own reference code (or a bug we injected into it) against its tests.

    An injected bug can turn a loop infinite (`k += 1` -> `k -= 1`), so runs are time-limited:
    with a signal alarm on Unix, otherwise in a subprocess."""
    import signal

    if not hasattr(signal, "setitimer"):
        from ycode.lm.evaluate import run_tests

        return run_tests(source, tests, timeout=timeout)

    def on_alarm(_signum, _frame):
        raise _Timeout

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        exec(compile(source + "\n\n" + tests, "<basics>", "exec"), {})
        return True
    except BaseException as exc:  # noqa: BLE001 - failing test, crash, or timeout
        if isinstance(exc, KeyboardInterrupt):
            raise
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def verify_concepts(concepts=CONCEPTS) -> None:
    """Run every reference solution against its tests; raise if any fails."""
    for c in concepts:
        if not _passes(c.source(), _rename_tests(c.tests, c.name)):
            raise AssertionError(f"reference solution for {c.name} fails its tests")


def assert_no_overlap(concepts=CONCEPTS) -> None:
    """Training concepts must not share a name with any evaluation problem."""
    from ycode.lm.evaluate import BUGGY, PROBLEMS

    held_out = {p.name for p in PROBLEMS} | {sig.split("(")[0] for sig, _ in BUGGY} | {p[0] for p in FRESH_PROBLEMS}
    for c in concepts:
        clash = {c.name, *c.aliases} & held_out
        if clash:
            raise AssertionError(f"training concept {c.name} overlaps the evaluation set: {sorted(clash)}")


def _with_doc(c: Concept, name: str, body: str) -> str:
    doc = c.task[0].upper() + c.task[1:]
    for verb in ("Returns", "Converts", "Splits", "Merges", "Builds", "Flattens"):
        if doc.startswith(verb + " "):
            doc = verb[:-1] + doc[len(verb):]  # docstring style: "Return ...", not "Returns ..."
    return f'def {name}({c.params}):\n    """{doc}"""\n{body}'


def write_examples(rng: random.Random, per_concept: int = 6) -> list[InstructionExample]:
    out = []
    for c in CONCEPTS:
        for i in range(per_concept):
            name = (c.name, *c.aliases)[i % (1 + len(c.aliases))]
            template = _WRITE_TEMPLATES[i % len(_WRITE_TEMPLATES)]
            prompt = template.format(sig=f"{name}({c.params})", task=c.task, name=name,
                                     params=c.params.replace(", ", " and ") or "no arguments")
            out.append(InstructionExample(prompt, f"```python\n{c.source(name)}\n```"))
    rng.shuffle(out)
    return out


# Everyday bugs the AST injector cannot make: off-by-one range ends, boundary comparisons, wrong
# index or builtin, a dropped reversal, the wrong variable returned. Each is applied to one line.
_TEXT_MUTATIONS = (
    (" + 1)", ")"), ("<=", "<"), (">=", ">"), (" < ", " <= "), (" > ", " >= "), ("[::-1]", ""),
    ("[0]", "[-1]"), ("[-1]", "[0]"), ("max(", "min("), ("min(", "max("), (".upper()", ".lower()"),
    (".lower()", ".upper()"), ("reverse=True", "reverse=False"), (" not in ", " in "), (" in ", " not in "),
    ("x * x", "x * 2"), ("return a\n", "return b\n"), ("return b\n", "return a\n"), ("+= 1", "+= 2"),
    ("[:n]", "[:n - 1]"), ("[n:]", "[n + 1:]"), ("startswith", "endswith"), ("endswith", "startswith"),
    ("[i:i + size]", "[i:i + size - 1]"), ("== 0", "!= 0"), (" // ", " / "), ("2 * (", "("),
)


def _text_bugs(fixed: str) -> list[tuple[str, str, str]]:
    """(buggy_source, buggy_line, fixed_line) for each single-line text mutation that applies."""
    lines = fixed.splitlines(keepends=True)
    bugs = []
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("def ", '"""')):
            continue
        for old, new in _TEXT_MUTATIONS:
            if old in line:
                mutated = line.replace(old, new, 1)
                buggy = "".join(lines[:i] + [mutated] + lines[i + 1:])
                bugs.append((buggy, mutated.strip(), line.strip()))
    return bugs


def bugfix_examples(rng: random.Random, per_concept: int = 5) -> list[InstructionExample]:
    out = []
    for c in CONCEPTS:
        names = (c.name, *c.aliases)
        made = set()
        name = names[0]
        candidates = _text_bugs(_with_doc(c, name, c.body))
        rng.shuffle(candidates)
        for attempt in range(per_concept * 6):
            if len(made) >= per_concept:
                break
            name = names[attempt % len(names)]
            fixed = _with_doc(c, name, c.body)
            if attempt % 2 == 0 and candidates:
                buggy, wrong, right = candidates.pop()
                buggy = buggy.replace(f"def {names[0]}(", f"def {name}(", 1)
                fixed_for_bug = _with_doc(c, name, c.body)
                bug = (buggy, wrong, right) if fixed_for_bug == fixed else None
            else:
                bug = inject_bug(fixed, rng)
            if bug is None:
                continue
            buggy, wrong, right = bug
            if buggy in made or not _parses(buggy) or _passes(buggy, _rename_tests(c.tests, name)):
                continue  # a realistic bug: valid Python that actually breaks the function
            made.add(buggy)
            answer = f"The bug is `{wrong}`: it should be `{right}`.\n```python\n{fixed}\n```"
            out.append(InstructionExample(BUGFIX_PROMPT.format(code=buggy), answer))
    rng.shuffle(out)
    return out


def basics_examples(seed: int = 1337, *, write_per_concept: int = 6,
                    bugfix_per_concept: int = 5) -> list[InstructionExample]:
    verify_concepts()
    assert_no_overlap()
    rng = random.Random(seed)
    examples = write_examples(rng, write_per_concept) + bugfix_examples(rng, bugfix_per_concept)
    rng.shuffle(examples)
    return examples


def write_basics_dataset(out_dir, tokenizer_path, *, replay_dir=None, replay: int = 0, seed: int = 1337,
                         log=print) -> dict:
    """Write an SFT dataset of basics examples, optionally mixed with ``replay`` examples drawn from
    an earlier SFT dataset (so fine-tuning does not erase what the model already learned)."""
    from pathlib import Path

    import numpy as np

    from ycode.lm.data import encode_example
    from ycode.lm.tokenizer import BPETokenizer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = BPETokenizer.load(tokenizer_path)
    tok.save(out_dir / "tokenizer.json")
    rng = random.Random(seed)
    rows = [encode_example(tok, ex) for ex in basics_examples(seed)]
    n_basics = len(rows)
    if replay_dir and replay:
        replay_dir = Path(replay_dir)
        tokens = np.memmap(replay_dir / "sft_tokens.bin", dtype=np.uint16, mode="r")
        mask = np.memmap(replay_dir / "sft_mask.bin", dtype=np.uint8, mode="r")
        index = np.load(replay_dir / "sft_index.npy")
        for i in rng.sample(range(len(index)), min(replay, len(index))):
            start, length = index[i]
            if length <= 400:  # short examples, like the basics ones
                rows.append((tokens[start:start + length].tolist(), mask[start:start + length].tolist()))
    rng.shuffle(rows)
    all_tokens, all_mask, idx = [], [], []
    for t, m in rows:
        idx.append((len(all_tokens), len(t)))
        all_tokens.extend(t)
        all_mask.extend(m)
    np.asarray(all_tokens, dtype=np.uint16).tofile(out_dir / "sft_tokens.bin")
    np.asarray(all_mask, dtype=np.uint8).tofile(out_dir / "sft_mask.bin")
    np.save(out_dir / "sft_index.npy", np.asarray(idx, dtype=np.int64).reshape(-1, 2))
    info = {"basics": n_basics, "replay": len(rows) - n_basics, "tokens": len(all_tokens)}
    log(f"Wrote {len(rows)} examples ({n_basics} basics + {len(rows) - n_basics} replay), "
        f"{len(all_tokens):,} tokens, to {out_dir}")
    return info
