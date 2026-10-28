"""Extra idiom families: more everyday programming concepts, in the format of ``idioms.FAMILIES``.

The families follow the conventions of :mod:`ycode.lm.idioms`: inputs come from the kinds that
``idioms._value`` generates, every task completes "Write a Python function `sig` that ...", and
every family has two or three implementations that ``idioms.candidate_programs`` runs and compares
on generated inputs. Level 1 is one idiom, level 2 several steps or state, level 3 a real algorithm
(sliding windows, prefix sums, two pointers, stacks, greedy choices, dynamic programming, grid search).
"""

from __future__ import annotations

from ycode.lm.idioms import _MINMAX, _PRED, NUM, ORD, Family, _k

EXTRA_FAMILIES: tuple[Family, ...] = (
    # ---- level 1: numbers and digits
    Family("count_digit", "n", ("nonneg",), ("count_digit_{k}", "digit_{k}_count"),
           ("returns how many times the digit {k} appears among the decimal digits of the non-negative integer n "
            "(0 is written as the single digit 0).",),
           ("return str(n).count('{k}')",
            "count = 0\nwhile True:\n    if n % 10 == {k}:\n        count += 1\n    n //= 10\n    if n == 0:\n"
            "        return count",
            "return sum(1 for d in str(n) if d == '{k}')"), _k(1, 5, 7)),
    Family("digit_sum_by_parity", "n", ("nonneg",), ("sum_of_{nm}_digits", "{nm}_digit_total"),
           ("returns the sum of the {nm} digits of the non-negative integer n (0 if it has none).",),
           ("return sum(int(d) for d in str(n) if int(d) % 2 == {r})",
            "total = 0\nwhile n > 0:\n    d = n % 10\n    if d % 2 == {r}:\n        total += d\n    n //= 10\nreturn total"),
           ({"nm": "even", "r": 0}, {"nm": "odd", "r": 1})),
    Family("count_one_bits", "n", ("nonneg",), ("count_one_bits", "ones_in_binary"),
           ("returns how many 1s there are in the binary representation of the non-negative integer n.",),
           ("return bin(n).count('1')", "count = 0\nwhile n > 0:\n    count += n % 2\n    n //= 2\nreturn count",
            "return sum(1 for b in format(n, 'b') if b == '1')")),
    Family("round_to_nearest", "n", ("int",), ("round_to_nearest_{k}", "nearest_multiple_of_{k}"),
           ("returns the multiple of {k} that is nearest to the integer n; when n lies exactly halfway between two "
            "multiples, the larger of the two.",),
           ("return (n + {h}) // {k} * {k}", "r = n % {k}\nif r * 2 < {k}:\n    return n - r\nreturn n - r + {k}"),
           ({"k": 10, "h": 5}, {"k": 4, "h": 2}, {"k": 5, "h": 2})),
    Family("sum_of_range", "low, high", ("lo", "hi"), ("sum_from_to", "range_total"),
           ("returns the sum of all the integers from low to high, both included (low <= high).",),
           ("return sum(range(low, high + 1))", "return (low + high) * (high - low + 1) // 2",
            "total = 0\nx = low\nwhile x <= high:\n    total += x\n    x += 1\nreturn total")),
    Family("multiples_in_range", "low, high", ("lo", "hi"), ("multiples_of_{k}_between", "count_multiples_of_{k}_in_range"),
           ("returns how many of the integers from low to high, both included (low <= high), are divisible by {k}.",),
           ("return sum(1 for x in range(low, high + 1) if x % {k} == 0)", "return high // {k} - (low - 1) // {k}",
            "count = 0\nfor x in range(low, high + 1):\n    if x % {k} == 0:\n        count += 1\nreturn count"),
           _k(3, 5, 7)),
    # ---- level 1: lists and sets
    Family("remove_value", "items, x", ("small_list", "small"), ("remove_all", "without_value"),
           ("returns a new list with every occurrence of x removed from items; the other items keep their order.",),
           ("return [v for v in items if v != x]",
            "result = []\nfor v in items:\n    if v != x:\n        result.append(v)\nreturn result",
            "return list(filter(lambda v: v != x, items))")),
    Family("replace_value", "items, old, new", ("small_list", "small", "small"), ("replace_value", "substitute_value"),
           ("returns a new list in which every item equal to old is replaced by new.",),
           ("return [new if v == old else v for v in items]",
            "result = list(items)\nfor i in range(len(result)):\n    if result[i] == old:\n        result[i] = new\nreturn result")),
    Family("values_set_op", "a, b", ("ints", "ints"), ("values_{nm}", "{nm}_values"),
           ("returns, in increasing order and without repeats, the values that {desc}.",),
           ("return sorted(set(a) {op} set(b))", "return sorted({{x for x in a + b if {cond}}})",
            "result = []\nfor x in sorted(a + b):\n    if {cond} and x not in result:\n        result.append(x)\nreturn result"),
           ({"nm": "in_both", "desc": "appear in both list a and list b", "op": "&", "cond": "x in a and x in b"},
            {"nm": "only_in_first", "desc": "appear in list a but not in list b", "op": "-", "cond": "x not in b"},
            {"nm": "in_exactly_one", "desc": "appear in exactly one of the lists a and b", "op": "^",
             "cond": "(x in a) != (x in b)"},
            {"nm": "in_either", "desc": "appear in list a or in list b (or in both)", "op": "|",
             "cond": "(x in a or x in b)"})),
    Family("all_present", "a, b", ("small_list", "small_list"), ("all_present_in", "is_covered_by"),
           ("returns True if every value of list a also appears in list b (True when a is empty).",),
           ("return set(a) <= set(b)", "return all(x in b for x in a)",
            "for x in a:\n    if x not in b:\n        return False\nreturn True")),
    Family("is_ordered", "numbers", ("ints",), ("is_{nm}", "in_{nm}_order"),
           ("returns True if every number of the list is {desc} the number before it (True for lists shorter than two).",),
           ("return all(b {op} a for a, b in zip(numbers, numbers[1:]))",
            "for i in range(1, len(numbers)):\n    if not numbers[i] {op} numbers[i - 1]:\n        return False\nreturn True"),
           ({"nm": "nondecreasing", "desc": "greater than or equal to", "op": ">="},
            {"nm": "increasing", "desc": "greater than", "op": ">"},
            {"nm": "nonincreasing", "desc": "less than or equal to", "op": "<="})),
    Family("flatten_rows", "rows", ("matrix",), ("flatten", "all_cells_in_order"),
           ("returns one list with the numbers of all the rows of the list of lists rows, row after row.",),
           ("return [x for row in rows for x in row]", "result = []\nfor row in rows:\n    result.extend(row)\nreturn result",
            "import itertools\nreturn list(itertools.chain.from_iterable(rows))")),
    Family("total_gains", "prices", ("ints",), ("sum_of_rises", "total_gain"),
           ("returns the sum of all the increases from each number of the list to the next one (decreases are ignored; "
            "0 if there are no increases).",),
           ("return sum(max(b - a, 0) for a, b in zip(prices, prices[1:]))",
            "total = 0\nfor i in range(1, len(prices)):\n    if prices[i] > prices[i - 1]:\n"
            "        total += prices[i] - prices[i - 1]\nreturn total")),
    Family("sort_numbers_by", "numbers", ("ints",), ("sort_by_{nm}", "order_by_{nm}"),
           ("returns the numbers sorted by {desc}; numbers that tie keep their original order.",),
           ("return sorted(numbers, key=lambda x: {key})",
            "decorated = [({key}, i, x) for i, x in enumerate(numbers)]\nreturn [x for _, _, x in sorted(decorated)]",
            "def key(x):\n    return {key}\nresult = []\nfor x in numbers:\n    i = len(result)\n"
            "    while i > 0 and key(result[i - 1]) > key(x):\n        i -= 1\n    result.insert(i, x)\nreturn result"),
           ({"nm": "absolute_value", "key": "abs(x)", "desc": "absolute value, smallest first"},
            {"nm": "last_digit", "key": "abs(x) % 10", "desc": "the last digit of their absolute value, smallest first"},
            {"nm": "distance_from_ten", "key": "abs(x - 10)", "desc": "their distance from 10, closest first"})),
    # ---- level 1: strings
    Family("capitalized_words", "text", ("text",), ("capitalized_words", "words_with_capitals"),
           ("returns a list of the words of text whose first character is an upper-case letter, in their original order.",),
           ("return [w for w in text.split() if w[0].isupper()]",
            "result = []\nfor w in text.split():\n    if w[0].isupper():\n        result.append(w)\nreturn result")),
    Family("swap_case", "s", ("str",), ("swap_case", "invert_letter_case"),
           ("returns s with every upper-case letter turned into lower case and every lower-case letter into upper case "
            "(other characters stay as they are).",),
           ("return s.swapcase()",
            "result = ''\nfor c in s:\n    if c.isupper():\n        result += c.lower()\n    elif c.islower():\n"
            "        result += c.upper()\n    else:\n        result += c\nreturn result",
            "return ''.join(c.lower() if c.isupper() else c.upper() for c in s)")),
    Family("alternate_case", "s", ("str",), ("alternate_case_{nm}", "zigzag_case_{nm}"),
           ("returns s with the characters at {nm} positions ({ex}) in upper case and all other characters in lower case.",),
           ("return ''.join(c.upper() if i % 2 == {r} else c.lower() for i, c in enumerate(s))",
            "result = ''\nfor i in range(len(s)):\n    result += s[i].upper() if i % 2 == {r} else s[i].lower()\nreturn result"),
           ({"nm": "even", "ex": "0, 2, 4, ...", "r": 0}, {"nm": "odd", "ex": "1, 3, 5, ...", "r": 1})),
    Family("camel_to_snake", "s", ("str",), ("camel_to_snake", "underscore_capitals"),
           ("returns s with every upper-case letter replaced by '_' followed by the same letter in lower case (other "
            "characters stay as they are).",),
           ("return ''.join('_' + c.lower() if c.isupper() else c for c in s)",
            "result = ''\nfor c in s:\n    if c.isupper():\n        result += '_' + c.lower()\n    else:\n        result += c\n"
            "return result",
            "import re\nreturn re.sub('[A-Z]', lambda m: '_' + m.group().lower(), s)")),
    Family("slug_words", "text", ("text",), ("{nm}_case", "to_{nm}_case"),
           ("returns the words of text in lower case joined with '{sep}' ('' if text has no words).",),
           ("return '{sep}'.join(text.lower().split())",
            "result = ''\nfor w in text.split():\n    if result:\n        result += '{sep}'\n    result += w.lower()\nreturn result",
            "return '{sep}'.join(w.lower() for w in text.split())"),
           ({"nm": "kebab", "sep": "-"}, {"nm": "snake", "sep": "_"})),
    Family("longest_word", "text", ("text",), ("longest_word", "first_longest_word"),
           ("returns the longest word of text (the first one if several are equally long, '' if text has no words).",),
           ("return max(text.split(), key=len, default='')",
            "best = ''\nfor w in text.split():\n    if len(w) > len(best):\n        best = w\nreturn best")),
    Family("sort_words_by", "words", ("strs",), ("sort_words_by_{nm}", "words_ordered_by_{nm}"),
           ("returns the words sorted by {desc}; words that tie keep their original order.",),
           ("return sorted(words, key=lambda w: {key})",
            "decorated = [({key}, i, w) for i, w in enumerate(words)]\nreturn [w for _, _, w in sorted(decorated)]",
            "def key(w):\n    return {key}\nresult = []\nfor w in words:\n    i = len(result)\n"
            "    while i > 0 and key(result[i - 1]) > key(w):\n        i -= 1\n    result.insert(i, w)\nreturn result"),
           ({"nm": "length", "key": "len(w)", "desc": "length, shortest first"},
            {"nm": "name", "key": "w.lower()", "desc": "alphabetical order, ignoring upper and lower case"},
            {"nm": "last_letter", "key": "w[-1].lower()", "desc": "their last letter, ignoring upper and lower case"})),
    Family("truncate", "s", ("str",), ("truncate_{k}", "shorten_to_{kw}"),
           ("returns s unchanged if it has at most {k} characters, otherwise its first {k} characters followed by '...'.",),
           ("return s if len(s) <= {k} else s[:{k}] + '...'", "if len(s) > {k}:\n    return s[:{k}] + '...'\nreturn s",
            "return s[:{k}] + '...' * (len(s) > {k})"), _k(4, 6, 8)),
    Family("repeat_chars", "s", ("str",), ("repeat_each_char_{k}", "stretch_{kw}_times"),
           ("returns s with every character repeated {kw} times in a row ('ab' gives '{ex}').",),
           ("return ''.join(c * {k} for c in s)",
            "result = ''\nfor c in s:\n    for _ in range({k}):\n        result += c\nreturn result"),
           ({"k": 2, "kw": "two", "ex": "aabb"}, {"k": 3, "kw": "three", "ex": "aaabbb"})),
    Family("mask_but_last", "s", ("str",), ("mask_all_but_last_{k}", "hide_except_last_{kw}"),
           ("returns s with every character except the last {kw} replaced by '*' (unchanged if s has at most {k} "
            "characters).",),
           ("return '*' * max(len(s) - {k}, 0) + s[-{k}:]",
            "return ''.join('*' if i < len(s) - {k} else c for i, c in enumerate(s))",
            "result = ''\nfor i, c in enumerate(s):\n    result += c if i >= len(s) - {k} else '*'\nreturn result"), _k(2, 4)),
    Family("numbered_lines", "words", ("strs",), ("numbered_lines_{nm}", "enumerate_lines_{nm}"),
           ("returns one string with a line '<number>{sep}<word>' for each word of the list, numbered from 1, the lines "
            "separated by newlines ('' for an empty list).",),
           ("return '\\n'.join(f'{{i}}{sep}{{w}}' for i, w in enumerate(words, 1))",
            "lines = []\nfor i in range(len(words)):\n    lines.append(str(i + 1) + '{sep}' + words[i])\nreturn '\\n'.join(lines)"),
           ({"sep": ". ", "nm": "dot"}, {"sep": ") ", "nm": "paren"})),
    # ---- level 1: dictionaries
    Family("count_by_sign", "numbers", ("ints",), ("count_by_sign", "sign_counts"),
           ("returns a dictionary {{'negative': ..., 'zero': ..., 'positive': ...}} (keys in that order) with how many "
            "numbers of the list are negative, zero and positive.",),
           ("counts = {{'negative': 0, 'zero': 0, 'positive': 0}}\nfor x in numbers:\n    if x < 0:\n"
            "        counts['negative'] += 1\n    elif x == 0:\n        counts['zero'] += 1\n    else:\n"
            "        counts['positive'] += 1\nreturn counts",
            "negative = sum(1 for x in numbers if x < 0)\npositive = sum(1 for x in numbers if x > 0)\n"
            "return {{'negative': negative, 'zero': numbers.count(0), 'positive': positive}}")),
    Family("filter_dict_values", "d", ("dict",), ("keep_{nm}_values", "only_{nm}_entries"),
           ("returns a new dictionary with only the entries of d whose value is {word}, in their original order.",),
           ("return {{k: x for k, x in d.items() if {cond}}}",
            "result = {{}}\nfor k in d:\n    x = d[k]\n    if {cond}:\n        result[k] = x\nreturn result"),
           tuple({"cond": c, "word": w, "nm": n} for c, w, n in (_PRED[0], _PRED[2], _PRED[4]))),
    Family("lookup_with_default", "d, keys", ("dict", "strs"), ("lookup_all_or_{nm}", "values_for_keys_or_{nm}"),
           ("returns a list with the value of d for each key in keys, in order, using {dflt} for a key that is not in d.",),
           ("return [d.get(k, {dflt}) for k in keys]",
            "result = []\nfor k in keys:\n    if k in d:\n        result.append(d[k])\n    else:\n        result.append({dflt})\n"
            "return result"),
           ({"dflt": 0, "nm": "zero"}, {"dflt": -1, "nm": "minus_one"})),
    Family("shared_keys", "a, b", ("dict", "dict"), ("shared_keys", "keys_in_both"),
           ("returns the keys that appear in both dictionaries a and b, as a sorted list.",),
           ("return sorted(set(a) & set(b))", "return sorted(k for k in a if k in b)",
            "result = []\nfor k in a:\n    if k in b:\n        result.append(k)\nresult.sort()\nreturn result")),
    Family("missing_digits", "n", ("nonneg",), ("missing_digits", "unused_digits"),
           ("returns a list of the digits 0 to 9 (as integers, in increasing order) that do not appear in the "
            "non-negative integer n.",),
           ("return [d for d in range(10) if str(d) not in str(n)]",
            "seen = set()\nwhile True:\n    seen.add(n % 10)\n    n //= 10\n    if n == 0:\n        break\n"
            "return [d for d in range(10) if d not in seen]",
            "return sorted(set(range(10)) - {{int(c) for c in str(n)}})")),
    Family("zero_pad", "n", ("nonneg",), ("zero_pad_{k}", "pad_to_{kw}_digits"),
           ("returns the non-negative integer n as a string, with zeros added in front to make it at least {k} "
            "characters long.",),
           ("return str(n).zfill({k})", "s = str(n)\nwhile len(s) < {k}:\n    s = '0' + s\nreturn s",
            "return f'{{n:0{k}d}}'"), _k(3, 5, 6)),
) + (
    # ---- level 2: numbers and digits
    Family("arrange_digits", "n", ("nonneg",), ("{nm}_digit_arrangement", "{nm}_from_same_digits"),
           ("{desc}",),
           ("return int(''.join(sorted(str(n){rev})))",
            "counts = [0] * 10\nwhile True:\n    counts[n % 10] += 1\n    n //= 10\n    if n == 0:\n        break\n"
            "result = 0\nfor d in {order}:\n    for _ in range(counts[d]):\n        result = result * 10 + d\nreturn result"),
           ({"nm": "largest", "rev": ", reverse=True", "order": "range(9, -1, -1)",
             "desc": "returns the largest number that can be written using exactly the digits of the non-negative "
                     "integer n (each digit as many times as it occurs in n)."},
            {"nm": "smallest", "rev": "", "order": "range(10)",
             "desc": "returns the number obtained by writing the digits of the non-negative integer n in increasing "
                     "order (leading zeros disappear, so 302 gives 23)."}), level=2),
    Family("to_base", "n", ("nonneg",), ("to_base_{k}", "in_base_{k}"),
           ("returns the base-{k} representation of the non-negative integer n as a string of digits ('0' for 0).",),
           ("if n == 0:\n    return '0'\ndigits = []\nwhile n > 0:\n    digits.append(str(n % {k}))\n    n //= {k}\n"
            "return ''.join(reversed(digits))",
            "return format(n, '{f}')",
            "s = ''\nwhile True:\n    s = str(n % {k}) + s\n    n //= {k}\n    if n == 0:\n        return s"),
           ({"k": 2, "f": "b"}, {"k": 8, "f": "o"}), level=2),
    Family("count_divisors", "n", ("pos",), ("count_divisors", "number_of_divisors"),
           ("returns how many positive integers divide the positive integer n exactly (1 and n included).",),
           ("return sum(1 for d in range(1, n + 1) if n % d == 0)",
            "count = 0\nd = 1\nwhile d * d <= n:\n    if n % d == 0:\n        count += 1 if d * d == n else 2\n    d += 1\n"
            "return count"), level=2),
    Family("greedy_change", "amount", ("pos",), ("coins_for_{nm}", "greedy_coin_count_{nm}"),
           ("returns how many coins are used to pay the positive amount with coins of {desc}, choosing greedily: "
            "always the largest coin that is not more than what is left to pay.",),
           ("count = 0\nfor coin in {coins}:\n    count += amount // coin\n    amount %= coin\nreturn count",
            "count = 0\nfor coin in {coins}:\n    while amount >= coin:\n        amount -= coin\n        count += 1\n"
            "return count"),
           ({"coins": "(25, 10, 5, 1)", "desc": "25, 10, 5 and 1", "nm": "cents"},
            {"coins": "(50, 20, 10, 5, 2, 1)", "desc": "50, 20, 10, 5, 2 and 1", "nm": "euro_cents"}), level=2),
    Family("clock_text", "minutes", ("nonneg",), ("clock_text", "as_hours_and_minutes"),
           ("returns the non-negative number of minutes written as hours and minutes in the form 'H:MM', with the "
            "minutes always two digits and the hours not limited to 24 (125 gives '2:05').",),
           ("return f'{{minutes // 60}}:{{minutes % 60:02d}}'",
            "hours, rest = divmod(minutes, 60)\nreturn str(hours) + ':' + str(rest).zfill(2)"), level=2),
    Family("ordinal_suffix", "n", ("pos",), ("ordinal", "with_ordinal_suffix"),
           ("returns the positive integer n as an English ordinal: 1 gives '1st', 2 gives '2nd', 3 gives '3rd', 4 gives "
            "'4th', 11, 12 and 13 give '11th', '12th' and '13th', 21 gives '21st', and so on.",),
           ("if 11 <= n % 100 <= 13:\n    return str(n) + 'th'\n"
            "return str(n) + {{1: 'st', 2: 'nd', 3: 'rd'}}.get(n % 10, 'th')",
            "last = n % 10\nif last == 1 and n % 100 != 11:\n    suffix = 'st'\nelif last == 2 and n % 100 != 12:\n"
            "    suffix = 'nd'\nelif last == 3 and n % 100 != 13:\n    suffix = 'rd'\nelse:\n    suffix = 'th'\n"
            "return f'{{n}}{{suffix}}'"), level=2),
    Family("luhn_check_digit", "n", ("nonneg",), ("luhn_check_digit", "check_digit_for"),
           ("returns the Luhn check digit of the non-negative integer n: double the first, third, fifth, ... digits of "
            "n counted from the right (subtracting 9 from a doubled value above 9), add up all the digits obtained this "
            "way, and return the digit (0 to 9) that brings this total up to a multiple of 10.",),
           ("total = 0\nfor i, c in enumerate(reversed(str(n))):\n    d = int(c)\n    if i % 2 == 0:\n        d *= 2\n"
            "        if d > 9:\n            d -= 9\n    total += d\nreturn (10 - total % 10) % 10",
            "total = 0\ndouble = True\nwhile True:\n    d = n % 10\n    if double:\n        d = d * 2 - 9 if d * 2 > 9 else d * 2\n"
            "    total += d\n    double = not double\n    n //= 10\n    if n == 0:\n        break\nreturn -total % 10"),
           level=2),
    # ---- level 2: lists (windows, records, ranks, partitions, searching, counting)
    Family("window_extremes", "numbers", ("ints",), ("window_{nm}_{k}", "sliding_{nm}_of_{kw}"),
           ("returns a list with the {big} of every {kw} consecutive numbers of the list, window by window from the left "
            "(an empty list if there are fewer than {kw} numbers).",),
           ("return [{fn}(numbers[i:i + {k}]) for i in range(len(numbers) - {k} + 1)]",
            "result = []\nfor i in range({k} - 1, len(numbers)):\n    best = numbers[i]\n    for j in range(i - {k} + 1, i):\n"
            "        if numbers[j] {cmp} best:\n            best = numbers[j]\n    result.append(best)\nreturn result",
            "return [{fn}(window) for window in zip(*(numbers[j:] for j in range({k})))]"),
           tuple({**m, **k} for m in _MINMAX for k in _k(2, 3)), level=2),
    Family("running_extreme", "numbers", ("ints",), ("running_{nm}", "{nm}_so_far"),
           ("returns a list whose i-th item is the {big} of the first i + 1 numbers of the list.",),
           ("result = []\nfor x in numbers:\n    result.append(x if not result else {fn}(result[-1], x))\nreturn result",
            "import itertools\nreturn list(itertools.accumulate(numbers, {fn}))",
            "return [{fn}(numbers[:i + 1]) for i in range(len(numbers))]"), _MINMAX, level=2),
    Family("count_records", "numbers", ("ints",), ("count_new_{nm}s", "record_{nm}s"),
           ("returns how many numbers of the list are {word} every number before them (the first number always counts; "
            "0 for an empty list).",),
           ("count = 0\nbest = None\nfor x in numbers:\n    if best is None or x {cmp} best:\n        count += 1\n"
            "        best = x\nreturn count",
            "return sum(1 for i, x in enumerate(numbers) if all(x {cmp} y for y in numbers[:i]))"),
           ({"cmp": ">", "word": "greater than", "nm": "high"}, {"cmp": "<", "word": "smaller than", "nm": "low"}),
           level=2),
    Family("leaders", "numbers", ("ints",), ("leaders", "greater_than_all_after"),
           ("returns, in their original order, the numbers of the list that are greater than every number after them "
            "(the last number always qualifies).",),
           ("result = []\nbest = None\nfor x in reversed(numbers):\n    if best is None or x > best:\n"
            "        result.append(x)\n        best = x\nreturn result[::-1]",
            "return [x for i, x in enumerate(numbers) if all(x > y for y in numbers[i + 1:])]"), level=2),
    Family("second_extreme", "numbers", ("ints",), ("second_{big}", "runner_up_{nm}"),
           ("returns the second {big} distinct value in the list, or None if the list has fewer than two distinct "
            "values.",),
           ("values = sorted(set(numbers){rev})\nreturn values[1] if len(values) > 1 else None",
            "first = second = None\nfor x in numbers:\n    if first is None or x {cmp} first:\n        first, second = x, first\n"
            "    elif x != first and (second is None or x {cmp} second):\n        second = x\nreturn second"),
           ({"big": "largest", "nm": "max", "cmp": ">", "rev": ", reverse=True"},
            {"big": "smallest", "nm": "min", "cmp": "<", "rev": ""}), level=2),
    Family("dense_ranks", "numbers", ("ints",), ("dense_ranks", "rank_each_number"),
           ("returns a list with the rank of each number: 1 for the smallest value, 2 for the next larger distinct value, "
            "and so on (equal numbers get the same rank).",),
           ("rank = {{x: i + 1 for i, x in enumerate(sorted(set(numbers)))}}\nreturn [rank[x] for x in numbers]",
            "return [len({{y for y in numbers if y < x}}) + 1 for x in numbers]",
            "distinct = sorted(set(numbers))\nresult = []\nfor x in numbers:\n    result.append(distinct.index(x) + 1)\n"
            "return result"), level=2),
    Family("move_to_end", "items, x", ("small_list", "small"), ("move_to_end", "send_value_to_back"),
           ("returns a new list with every occurrence of x moved to the end of the list; the other items keep their "
            "order.",),
           ("return [v for v in items if v != x] + [v for v in items if v == x]",
            "return sorted(items, key=lambda v: v == x)",
            "result = []\ncount = 0\nfor v in items:\n    if v == x:\n        count += 1\n    else:\n        result.append(v)\n"
            "return result + [x] * count"), level=2),
    Family("split_on_value", "items, sep", ("small_list", "small"), ("split_on_value", "pieces_between_separators"),
           ("returns the pieces of the list between the occurrences of sep, as a list of lists, the way str.split works "
            "for strings: neighbouring separators give an empty piece, a list without sep gives one piece, and an empty "
            "list gives [[]].",),
           ("pieces = [[]]\nfor v in items:\n    if v == sep:\n        pieces.append([])\n    else:\n"
            "        pieces[-1].append(v)\nreturn pieces",
            "pieces = []\nstart = 0\nfor i, v in enumerate(items):\n    if v == sep:\n        pieces.append(items[start:i])\n"
            "        start = i + 1\npieces.append(items[start:])\nreturn pieces"), level=2),
    Family("add_lists_padded", "a, b", ("ints", "ints"), ("add_lists", "elementwise_sum_padded"),
           ("returns a list with a[i] + b[i] at each position i; where one list is longer than the other, its extra "
            "numbers are kept as they are.",),
           ("import itertools\nreturn [x + y for x, y in itertools.zip_longest(a, b, fillvalue=0)]",
            "result = []\nfor i in range(max(len(a), len(b))):\n    x = a[i] if i < len(a) else 0\n"
            "    y = b[i] if i < len(b) else 0\n    result.append(x + y)\nreturn result",
            "if len(a) < len(b):\n    a, b = b, a\nreturn [x + (b[i] if i < len(b) else 0) for i, x in enumerate(a)]"),
           level=2),
    Family("first_repeat", "items", ("ints",), ("first_repeat", "first_seen_twice"),
           ("returns the first item of the list, scanning from the left, that is equal to an earlier item, or None if "
            "all items are different.",),
           ("seen = set()\nfor x in items:\n    if x in seen:\n        return x\n    seen.add(x)\nreturn None",
            "for i, x in enumerate(items):\n    if x in items[:i]:\n        return x\nreturn None"), level=2),
    Family("smallest_missing_positive", "numbers", ("ints",), ("smallest_missing_positive", "first_absent_positive"),
           ("returns the smallest positive integer that does not appear in the list.",),
           ("present = set(numbers)\nk = 1\nwhile k in present:\n    k += 1\nreturn k",
            "return next(k for k in range(1, len(numbers) + 2) if k not in numbers)",
            "k = 1\nfor x in sorted(set(numbers)):\n    if x == k:\n        k += 1\n    elif x > k:\n        break\nreturn k"),
           level=2),
    Family("missing_between", "numbers", ("sorted_distinct",), ("missing_between", "gaps_in_sorted"),
           ("returns, in increasing order, the integers between the smallest and the largest number of the sorted list "
            "of distinct numbers that do not appear in it (an empty list if it has fewer than two numbers).",),
           ("if not numbers:\n    return []\npresent = set(numbers)\n"
            "return [x for x in range(numbers[0], numbers[-1]) if x not in present]",
            "result = []\nfor a, b in zip(numbers, numbers[1:]):\n    result.extend(range(a + 1, b))\nreturn result"),
           level=2),
    Family("longest_monotone_run", "numbers", ("ints",), ("longest_{nm}_stretch", "longest_{nm}_streak"),
           ("returns the length of the longest stretch of neighbouring numbers in which each number is {desc} the one "
            "before it (1 for a single number, 0 for an empty list).",),
           ("best = 0\nstart = 0\nfor i in range(len(numbers)):\n    if i > 0 and not numbers[i] {op} numbers[i - 1]:\n"
            "        start = i\n    best = max(best, i - start + 1)\nreturn best",
            "if not numbers:\n    return 0\nlengths = [1]\nfor a, b in zip(numbers, numbers[1:]):\n"
            "    lengths.append(lengths[-1] + 1 if b {op} a else 1)\nreturn max(lengths)"),
           ({"nm": "rising", "desc": "greater than", "op": ">"}, {"nm": "falling", "desc": "smaller than", "op": "<"}),
           level=2),
    Family("closest_to", "numbers, target", ("ints1", "int_small"), ("closest_to", "nearest_value_to"),
           ("returns the number of the non-empty list that is closest to target; of two equally close numbers, the "
            "smaller one.",),
           ("return min(numbers, key=lambda x: (abs(x - target), x))",
            "best = numbers[0]\nfor x in numbers[1:]:\n    d, e = abs(x - target), abs(best - target)\n"
            "    if d < e or (d == e and x < best):\n        best = x\nreturn best"), level=2),
    Family("pairs_with_difference", "numbers", ("ints",), ("pairs_differing_by_{k}", "count_gaps_of_{kw}"),
           ("returns how many pairs of positions i < j hold numbers that differ by exactly {k} (in either direction).",),
           ("count = 0\nfor i in range(len(numbers)):\n    for j in range(i + 1, len(numbers)):\n"
            "        if abs(numbers[i] - numbers[j]) == {k}:\n            count += 1\nreturn count",
            "from collections import Counter\ncounts = Counter(numbers)\n"
            "return sum(counts[x] * counts[x + {k}] for x in counts)",
            "import itertools\nreturn sum(1 for a, b in itertools.combinations(numbers, 2) if abs(a - b) == {k})"),
           _k(1, 2, 3), level=2),
    Family("pairs_sum_divisible", "numbers", ("ints",), ("pairs_divisible_by_{k}", "pair_sums_multiple_of_{kw}"),
           ("returns how many pairs of positions i < j have numbers[i] + numbers[j] divisible by {k}.",),
           ("count = 0\nfor i in range(len(numbers)):\n    for j in range(i + 1, len(numbers)):\n"
            "        if (numbers[i] + numbers[j]) % {k} == 0:\n            count += 1\nreturn count",
            "seen = [0] * {k}\ncount = 0\nfor x in numbers:\n    count += seen[-x % {k}]\n    seen[x % {k}] += 1\nreturn count"),
           _k(3, 5), level=2),
    Family("reverse_in_groups", "items", ("ints",), ("reverse_groups_of_{kw}", "flip_each_{kw}"),
           ("returns the list with each consecutive group of {kw} items reversed (a shorter last group is reversed too).",),
           ("result = []\nfor i in range(0, len(items), {k}):\n    result.extend(reversed(items[i:i + {k}]))\nreturn result",
            "return [x for i in range(0, len(items), {k}) for x in items[i:i + {k}][::-1]]",
            "result = list(items)\nfor i in range(0, len(result), {k}):\n    result[i:i + {k}] = result[i:i + {k}][::-1]\n"
            "return result"), _k(2, 3), level=2),
    Family("above_average", "numbers", ("ints",), ("count_above_average", "how_many_above_mean"),
           ("returns how many numbers of the list are strictly greater than the average of the list (0 for an empty "
            "list).",),
           ("if not numbers:\n    return 0\naverage = sum(numbers) / len(numbers)\n"
            "return sum(1 for x in numbers if x > average)",
            "total = sum(numbers)\nreturn sum(1 for x in numbers if x * len(numbers) > total)"), level=2),
    # ---- level 2: grids
    Family("column_sums", "rows", ("matrix",), ("column_sums", "sum_each_column"),
           ("returns a list with the sum of each column of the rectangular grid rows (an empty list for an empty grid).",),
           ("return [sum(col) for col in zip(*rows)]",
            "if not rows:\n    return []\nsums = [0] * len(rows[0])\nfor row in rows:\n    for j, x in enumerate(row):\n"
            "        sums[j] += x\nreturn sums"), level=2),
    Family("border_sum", "grid", ("matrix1",), ("border_sum", "edge_total"),
           ("returns the sum of the numbers on the outer border of the non-empty rectangular grid (its first and last "
            "row and its first and last column), counting each cell once.",),
           ("n, m = len(grid), len(grid[0])\n"
            "return sum(grid[i][j] for i in range(n) for j in range(m) if i in (0, n - 1) or j in (0, m - 1))",
            "total = sum(grid[0])\nif len(grid) > 1:\n    total += sum(grid[-1])\nfor row in grid[1:-1]:\n"
            "    total += row[0]\n    if len(row) > 1:\n        total += row[-1]\nreturn total"), level=2),
    Family("rotate_grid", "rows", ("matrix",), ("rotate_{nm}", "quarter_turn_{nm}"),
           ("returns the rectangular grid rows turned a quarter turn {nm}, as a new list of lists ([] for an empty "
            "grid).",),
           ("return {zipped}",
            "if not rows:\n    return []\nn, m = len(rows), len(rows[0])\n"
            "return [[{cell} for i in range(n)] for j in range(m)]"),
           ({"nm": "clockwise", "zipped": "[list(r) for r in zip(*rows[::-1])]", "cell": "rows[n - 1 - i][j]"},
            {"nm": "counterclockwise", "zipped": "[list(r) for r in zip(*rows)][::-1]", "cell": "rows[i][m - 1 - j]"}),
           level=2),
    # ---- level 2: strings
    Family("camel_case", "text", ("text",), ("camel_case", "to_camel_case"),
           ("returns the words of text joined without spaces in camelCase: the first word all in lower case, every "
            "other word with its first letter in upper case and the rest in lower case ('' if text has no words).",),
           ("words = text.split()\nif not words:\n    return ''\n"
            "return words[0].lower() + ''.join(w.capitalize() for w in words[1:])",
            "result = ''\nfor i, w in enumerate(text.split()):\n"
            "    result += w.lower() if i == 0 else w[0].upper() + w[1:].lower()\nreturn result"), level=2),
    Family("count_overlapping", "s, part", ("runs", "runs"), ("count_overlapping", "occurrences_with_overlap"),
           ("returns how many times part occurs in s, counting occurrences that overlap ('aaa' contains 'aa' twice); "
            "0 if part is empty.",),
           ("if not part:\n    return 0\n"
            "return sum(1 for i in range(len(s) - len(part) + 1) if s[i:i + len(part)] == part)",
            "if part == '':\n    return 0\ncount = 0\nstart = s.find(part)\nwhile start != -1:\n    count += 1\n"
            "    start = s.find(part, start + 1)\nreturn count",
            "return sum(1 for i in range(len(s)) if part and s.startswith(part, i))"), level=2),
    Family("is_subsequence", "part, s", ("word0", "word0"), ("is_subsequence", "letters_in_order"),
           ("returns True if all the characters of part appear in s in the same order, not necessarily next to each "
            "other (True when part is empty).",),
           ("i = 0\nfor c in s:\n    if i < len(part) and c == part[i]:\n        i += 1\nreturn i == len(part)",
            "it = iter(s)\nreturn all(c in it for c in part)",
            "pos = 0\nfor c in part:\n    pos = s.find(c, pos) + 1\n    if pos == 0:\n        return False\nreturn True"),
           level=2),
    Family("first_unique_char", "s", ("str",), ("first_unique_char", "first_unrepeated_char"),
           ("returns the first character of s that occurs only once in s, or '' if there is none.",),
           ("for c in s:\n    if s.count(c) == 1:\n        return c\nreturn ''",
            "from collections import Counter\ncounts = Counter(s)\nreturn next((c for c in s if counts[c] == 1), '')"),
           level=2),
    Family("english_list", "words", ("strs",), ("english_list", "join_with_and"),
           ("returns the words joined the way a list is written in English: '' for no words, 'x' for one, 'x and y' "
            "for two, and 'x, y and z' (commas between all but the last two) for more.",),
           ("if not words:\n    return ''\nif len(words) == 1:\n    return words[0]\n"
            "return ', '.join(words[:-1]) + ' and ' + words[-1]",
            "result = ''\nfor i, w in enumerate(words):\n    if i == 0:\n        result = w\n    elif i == len(words) - 1:\n"
            "        result += ' and ' + w\n    else:\n        result += ', ' + w\nreturn result"), level=2),
    Family("thousands_separator", "n", ("nonneg",), ("with_commas", "group_thousands"),
           ("returns the non-negative integer n as a string with a comma between each group of three digits, counting "
            "from the right (1234567 gives '1,234,567').",),
           ("return f'{{n:,}}'",
            "s = str(n)\ngroups = []\nwhile len(s) > 3:\n    groups.insert(0, s[-3:])\n    s = s[:-3]\ngroups.insert(0, s)\n"
            "return ','.join(groups)",
            "digits = str(n)[::-1]\nparts = [digits[i:i + 3] for i in range(0, len(digits), 3)]\n"
            "return ','.join(parts)[::-1]"), level=2),
    Family("numeronyms", "text", ("text",), ("numeronyms_over_{k}", "abbreviate_words_over_{k}"),
           ("returns the words of text separated by single spaces, where every word longer than {k} characters is "
            "replaced by its first character, the number of characters in between and its last character ('Umbrella' "
            "gives 'U6a').",),
           ("return ' '.join(w[0] + str(len(w) - 2) + w[-1] if len(w) > {k} else w for w in text.split())",
            "out = []\nfor w in text.split():\n    if len(w) > {k}:\n        out.append(f'{{w[0]}}{{len(w) - 2}}{{w[-1]}}')\n"
            "    else:\n        out.append(w)\nreturn ' '.join(out)"), _k(3, 4), level=2),
    Family("split_digit_runs", "text", ("mixed_text",), ("split_digit_runs", "digit_and_other_pieces"),
           ("returns text cut into its longest pieces that consist only of digits or only of non-digits, in order "
            "('ab12x' gives ['ab', '12', 'x']; [] for an empty string).",),
           ("pieces = []\nfor c in text:\n    if pieces and pieces[-1][-1].isdigit() == c.isdigit():\n"
            "        pieces[-1] += c\n    else:\n        pieces.append(c)\nreturn pieces",
            "import re\nreturn re.findall('[0-9]+|[^0-9]+', text)",
            "import itertools\nreturn [''.join(group) for _, group in itertools.groupby(text, key=str.isdigit)]"), level=2),
    Family("nesting_depth", "s", ("brackets",), ("max_nesting", "deepest_parenthesis_level"),
           ("returns the largest depth reached while reading s from left to right, where the depth starts at 0, goes up "
            "by one at each '(' and down by one at each ')' (other characters are ignored).",),
           ("depth = best = 0\nfor c in s:\n    if c == '(':\n        depth += 1\n        best = max(best, depth)\n"
            "    elif c == ')':\n        depth -= 1\nreturn best",
            "import itertools\nsteps = [1 if c == '(' else -1 for c in s if c in '()']\n"
            "return max(itertools.accumulate(steps, initial=0))"), level=2),
    # ---- level 2: dictionaries
    Family("word_positions", "text", ("text",), ("word_positions", "where_each_word_occurs"),
           ("returns a dictionary that maps each distinct word of text to the list of its positions among the words "
            "(starting at 0), with the words in order of first occurrence.",),
           ("result = {{}}\nfor i, w in enumerate(text.split()):\n    result.setdefault(w, []).append(i)\nreturn result",
            "words = text.split()\nreturn {{w: [i for i, v in enumerate(words) if v == w] for w in words}}"), level=2),
    Family("merge_add", "a, b", ("dict", "dict"), ("merge_add", "combine_totals"),
           ("returns a new dictionary with every key of a or b, where a key found in both maps to the sum of its two "
            "values; the keys of a come first in their order, then the other keys of b in theirs.",),
           ("result = dict(a)\nfor k, v in b.items():\n    result[k] = result.get(k, 0) + v\nreturn result",
            "keys = list(a) + [k for k in b if k not in a]\nreturn {{k: a.get(k, 0) + b.get(k, 0) for k in keys}}"),
           level=2),
    Family("keys_by_value", "d", ("dict",), ("keys_by_value_{nm}", "rank_keys_{nm}"),
           ("returns the keys of d sorted by their values, {desc}; keys with equal values keep their order in d.",),
           ("return sorted(d, key=d.get{rev})",
            "return [k for k, _ in sorted(d.items(), key=lambda item: {neg}item[1])]"),
           ({"nm": "ascending", "desc": "smallest value first", "rev": "", "neg": ""},
            {"nm": "descending", "desc": "largest value first", "rev": ", reverse=True", "neg": "-"}), level=2),
    Family("group_keys_by_value", "d", ("dict",), ("group_keys_by_value", "invert_to_lists"),
           ("returns a dictionary that maps each value of d to the list of the keys that have that value, the keys in "
            "their order in d and the values in order of first appearance.",),
           ("result = {{}}\nfor k, v in d.items():\n    result.setdefault(v, []).append(k)\nreturn result",
            "result = {{}}\nfor k in d:\n    if d[k] not in result:\n        result[d[k]] = []\n    result[d[k]].append(k)\n"
            "return result",
            "return {{v: [k for k in d if d[k] == v] for v in d.values()}}"), level=2),
    Family("election_winner", "votes", ("strs",), ("election_winner", "most_voted_name"),
           ("returns the name that occurs most often in the list of votes; on a tie, the one that comes first in sorted "
            "order; '' if there are no votes.",),
           ("if not votes:\n    return ''\ncounts = {{}}\nfor v in votes:\n    counts[v] = counts.get(v, 0) + 1\n"
            "return min(counts, key=lambda name: (-counts[name], name))",
            "best = ''\nfor name in sorted(set(votes)):\n    if best == '' or votes.count(name) > votes.count(best):\n"
            "        best = name\nreturn best"), level=2),
    Family("sort_by_frequency", "numbers", ("ints",), ("sort_by_frequency", "most_frequent_first"),
           ("returns the numbers of the list sorted so that values that occur more often come first; values that occur "
            "equally often are ordered from smallest to largest.",),
           ("return sorted(numbers, key=lambda x: (-numbers.count(x), x))",
            "from collections import Counter\ncounts = Counter(numbers)\nresult = []\n"
            "for x, c in sorted(counts.items(), key=lambda item: (-item[1], item[0])):\n    result.extend([x] * c)\n"
            "return result"), level=2),
    # ---- level 2: small simulations
    Family("balance_no_overdraft", "transactions", ("ints",),
           ("final_balance_from_{start}", "apply_transactions_from_{start}"),
           ("returns the final balance of an account that starts at {start} and applies the transactions in order: a "
            "positive number is a deposit, a negative number a withdrawal, and a withdrawal that would make the balance "
            "negative is skipped.",),
           ("balance = {start}\nfor t in transactions:\n    if balance + t >= 0:\n        balance += t\nreturn balance",
            "balance = {start}\nfor t in transactions:\n    if t >= 0 or -t <= balance:\n        balance += t\n"
            "return balance"),
           ({"start": 0}, {"start": 10}), level=2),
    Family("bounded_walk", "moves", ("ints",), ("bounded_walk_{k}", "position_kept_within_{k}"),
           ("returns the final value of a counter that starts at 0 and adds each move in turn, where after every move a "
            "value below 0 is set to 0 and a value above {k} is set to {k}.",),
           ("pos = 0\nfor m in moves:\n    pos = min(max(pos + m, 0), {k})\nreturn pos",
            "pos = 0\nfor m in moves:\n    pos += m\n    if pos < 0:\n        pos = 0\n    elif pos > {k}:\n        pos = {k}\n"
            "return pos"), _k(10, 25), level=2),
    Family("toggle_lamps", "presses", ("small_list",), ("lamps_left_on", "toggle_lamps"),
           ("returns, in increasing order, the numbers of the lamps that are on at the end, if all lamps start off and "
            "each number in presses switches that lamp (off to on, on to off).",),
           ("on = set()\nfor p in presses:\n    if p in on:\n        on.remove(p)\n    else:\n        on.add(p)\n"
            "return sorted(on)",
            "return sorted(p for p in set(presses) if presses.count(p) % 2 == 1)",
            "on = set()\nfor p in presses:\n    on ^= {{p}}\nreturn sorted(on)"), level=2),
    Family("stack_commands", "numbers", ("ints",), ("run_stack", "stack_after_commands"),
           ("returns the final stack, bottom first, after reading the numbers in order: a positive number is pushed onto "
            "the stack, a negative number removes the top item (if the stack is not empty), and zero does nothing.",),
           ("stack = []\nfor x in numbers:\n    if x > 0:\n        stack.append(x)\n    elif x < 0 and stack:\n"
            "        stack.pop()\nreturn stack",
            "stack = []\nfor x in numbers:\n    if x > 0:\n        stack = stack + [x]\n    elif x < 0:\n"
            "        stack = stack[:-1]\nreturn stack"), level=2),
    Family("total_waiting", "times", ("small_list",), ("total_waiting{sfx}", "waiting_time{sfx}"),
           ("returns the total time that customers spend waiting when one server serves them {desc}, where times holds "
            "each customer's service time and every customer waits for the service times of all those served before "
            "them.",),
           ("total = 0\nwaited = 0\nfor t in {src}:\n    total += waited\n    waited += t\nreturn total",
            "order = {src}\nreturn sum(sum(order[:i]) for i in range(len(order)))"),
           ({"sfx": "", "src": "times", "desc": "in the given order"},
            {"sfx": "_shortest_first", "src": "sorted(times)", "desc": "in order of service time, shortest first"}),
           level=2),
    Family("distinct_positions", "moves", ("ints",), ("positions_visited", "distinct_stops"),
           ("returns how many different positions a walker stands on, starting at position 0 (which counts) and moving "
            "by each number of moves in turn (positive to the right, negative to the left); positions passed over do "
            "not count.",),
           ("pos = 0\nseen = {{0}}\nfor m in moves:\n    pos += m\n    seen.add(pos)\nreturn len(seen)",
            "import itertools\nreturn len(set(itertools.accumulate(moves, initial=0)))"), level=2),
) + (
    # ---- level 3: sliding windows and prefix sums
    Family("best_window_sum", "numbers", ("ints",), ("best_sum_of_{kw}", "max_window_sum_{k}"),
           ("returns the largest sum of {kw} consecutive numbers of the list, or None if the list has fewer than {kw} "
            "numbers.",),
           ("if len(numbers) < {k}:\n    return None\nwindow = sum(numbers[:{k}])\nbest = window\n"
            "for i in range({k}, len(numbers)):\n    window += numbers[i] - numbers[i - {k}]\n    best = max(best, window)\n"
            "return best",
            "sums = [sum(numbers[i:i + {k}]) for i in range(len(numbers) - {k} + 1)]\nreturn max(sums) if sums else None"),
           _k(2, 3, 4), level=3),
    Family("balance_point", "numbers", ("ints",), ("balance_point", "pivot_position"),
           ("returns the first position i at which the sum of the numbers before position i equals the sum of the "
            "numbers after it, or -1 if there is no such position.",),
           ("total = sum(numbers)\nleft = 0\nfor i, x in enumerate(numbers):\n    if left == total - left - x:\n"
            "        return i\n    left += x\nreturn -1",
            "for i in range(len(numbers)):\n    if sum(numbers[:i]) == sum(numbers[i + 1:]):\n        return i\nreturn -1"),
           level=3),
    Family("count_stretches_with_sum", "numbers, target", ("ints", "int_small"),
           ("stretches_summing_to", "count_runs_with_sum"),
           ("returns how many stretches of one or more neighbouring numbers of the list add up to target.",),
           ("count = 0\nfor i in range(len(numbers)):\n    total = 0\n    for j in range(i, len(numbers)):\n"
            "        total += numbers[j]\n        if total == target:\n            count += 1\nreturn count",
            "seen = {{0: 1}}\ntotal = count = 0\nfor x in numbers:\n    total += x\n    count += seen.get(total - target, 0)\n"
            "    seen[total] = seen.get(total, 0) + 1\nreturn count"), level=3),
    Family("longest_window_within", "numbers, limit", ("runs_list", "small"),
           ("longest_stretch_within", "longest_window_at_most"),
           ("returns the length of the longest stretch of neighbouring numbers of the list of non-negative numbers whose "
            "sum is at most limit (0 if there is none).",),
           ("best = 0\nleft = 0\ntotal = 0\nfor right, x in enumerate(numbers):\n    total += x\n    while total > limit:\n"
            "        total -= numbers[left]\n        left += 1\n    best = max(best, right - left + 1)\nreturn best",
            "best = 0\nfor i in range(len(numbers)):\n    for j in range(i, len(numbers)):\n"
            "        if sum(numbers[i:j + 1]) <= limit:\n            best = max(best, j - i + 1)\nreturn best"), level=3),
    Family("shortest_window_reaching", "numbers, target", ("runs_list", "small"),
           ("shortest_stretch_reaching", "min_window_reaching"),
           ("returns the length of the shortest stretch of one or more neighbouring numbers of the list of non-negative "
            "numbers whose sum is at least target, or 0 if no stretch reaches target.",),
           ("best = 0\nleft = 0\ntotal = 0\nfor right, x in enumerate(numbers):\n    total += x\n"
            "    while left < right and total - numbers[left] >= target:\n        total -= numbers[left]\n        left += 1\n"
            "    if total >= target and (best == 0 or right - left + 1 < best):\n        best = right - left + 1\nreturn best",
            "best = 0\nfor i in range(len(numbers)):\n    total = 0\n    for j in range(i, len(numbers)):\n"
            "        total += numbers[j]\n        if total >= target:\n            if best == 0 or j - i + 1 < best:\n"
            "                best = j - i + 1\n            break\nreturn best"), level=3),
    # ---- level 3: two pointers, binary search, counting
    Family("sorted_squares", "numbers", ("sorted_ints",), ("sorted_squares", "squares_in_order"),
           ("returns the squares of the numbers of the sorted list numbers, in increasing order.",),
           ("result = []\ni, j = 0, len(numbers) - 1\nwhile i <= j:\n    if abs(numbers[i]) > abs(numbers[j]):\n"
            "        result.append(numbers[i] * numbers[i])\n        i += 1\n    else:\n"
            "        result.append(numbers[j] * numbers[j])\n        j -= 1\nreturn result[::-1]",
            "return sorted(x * x for x in numbers)"), level=3),
    Family("sorted_pairs_with_sum", "numbers, target", ("sorted_distinct", "int_small"),
           ("count_pairs_with_sum", "pairs_adding_to"),
           ("returns how many pairs of numbers in the sorted list numbers (no number appears twice) add up to target.",),
           ("count = 0\ni, j = 0, len(numbers) - 1\nwhile i < j:\n    s = numbers[i] + numbers[j]\n    if s == target:\n"
            "        count += 1\n        i += 1\n        j -= 1\n    elif s < target:\n        i += 1\n    else:\n"
            "        j -= 1\nreturn count",
            "present = set(numbers)\nreturn sum(1 for x in numbers if target - x in present and target - x > x)",
            "return sum(1 for i in range(len(numbers)) for j in range(i + 1, len(numbers)) "
            "if numbers[i] + numbers[j] == target)"), level=3),
    Family("three_sum", "numbers, target", ("ints", "int_small"), ("three_add_up_to", "has_triple_with_sum"),
           ("returns True if three numbers at three different positions of the list add up to target.",),
           ("import itertools\nreturn any(a + b + c == target for a, b, c in itertools.combinations(numbers, 3))",
            "values = sorted(numbers)\nfor i in range(len(values) - 2):\n    j, k = i + 1, len(values) - 1\n"
            "    while j < k:\n        s = values[i] + values[j] + values[k]\n        if s == target:\n"
            "            return True\n        if s < target:\n            j += 1\n        else:\n            k -= 1\n"
            "return False"), level=3),
    Family("insert_position", "numbers, x", ("sorted_ints", "int_small"), ("insert_position_{nm}", "insertion_index_{nm}"),
           ("returns the position at which x would be inserted into the sorted list numbers to keep it sorted, {desc}.",),
           ("import bisect\nreturn bisect.bisect_{nm}(numbers, x)",
            "low, high = 0, len(numbers)\nwhile low < high:\n    mid = (low + high) // 2\n    if numbers[mid] {op} x:\n"
            "        low = mid + 1\n    else:\n        high = mid\nreturn low",
            "return sum(1 for v in numbers if v {op} x)"),
           ({"nm": "left", "op": "<", "desc": "placing it before any numbers equal to x"},
            {"nm": "right", "op": "<=", "desc": "placing it after any numbers equal to x"}), level=3),
    Family("integer_sqrt", "n", ("nonneg",), ("integer_sqrt", "floor_square_root"),
           ("returns the largest integer r with r * r <= n, for the non-negative integer n.",),
           ("low, high = 0, n\nwhile low < high:\n    mid = (low + high + 1) // 2\n    if mid * mid <= n:\n        low = mid\n"
            "    else:\n        high = mid - 1\nreturn low",
            "import math\nreturn math.isqrt(n)",
            "r = 0\nwhile (r + 1) * (r + 1) <= n:\n    r += 1\nreturn r"), level=3),
    Family("majority_value", "items", ("runs_list",), ("majority_value", "more_than_half"),
           ("returns the value that occurs in more than half of the positions of the list, or None if no value does.",),
           ("for x in set(items):\n    if items.count(x) * 2 > len(items):\n        return x\nreturn None",
            "candidate, votes = None, 0\nfor x in items:\n    if votes == 0:\n        candidate = x\n"
            "    votes += 1 if x == candidate else -1\n"
            "if candidate is not None and items.count(candidate) * 2 > len(items):\n    return candidate\nreturn None"),
           level=3),
    Family("summarize_ranges", "numbers", ("sorted_distinct",), ("summarize_ranges", "compact_ranges"),
           ("returns a string that describes the sorted list of distinct integers as ranges separated by ', ': each "
            "longest stretch of consecutive integers is written 'a..b', or just 'a' if it has one number "
            "([1, 2, 3, 5, 7, 8] gives '1..3, 5, 7..8'; '' for an empty list).",),
           ("parts = []\ni = 0\nwhile i < len(numbers):\n    j = i\n"
            "    while j + 1 < len(numbers) and numbers[j + 1] == numbers[j] + 1:\n        j += 1\n"
            "    parts.append(str(numbers[i]) if i == j else f'{{numbers[i]}}..{{numbers[j]}}')\n    i = j + 1\n"
            "return ', '.join(parts)",
            "groups = []\nfor x in numbers:\n    if groups and x == groups[-1][1] + 1:\n        groups[-1][1] = x\n"
            "    else:\n        groups.append([x, x])\nreturn ', '.join(str(a) if a == b else f'{{a}}..{{b}}' for a, b in groups)"),
           level=3),
    Family("longest_rising_subsequence", "numbers", ("ints",), ("longest_rising_subsequence", "lis_length"),
           ("returns the length of the longest strictly increasing subsequence of the list (items kept in their order but "
            "not necessarily neighbouring; 0 for an empty list).",),
           ("best = [1] * len(numbers)\nfor i in range(len(numbers)):\n    for j in range(i):\n"
            "        if numbers[j] < numbers[i]:\n            best[i] = max(best[i], best[j] + 1)\nreturn max(best, default=0)",
            "import bisect\ntails = []\nfor x in numbers:\n    i = bisect.bisect_left(tails, x)\n    if i == len(tails):\n"
            "        tails.append(x)\n    else:\n        tails[i] = x\nreturn len(tails)"), level=3),
    # ---- level 3: stacks
    Family("next_greater", "numbers", ("ints",), ("next_greater", "next_larger_to_the_right"),
           ("returns a list giving, for each number, the first number after it in the list that is greater than it, or "
            "None if there is none.",),
           ("result = []\nfor i, x in enumerate(numbers):\n    result.append(next((y for y in numbers[i + 1:] if y > x), None))\n"
            "return result",
            "result = [None] * len(numbers)\nstack = []\nfor i, x in enumerate(numbers):\n"
            "    while stack and numbers[stack[-1]] < x:\n        result[stack.pop()] = x\n    stack.append(i)\nreturn result"),
           level=3),
    Family("span_back", "numbers", ("ints",), ("spans_back", "stretch_not_above"),
           ("returns a list giving, for each number, the length of the longest stretch of neighbouring numbers that ends "
            "at it and contains no number greater than it (counting the number itself).",),
           ("result = []\nfor i, x in enumerate(numbers):\n    j = i\n    while j > 0 and numbers[j - 1] <= x:\n        j -= 1\n"
            "    result.append(i - j + 1)\nreturn result",
            "result = []\nstack = []\nfor i, x in enumerate(numbers):\n    while stack and numbers[stack[-1]] <= x:\n"
            "        stack.pop()\n    result.append(i - stack[-1] if stack else i + 1)\n    stack.append(i)\nreturn result"),
           level=3),
    Family("parens_to_add", "s", ("brackets",), ("parens_to_add", "fix_parens_count"),
           ("returns the smallest number of '(' and ')' characters that must be inserted into s so that its parentheses "
            "are balanced (other characters are ignored).",),
           ("open_count = 0\nadded = 0\nfor c in s:\n    if c == '(':\n        open_count += 1\n    elif c == ')':\n"
            "        if open_count > 0:\n            open_count -= 1\n        else:\n            added += 1\n"
            "return added + open_count",
            "t = ''.join(c for c in s if c in '()')\nwhile '()' in t:\n    t = t.replace('()', '')\nreturn len(t)"), level=3),
    Family("cancel_pairs", "s", ("runs",), ("cancel_pairs", "remove_adjacent_doubles"),
           ("returns what is left of s after repeatedly removing two equal neighbouring characters, until no two "
            "neighbouring characters are equal ('abba' gives '').",),
           ("stack = []\nfor c in s:\n    if stack and stack[-1] == c:\n        stack.pop()\n    else:\n"
            "        stack.append(c)\nreturn ''.join(stack)",
            "changed = True\nwhile changed:\n    changed = False\n    for i in range(len(s) - 1):\n"
            "        if s[i] == s[i + 1]:\n            s = s[:i] + s[i + 2:]\n            changed = True\n            break\n"
            "return s"), level=3),
    Family("longest_distinct_stretch", "s", ("str",), ("longest_distinct_stretch", "longest_without_repeat"),
           ("returns the length of the longest stretch of neighbouring characters of s in which no character appears "
            "twice (0 for the empty string).",),
           ("best = 0\nstart = 0\nlast = {{}}\nfor i, c in enumerate(s):\n    if c in last and last[c] >= start:\n"
            "        start = last[c] + 1\n    last[c] = i\n    best = max(best, i - start + 1)\nreturn best",
            "best = 0\nfor i in range(len(s)):\n    for j in range(i + 1, len(s) + 1):\n"
            "        if len(set(s[i:j])) == j - i:\n            best = max(best, j - i)\nreturn best"), level=3),
    # ---- level 3: greedy choices
    Family("best_trade", "prices", ("ints",), ("best_trade", "max_single_profit"),
           ("returns the largest value of prices[j] - prices[i] with i < j, or 0 if no price is higher than an "
            "earlier one.",),
           ("best = 0\nlowest = None\nfor p in prices:\n    if lowest is not None:\n        best = max(best, p - lowest)\n"
            "    if lowest is None or p < lowest:\n        lowest = p\nreturn best",
            "return max([0] + [prices[j] - prices[i] for i in range(len(prices)) for j in range(i + 1, len(prices))])"),
           level=3),
    Family("can_reach_end", "steps", ("runs_list",), ("can_reach_end", "jumps_reach_last"),
           ("returns True if the last position of the list can be reached from position 0, where from position i you "
            "may jump forward by any number of places from 1 up to steps[i] (a list with at most one item counts as "
            "reachable).",),
           ("reach = 0\nfor i, s in enumerate(steps):\n    if i > reach:\n        return False\n    reach = max(reach, i + s)\n"
            "return True",
            "if len(steps) <= 1:\n    return True\nok = [False] * len(steps)\nok[0] = True\nfor i in range(len(steps)):\n"
            "    if ok[i]:\n        for j in range(i + 1, min(i + steps[i], len(steps) - 1) + 1):\n            ok[j] = True\n"
            "return ok[-1]"), level=3),
    Family("satisfied_children", "greed, sizes", ("small_list", "small_list"), ("content_children", "match_cookies"),
           ("returns the largest number of children that can each get one cookie at least as big as their greed "
            "(greed[i] for child i, sizes[j] for cookie j), if every cookie goes to at most one child.",),
           ("g = sorted(greed)\nc = sorted(sizes)\ni = j = 0\nwhile i < len(g) and j < len(c):\n    if c[j] >= g[i]:\n"
            "        i += 1\n    j += 1\nreturn i",
            "left = sorted(sizes)\ncount = 0\nfor g in sorted(greed):\n    for k, c in enumerate(left):\n        if c >= g:\n"
            "            del left[k]\n            count += 1\n            break\nreturn count"), level=3),
    Family("min_boats", "weights", ("small_list",), ("min_boats_{k}", "boats_needed_{k}"),
           ("returns the smallest number of boats needed to carry everyone, where weights lists each person's weight "
            "(each at most {k}) and a boat carries at most two people with a total weight of at most {k}.",),
           ("w = sorted(weights)\ni, j = 0, len(w) - 1\nboats = 0\nwhile i <= j:\n    if w[i] + w[j] <= {k}:\n        i += 1\n"
            "    j -= 1\n    boats += 1\nreturn boats",
            "from collections import deque\nq = deque(sorted(weights, reverse=True))\nboats = 0\nwhile q:\n"
            "    heavy = q.popleft()\n    if q and heavy + q[-1] <= {k}:\n        q.pop()\n    boats += 1\nreturn boats"),
           _k(5, 6, 8), level=3),
    Family("fewest_candies", "ratings", ("ints1",), ("fewest_candies", "candy_total"),
           ("returns the smallest total number of candies for children standing in a row with the given ratings, if "
            "every child gets at least one candy and a child with a higher rating than a neighbour gets more candies "
            "than that neighbour.",),
           ("n = len(ratings)\ncandies = [1] * n\nfor i in range(1, n):\n    if ratings[i] > ratings[i - 1]:\n"
            "        candies[i] = candies[i - 1] + 1\nfor i in range(n - 2, -1, -1):\n    if ratings[i] > ratings[i + 1]:\n"
            "        candies[i] = max(candies[i], candies[i + 1] + 1)\nreturn sum(candies)",
            "candies = [1] * len(ratings)\nchanged = True\nwhile changed:\n    changed = False\n"
            "    for i in range(len(ratings)):\n        for j in (i - 1, i + 1):\n"
            "            if 0 <= j < len(ratings) and ratings[i] > ratings[j] and candies[i] <= candies[j]:\n"
            "                candies[i] = candies[j] + 1\n                changed = True\nreturn sum(candies)"), level=3),
    Family("trapped_water", "heights", ("runs_list",), ("trapped_water", "rain_between_bars"),
           ("returns how many units of water stay between bars of the given non-negative heights (each bar 1 wide) after "
            "rain: above each bar the water rises to the smaller of the tallest bar on its left and the tallest bar on "
            "its right, both counting the bar itself.",),
           ("n = len(heights)\nif n == 0:\n    return 0\nleft = [0] * n\nright = [0] * n\nleft[0] = heights[0]\n"
            "for i in range(1, n):\n    left[i] = max(left[i - 1], heights[i])\nright[-1] = heights[-1]\n"
            "for i in range(n - 2, -1, -1):\n    right[i] = max(right[i + 1], heights[i])\n"
            "return sum(min(left[i], right[i]) - heights[i] for i in range(n))",
            "i, j = 0, len(heights) - 1\nleft_max = right_max = 0\nwater = 0\nwhile i < j:\n    if heights[i] < heights[j]:\n"
            "        left_max = max(left_max, heights[i])\n        water += left_max - heights[i]\n        i += 1\n"
            "    else:\n        right_max = max(right_max, heights[j])\n        water += right_max - heights[j]\n"
            "        j -= 1\nreturn water",
            "return sum(min(max(heights[:i + 1]), max(heights[i:])) - h for i, h in enumerate(heights))"), level=3),
    Family("wrap_words", "text", ("text",), ("wrap_words_{k}", "fill_lines_{k}"),
           ("returns the words of text arranged into lines of at most {k} characters, as a list of lines: words in a "
            "line are separated by single spaces, each line takes as many of the remaining words as fit, and a word "
            "longer than {k} characters gets a line of its own ([] if text has no words).",),
           ("lines = []\nline = ''\nfor w in text.split():\n    if line and len(line) + 1 + len(w) <= {k}:\n"
            "        line += ' ' + w\n    else:\n        if line:\n            lines.append(line)\n        line = w\n"
            "if line:\n    lines.append(line)\nreturn lines",
            "lines = []\nfor w in text.split():\n    if lines and len(lines[-1]) + 1 + len(w) <= {k}:\n"
            "        lines[-1] += ' ' + w\n    else:\n        lines.append(w)\nreturn lines",
            "import textwrap\nreturn textwrap.wrap(text, width={k}, break_long_words=False)"), _k(8, 12), level=3),
    # ---- level 3: dynamic programming and exhaustive search
    Family("ways_to_pay", "amount", ("small",), ("ways_to_pay_{nm}", "coin_combinations_{nm}"),
           ("returns in how many different ways the non-negative amount can be paid with coins of {desc} (as many coins "
            "of each kind as needed; the order of the coins does not matter; 1 way for amount 0).",),
           ("ways = [1] + [0] * amount\nfor coin in {coins}:\n    for x in range(coin, amount + 1):\n"
            "        ways[x] += ways[x - coin]\nreturn ways[amount]",
            "coins = {coins}\ndef count(rest, i):\n    if rest == 0:\n        return 1\n    if i == len(coins):\n"
            "        return 0\n    total = 0\n    for used in range(rest // coins[i] + 1):\n"
            "        total += count(rest - used * coins[i], i + 1)\n    return total\nreturn count(amount, 0)"),
           ({"coins": "(1, 2, 5)", "desc": "1, 2 and 5", "nm": "with_1_2_5"},
            {"coins": "(2, 3, 7)", "desc": "2, 3 and 7", "nm": "with_2_3_7"}), level=3),
    Family("fewest_coins", "amount", ("small",), ("fewest_coins_{nm}", "min_coins_{nm}"),
           ("returns the smallest number of coins of {desc} (as many of each kind as needed) that add up exactly to the "
            "non-negative amount, or -1 if that is impossible (0 for amount 0).",),
           ("big = amount + 1\nbest = [0] + [big] * amount\nfor x in range(1, amount + 1):\n    for coin in {coins}:\n"
            "        if coin <= x and best[x - coin] + 1 < best[x]:\n            best[x] = best[x - coin] + 1\n"
            "return best[amount] if best[amount] < big else -1",
            "from collections import deque\nseen = {{0}}\nqueue = deque([(0, 0)])\nwhile queue:\n"
            "    total, count = queue.popleft()\n    if total == amount:\n        return count\n    for coin in {coins}:\n"
            "        if total + coin <= amount and total + coin not in seen:\n            seen.add(total + coin)\n"
            "            queue.append((total + coin, count + 1))\nreturn -1"),
           ({"coins": "(1, 3, 4)", "desc": "1, 3 and 4", "nm": "with_1_3_4"},
            {"coins": "(3, 5)", "desc": "3 and 5", "nm": "with_3_5"}), level=3),
    Family("edit_distance", "a, b", ("word0", "word0"), ("edit_distance", "changes_needed"),
           ("returns the smallest number of single-character insertions, deletions and replacements that turn a into b.",),
           ("rows, cols = len(a) + 1, len(b) + 1\nd = [[0] * cols for _ in range(rows)]\nfor i in range(rows):\n"
            "    d[i][0] = i\nfor j in range(cols):\n    d[0][j] = j\nfor i in range(1, rows):\n    for j in range(1, cols):\n"
            "        cost = 0 if a[i - 1] == b[j - 1] else 1\n"
            "        d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)\nreturn d[-1][-1]",
            "prev = list(range(len(b) + 1))\nfor i, x in enumerate(a, 1):\n    cur = [i]\n    for j, y in enumerate(b, 1):\n"
            "        cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))\n    prev = cur\nreturn prev[-1]",
            "import functools\n@functools.lru_cache(maxsize=None)\ndef dist(i, j):\n    if i == len(a):\n"
            "        return len(b) - j\n    if j == len(b):\n        return len(a) - i\n    if a[i] == b[j]:\n"
            "        return dist(i + 1, j + 1)\n    return 1 + min(dist(i + 1, j), dist(i, j + 1), dist(i + 1, j + 1))\n"
            "return dist(0, 0)"), level=3),
    Family("common_subsequence_length", "a, b", ("word0", "word0"), ("common_subsequence_length", "lcs_length"),
           ("returns the length of the longest sequence of characters that appears in both a and b in the same order, "
            "not necessarily next to each other.",),
           ("table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]\nfor i in range(len(a) - 1, -1, -1):\n"
            "    for j in range(len(b) - 1, -1, -1):\n        if a[i] == b[j]:\n            table[i][j] = table[i + 1][j + 1] + 1\n"
            "        else:\n            table[i][j] = max(table[i + 1][j], table[i][j + 1])\nreturn table[0][0]",
            "prev = [0] * (len(b) + 1)\nfor x in a:\n    cur = [0]\n    for j, y in enumerate(b):\n"
            "        cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))\n    prev = cur\nreturn prev[-1]"),
           level=3),
    Family("subset_sum", "numbers, target", ("ints", "int_small"), ("subset_adds_to", "can_pick_sum"),
           ("returns True if some of the numbers, each position used at most once, add up to target (choosing none of "
            "them gives 0).",),
           ("sums = {{0}}\nfor x in numbers:\n    sums |= {{s + x for s in sums}}\nreturn target in sums",
            "import itertools\nreturn any(sum(c) == target for r in range(len(numbers) + 1) "
            "for c in itertools.combinations(numbers, r))",
            "def can(i, rest):\n    if i == len(numbers):\n        return rest == 0\n"
            "    return can(i + 1, rest - numbers[i]) or can(i + 1, rest)\nreturn can(0, target)"), level=3),
    Family("fairest_split", "numbers", ("ints",), ("fairest_split", "min_split_difference"),
           ("returns the smallest possible absolute difference between the sums of two groups when every number of the "
            "list is put into exactly one of the two groups (either group may be empty).",),
           ("sums = {{0}}\nfor x in numbers:\n    sums = sums | {{s + x for s in sums}}\ntotal = sum(numbers)\n"
            "return min(abs(total - 2 * s) for s in sums)",
            "import itertools\ntotal = sum(numbers)\nbest = abs(total)\nfor r in range(1, len(numbers) + 1):\n"
            "    for group in itertools.combinations(numbers, r):\n        best = min(best, abs(total - 2 * sum(group)))\n"
            "return best"), level=3),
    Family("next_bigger_same_digits", "n", ("nonneg",), ("next_bigger_same_digits", "next_digit_permutation"),
           ("returns the smallest integer greater than the non-negative integer n that is written with exactly the same "
            "digits (each as many times as in n), or -1 if there is none.",),
           ("d = list(str(n))\ni = len(d) - 2\nwhile i >= 0 and d[i] >= d[i + 1]:\n    i -= 1\nif i < 0:\n    return -1\n"
            "j = len(d) - 1\nwhile d[j] <= d[i]:\n    j -= 1\nd[i], d[j] = d[j], d[i]\nd[i + 1:] = reversed(d[i + 1:])\n"
            "return int(''.join(d))",
            "import itertools\nbigger = [int(''.join(p)) for p in itertools.permutations(str(n))]\n"
            "bigger = [x for x in bigger if x > n]\nreturn min(bigger) if bigger else -1"), level=3),
    Family("last_standing", "n", ("pos",), ("last_standing_{kw}", "survivor_every_{kw}"),
           ("returns the number of the last person left when n people numbered 1 to n stand in a circle and, counting "
            "from person 1, every {ord} person leaves the circle, the counting going on from the person after them.",),
           ("people = list(range(1, n + 1))\ni = 0\nwhile len(people) > 1:\n    i = (i + {k} - 1) % len(people)\n"
            "    people.pop(i)\nreturn people[0]",
            "survivor = 0\nfor size in range(2, n + 1):\n    survivor = (survivor + {k}) % size\nreturn survivor + 1"),
           tuple({"k": k, "kw": NUM[k], "ord": ORD[k]} for k in (2, 3)), level=3),
    # ---- level 3: grids
    Family("spiral_order", "rows", ("matrix",), ("spiral_order", "read_in_spiral"),
           ("returns the numbers of the rectangular grid rows in clockwise spiral order, starting at the top-left corner "
            "and going right first (an empty list for an empty grid).",),
           ("grid = [list(row) for row in rows]\nresult = []\nwhile grid:\n    result.extend(grid.pop(0))\n"
            "    grid = [list(r) for r in zip(*grid)][::-1]\nreturn result",
            "result = []\nif not rows:\n    return result\ntop, bottom, left, right = 0, len(rows) - 1, 0, len(rows[0]) - 1\n"
            "while top <= bottom and left <= right:\n    for j in range(left, right + 1):\n        result.append(rows[top][j])\n"
            "    for i in range(top + 1, bottom + 1):\n        result.append(rows[i][right])\n"
            "    if top < bottom and left < right:\n        for j in range(right - 1, left - 1, -1):\n"
            "            result.append(rows[bottom][j])\n        for i in range(bottom - 1, top, -1):\n"
            "            result.append(rows[i][left])\n    top, bottom, left, right = top + 1, bottom - 1, left + 1, right - 1\n"
            "return result",
            "if not rows:\n    return []\nn, m = len(rows), len(rows[0])\nseen = [[False] * m for _ in range(n)]\n"
            "moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]\nresult = []\ni = j = d = 0\nfor _ in range(n * m):\n"
            "    result.append(rows[i][j])\n    seen[i][j] = True\n    di, dj = moves[d]\n"
            "    if not (0 <= i + di < n and 0 <= j + dj < m) or seen[i + di][j + dj]:\n        d = (d + 1) % 4\n"
            "        di, dj = moves[d]\n    i, j = i + di, j + dj\nreturn result"), level=3),
    Family("count_positive_regions", "grid", ("matrix",), ("count_positive_regions", "positive_islands"),
           ("returns how many groups of positive cells the grid has, where positive cells that touch horizontally or "
            "vertically belong to the same group (0 for an empty grid).",),
           ("seen = set()\ncount = 0\nfor r in range(len(grid)):\n    for c in range(len(grid[r])):\n"
            "        if grid[r][c] > 0 and (r, c) not in seen:\n            count += 1\n            seen.add((r, c))\n"
            "            stack = [(r, c)]\n            while stack:\n                i, j = stack.pop()\n"
            "                for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):\n"
            "                    if 0 <= a < len(grid) and 0 <= b < len(grid[a]) and grid[a][b] > 0 and (a, b) not in seen:\n"
            "                        seen.add((a, b))\n                        stack.append((a, b))\nreturn count",
            "cells = [list(row) for row in grid]\ndef sink(i, j):\n"
            "    if 0 <= i < len(cells) and 0 <= j < len(cells[i]) and cells[i][j] > 0:\n        cells[i][j] = 0\n"
            "        sink(i + 1, j)\n        sink(i - 1, j)\n        sink(i, j + 1)\n        sink(i, j - 1)\ncount = 0\n"
            "for i in range(len(cells)):\n    for j in range(len(cells[i])):\n        if cells[i][j] > 0:\n"
            "            count += 1\n            sink(i, j)\nreturn count"), level=3),
    Family("grid_peaks", "grid", ("matrix1",), ("grid_peaks", "count_grid_maxima"),
           ("returns how many cells of the non-empty grid are strictly greater than every cell that touches them "
            "horizontally or vertically (a cell with no such neighbour counts).",),
           ("count = 0\nfor i in range(len(grid)):\n    for j in range(len(grid[i])):\n"
            "        around = [grid[a][b] for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))\n"
            "                  if 0 <= a < len(grid) and 0 <= b < len(grid[a])]\n"
            "        if all(grid[i][j] > v for v in around):\n            count += 1\nreturn count",
            "n, m = len(grid), len(grid[0])\nlow = min(min(row) for row in grid) - 1\n"
            "pad = [[low] * (m + 2)] + [[low] + row + [low] for row in grid] + [[low] * (m + 2)]\n"
            "return sum(1 for i in range(1, n + 1) for j in range(1, m + 1)\n"
            "           if pad[i][j] > max(pad[i - 1][j], pad[i + 1][j], pad[i][j - 1], pad[i][j + 1]))"), level=3),
    Family("neighbour_counts", "grid", ("matrix1",), ("positive_neighbour_counts", "count_live_neighbours"),
           ("returns a grid of the same shape in which each cell holds how many of the up to eight cells around it in "
            "grid (diagonal ones included) hold a positive number.",),
           ("n, m = len(grid), len(grid[0])\nresult = []\nfor i in range(n):\n    row = []\n    for j in range(m):\n"
            "        count = 0\n        for di in (-1, 0, 1):\n            for dj in (-1, 0, 1):\n"
            "                if (di or dj) and 0 <= i + di < n and 0 <= j + dj < m and grid[i + di][j + dj] > 0:\n"
            "                    count += 1\n        row.append(count)\n    result.append(row)\nreturn result",
            "n, m = len(grid), len(grid[0])\n"
            "return [[sum(1 for a in range(max(i - 1, 0), min(i + 2, n)) for b in range(max(j - 1, 0), min(j + 2, m))\n"
            "             if (a, b) != (i, j) and grid[a][b] > 0) for j in range(m)] for i in range(n)]"), level=3),
    Family("count_grid_paths", "grid", ("matrix1",), ("count_open_paths", "paths_avoiding_negatives"),
           ("returns how many paths lead from the top-left to the bottom-right cell of the grid moving only right or "
            "down and never stepping on a cell with a negative number (0 if either corner is negative).",),
           ("n, m = len(grid), len(grid[0])\nways = [[0] * m for _ in range(n)]\nfor i in range(n):\n    for j in range(m):\n"
            "        if grid[i][j] < 0:\n            continue\n        if i == 0 and j == 0:\n            ways[i][j] = 1\n"
            "        else:\n            ways[i][j] = (ways[i - 1][j] if i > 0 else 0) + (ways[i][j - 1] if j > 0 else 0)\n"
            "return ways[-1][-1]",
            "n, m = len(grid), len(grid[0])\ndef walk(i, j):\n    if i >= n or j >= m or grid[i][j] < 0:\n        return 0\n"
            "    if i == n - 1 and j == m - 1:\n        return 1\n    return walk(i + 1, j) + walk(i, j + 1)\nreturn walk(0, 0)"),
           level=3),
    Family("cheapest_path", "grid", ("matrix1",), ("cheapest_path", "min_path_sum"),
           ("returns the smallest possible sum of the numbers on a path from the top-left to the bottom-right cell of the "
            "grid that moves only right or down (both corners included).",),
           ("n, m = len(grid), len(grid[0])\ncost = [[0] * m for _ in range(n)]\nfor i in range(n):\n    for j in range(m):\n"
            "        if i == 0 and j == 0:\n            cost[i][j] = grid[0][0]\n        elif i == 0:\n"
            "            cost[i][j] = cost[i][j - 1] + grid[i][j]\n        elif j == 0:\n"
            "            cost[i][j] = cost[i - 1][j] + grid[i][j]\n        else:\n"
            "            cost[i][j] = min(cost[i - 1][j], cost[i][j - 1]) + grid[i][j]\nreturn cost[-1][-1]",
            "n, m = len(grid), len(grid[0])\ndef best(i, j):\n    if i == n - 1 and j == m - 1:\n        return grid[i][j]\n"
            "    options = []\n    if i + 1 < n:\n        options.append(best(i + 1, j))\n    if j + 1 < m:\n"
            "        options.append(best(i, j + 1))\n    return grid[i][j] + min(options)\nreturn best(0, 0)",
            "row = []\nfor j, x in enumerate(grid[0]):\n    row.append(x + (row[j - 1] if j else 0))\nfor r in grid[1:]:\n"
            "    row[0] += r[0]\n    for j in range(1, len(r)):\n        row[j] = min(row[j], row[j - 1]) + r[j]\nreturn row[-1]"),
           level=3),
    Family("grid_steps", "grid", ("matrix1",), ("fewest_steps", "shortest_route"),
           ("returns the fewest moves (up, down, left or right, one cell per move) needed to go from the top-left to the "
            "bottom-right cell of the grid without ever standing on a cell with a negative number, or -1 if that is "
            "impossible.",),
           ("from collections import deque\nn, m = len(grid), len(grid[0])\nif grid[0][0] < 0 or grid[n - 1][m - 1] < 0:\n"
            "    return -1\ndist = {{(0, 0): 0}}\nqueue = deque([(0, 0)])\nwhile queue:\n    i, j = queue.popleft()\n"
            "    if (i, j) == (n - 1, m - 1):\n        return dist[(i, j)]\n"
            "    for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):\n"
            "        if 0 <= a < n and 0 <= b < m and grid[a][b] >= 0 and (a, b) not in dist:\n"
            "            dist[(a, b)] = dist[(i, j)] + 1\n            queue.append((a, b))\nreturn -1",
            "n, m = len(grid), len(grid[0])\nif grid[0][0] < 0:\n    return -1\nfrontier = {{(0, 0)}}\nseen = {{(0, 0)}}\n"
            "steps = 0\nwhile frontier:\n    if (n - 1, m - 1) in frontier:\n        return steps\n    nxt = set()\n"
            "    for i, j in frontier:\n        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):\n"
            "            if 0 <= a < n and 0 <= b < m and grid[a][b] >= 0 and (a, b) not in seen:\n"
            "                seen.add((a, b))\n                nxt.add((a, b))\n    frontier = nxt\n    steps += 1\nreturn -1"),
           level=3),
)
