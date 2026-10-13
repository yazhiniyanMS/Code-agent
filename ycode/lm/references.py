"""Reference solutions for the benchmark (evaluation only: never imported by the training
data generators). Used by the error analysis to show the expected answer next to a failure."""

# The 30 standard problems (ycode.lm.evaluate.PROBLEMS); a bug fix's expected answer is the same function.
REFERENCE = {
    "add": "def add(a, b):\n    return a + b",
    "subtract": "def subtract(a, b):\n    return a - b",
    "is_even": "def is_even(n):\n    return n % 2 == 0",
    "square": "def square(x):\n    return x * x",
    "maximum": "def maximum(items):\n    return max(items)",
    "minimum": "def minimum(items):\n    return min(items)",
    "total": "def total(numbers):\n    return sum(numbers)",
    "average": "def average(numbers):\n    return sum(numbers) / len(numbers)",
    "reverse_string": "def reverse_string(s):\n    return s[::-1]",
    "is_palindrome": "def is_palindrome(s):\n    return s == s[::-1]",
    "factorial": "def factorial(n):\n    return 1 if n < 2 else n * factorial(n - 1)",
    "fibonacci": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "count_vowels": "def count_vowels(s):\n    return sum(c in 'aeiou' for c in s)",
    "is_prime": "def is_prime(n):\n    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))",
    "gcd": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
    "celsius_to_fahrenheit": "def celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 32",
    "filter_even": "def filter_even(numbers):\n    return [n for n in numbers if n % 2 == 0]",
    "unique": "def unique(items):\n    return list(dict.fromkeys(items))",
    "count_words": "def count_words(text):\n    return len(text.split())",
    "flatten": "def flatten(lists):\n    return [x for l in lists for x in l]",
    "clamp": "def clamp(x, low, high):\n    return max(low, min(x, high))",
    "char_count": "def char_count(s, ch):\n    return s.count(ch)",
    "merge_dicts": "def merge_dicts(a, b):\n    return {**a, **b}",
    "first_word": "def first_word(text):\n    return text.split()[0]",
    "to_upper": "def to_upper(s):\n    return s.upper()",
    "is_sorted": "def is_sorted(items):\n    return items == sorted(items)",
    "second_largest": "def second_largest(numbers):\n    return sorted(set(numbers))[-2]",
    "binary_search": ("def binary_search(items, target):\n    lo, hi = 0, len(items) - 1\n    while lo <= hi:\n"
                      "        mid = (lo + hi) // 2\n        if items[mid] == target:\n            return mid\n"
                      "        if items[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n"
                      "    return -1"),
    "is_anagram": "def is_anagram(a, b):\n    return sorted(a) == sorted(b)",
    "digit_sum": "def digit_sum(n):\n    return sum(int(d) for d in str(n))",
}

# The 20 fresh problems (ycode.lm.basics.FRESH_PROBLEMS).
FRESH_REFERENCE = {
    "triple": "def triple(x):\n    return 3 * x",
    "is_zero": "def is_zero(x):\n    return x == 0",
    "area_of_square": "def area_of_square(side):\n    return side * side",
    "count_spaces": "def count_spaces(s):\n    return s.count(' ')",
    "ends_with_dot": "def ends_with_dot(s):\n    return s.endswith('.')",
    "second_item": "def second_item(items):\n    return items[1]",
    "sum_of_evens": "def sum_of_evens(numbers):\n    return sum(x for x in numbers if x % 2 == 0)",
    "count_zeros": "def count_zeros(numbers):\n    return numbers.count(0)",
    "halve_all": "def halve_all(numbers):\n    return [x / 2 for x in numbers]",
    "max_length": "def max_length(words):\n    return max(len(w) for w in words)",
    "seconds_to_minutes": "def seconds_to_minutes(seconds):\n    return seconds / 60",
    "is_empty": "def is_empty(items):\n    return len(items) == 0",
    "add_one": "def add_one(numbers):\n    return [x + 1 for x in numbers]",
    "starts_with_vowel": "def starts_with_vowel(word):\n    return word[0].lower() in 'aeiou'",
    "middle_item": "def middle_item(items):\n    return items[len(items) // 2]",
    "count_letter_a": "def count_letter_a(s):\n    return s.count('a')",
    "sum_of_cubes": "def sum_of_cubes(numbers):\n    return sum(x ** 3 for x in numbers)",
    "all_strings_short": "def all_strings_short(words, limit):\n    return all(len(w) < limit for w in words)",
    "dict_size": "def dict_size(d):\n    return len(d)",
    "join_words": "def join_words(words):\n    return ' '.join(words)",
}
