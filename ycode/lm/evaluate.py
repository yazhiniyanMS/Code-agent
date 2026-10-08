"""Measure how good a YCode-LM model is.

Two complementary metrics:

* **Bits per byte (BPB)** on held-out code: how well the model predicts real
  code it never saw. Measured per byte rather than per token, so models with
  different tokenizers (e.g. v1 vs v2) are directly comparable. Lower is better.
* **Functional correctness (pass@k)**: the model writes solutions to small
  programming problems and the code is actually executed against unit tests.
  Higher is better.

Generated code runs in a separate Python process with a timeout, in a
temporary directory, in isolated mode. It is still untrusted code: only
evaluate models you trained yourself.
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

Log = Callable[[str], None]


@dataclass(frozen=True)
class Problem:
    name: str
    prompt: str
    tests: str


def _p(name: str, sig: str, task: str, tests: str) -> Problem:
    return Problem(name, f"Write a Python function `{sig}` that {task}", tests.strip())


# Small, unambiguous problems in the same phrasing the model was tuned on.
PROBLEMS: tuple[Problem, ...] = (
    _p("add", "add(a, b)", "returns the sum of a and b.", "assert add(2, 3) == 5\nassert add(-1, 1) == 0"),
    _p("subtract", "subtract(a, b)", "returns a minus b.", "assert subtract(5, 3) == 2\nassert subtract(0, 4) == -4"),
    _p("is_even", "is_even(n)", "returns True if n is even, otherwise False.",
       "assert is_even(4) is True\nassert is_even(7) is False"),
    _p("square", "square(x)", "returns x squared.", "assert square(3) == 9\nassert square(-2) == 4"),
    _p("maximum", "maximum(items)", "returns the largest item in a list.",
       "assert maximum([3, 9, 2]) == 9\nassert maximum([-5, -1]) == -1"),
    _p("minimum", "minimum(items)", "returns the smallest item in a list.",
       "assert minimum([3, 9, 2]) == 2\nassert minimum([-5, -1]) == -5"),
    _p("total", "total(numbers)", "returns the sum of a list of numbers.",
       "assert total([1, 2, 3]) == 6\nassert total([]) == 0"),
    _p("average", "average(numbers)", "returns the average of a non-empty list of numbers.",
       "assert average([1, 2, 3]) == 2\nassert average([4]) == 4"),
    _p("reverse_string", "reverse_string(s)", "returns the string s reversed.",
       "assert reverse_string('abc') == 'cba'\nassert reverse_string('') == ''"),
    _p("is_palindrome", "is_palindrome(s)", "returns True if the string s reads the same forwards and backwards.",
       "assert is_palindrome('racecar')\nassert not is_palindrome('abc')"),
    _p("factorial", "factorial(n)", "returns the factorial of n.",
       "assert factorial(0) == 1\nassert factorial(5) == 120"),
    _p("fibonacci", "fibonacci(n)", "returns the n-th Fibonacci number, with fibonacci(0) == 0 and fibonacci(1) == 1.",
       "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55"),
    _p("count_vowels", "count_vowels(s)", "returns the number of vowels (a, e, i, o, u) in the string s.",
       "assert count_vowels('hello') == 2\nassert count_vowels('xyz') == 0"),
    _p("is_prime", "is_prime(n)", "returns True if n is a prime number.",
       "assert is_prime(7)\nassert not is_prime(8)\nassert not is_prime(1)\nassert is_prime(2)"),
    _p("gcd", "gcd(a, b)", "returns the greatest common divisor of a and b.",
       "assert gcd(12, 18) == 6\nassert gcd(7, 5) == 1"),
    _p("celsius_to_fahrenheit", "celsius_to_fahrenheit(c)", "converts a temperature from Celsius to Fahrenheit.",
       "assert celsius_to_fahrenheit(0) == 32\nassert celsius_to_fahrenheit(100) == 212"),
    _p("filter_even", "filter_even(numbers)", "returns a list of the even numbers in numbers.",
       "assert filter_even([1, 2, 3, 4]) == [2, 4]\nassert filter_even([]) == []"),
    _p("unique", "unique(items)", "returns the items with duplicates removed, keeping the original order.",
       "assert unique([1, 2, 1, 3, 2]) == [1, 2, 3]"),
    _p("count_words", "count_words(text)", "returns the number of words in text, split on whitespace.",
       "assert count_words('a bc  d') == 3\nassert count_words('') == 0"),
    _p("flatten", "flatten(lists)", "flattens a list of lists into a single list.",
       "assert flatten([[1, 2], [3], []]) == [1, 2, 3]"),
    _p("clamp", "clamp(x, low, high)", "returns x limited to the range from low to high.",
       "assert clamp(5, 0, 3) == 3\nassert clamp(-1, 0, 3) == 0\nassert clamp(2, 0, 3) == 2"),
    _p("char_count", "char_count(s, ch)", "returns how many times the character ch occurs in s.",
       "assert char_count('banana', 'a') == 3\nassert char_count('abc', 'z') == 0"),
    _p("merge_dicts", "merge_dicts(a, b)", "returns a new dict with the keys of a and b, b winning on conflicts.",
       "assert merge_dicts({'x': 1}, {'x': 2, 'y': 3}) == {'x': 2, 'y': 3}\n"
       "assert merge_dicts({'a': 1}, {'b': 2}) == {'a': 1, 'b': 2}"),
    _p("first_word", "first_word(text)", "returns the first word of text.",
       "assert first_word('hello world') == 'hello'"),
    _p("to_upper", "to_upper(s)", "returns s converted to upper case.", "assert to_upper('abC') == 'ABC'"),
    _p("is_sorted", "is_sorted(items)", "returns True if the list is sorted in ascending order.",
       "assert is_sorted([1, 2, 2, 5])\nassert not is_sorted([3, 1])"),
    _p("second_largest", "second_largest(numbers)", "returns the second largest distinct number in the list.",
       "assert second_largest([4, 1, 9, 9, 7]) == 7"),
    _p("binary_search", "binary_search(items, target)",
       "returns the index of target in the sorted list items, or -1 if it is not present.",
       "assert binary_search([1, 3, 5, 7], 5) == 2\nassert binary_search([1, 3, 5, 7], 4) == -1"),
    _p("is_anagram", "is_anagram(a, b)", "returns True if the strings a and b are anagrams of each other.",
       "assert is_anagram('listen', 'silent')\nassert not is_anagram('ab', 'abc')"),
    _p("digit_sum", "digit_sum(n)", "returns the sum of the decimal digits of the non-negative integer n.",
       "assert digit_sum(1234) == 10\nassert digit_sum(0) == 0"),
)

def _bug(name: str, doc: str, buggy_body: str) -> tuple[str, str]:
    return name, f'def {name}:\n    """{doc}"""\n{buggy_body}'


# One realistic bug each (operator swap, off-by-one, flipped comparison, wrong constant).
# The fixed code must pass the matching problem's tests.
BUGGY: tuple[tuple[str, str], ...] = (
    _bug("add(a, b)", "Return the sum of a and b.", "    return a - b"),
    _bug("subtract(a, b)", "Return a minus b.", "    return a + b"),
    _bug("is_even(n)", "Return True if n is even, otherwise False.", "    return n % 2 == 1"),
    _bug("square(x)", "Return x squared.", "    return x * 2"),
    _bug("total(numbers)", "Return the sum of a list of numbers.",
         "    result = 1\n    for x in numbers:\n        result += x\n    return result"),
    _bug("factorial(n)", "Return the factorial of n.",
         "    result = 1\n    for i in range(1, n):\n        result *= i\n    return result"),
    _bug("maximum(items)", "Return the largest item in a list.",
         "    best = items[0]\n    for x in items:\n        if x < best:\n            best = x\n    return best"),
    _bug("minimum(items)", "Return the smallest item in a list.",
         "    best = items[0]\n    for x in items:\n        if x > best:\n            best = x\n    return best"),
    _bug("count_vowels(s)", "Return the number of vowels in the string s.",
         "    count = 0\n    for c in s:\n        if c not in 'aeiou':\n            count += 1\n    return count"),
    _bug("filter_even(numbers)", "Return a list of the even numbers in numbers.",
         "    return [n for n in numbers if n % 2 == 1]"),
    _bug("is_sorted(items)", "Return True if the list is sorted in ascending order.",
         "    for i in range(len(items) - 1):\n        if items[i] < items[i + 1]:\n            return False\n"
         "    return True"),
    _bug("celsius_to_fahrenheit(c)", "Convert a temperature from Celsius to Fahrenheit.",
         "    return c * 9 / 5 - 32"),
    _bug("char_count(s, ch)", "Return how many times the character ch occurs in s.",
         "    count = 0\n    for c in s:\n        if c != ch:\n            count += 1\n    return count"),
    _bug("clamp(x, low, high)", "Return x limited to the range from low to high.",
         "    if x < low:\n        return high\n    if x > high:\n        return high\n    return x"),
    _bug("is_palindrome(s)", "Return True if the string s reads the same forwards and backwards.",
         "    return s != s[::-1]"),
)

_ALL_TESTS = {p.name: p.tests for p in PROBLEMS}


def fresh_problems() -> tuple[Problem, ...]:
    """20 more problems, none of them a training concept in ycode.lm.basics."""
    from ycode.lm.basics import FRESH_PROBLEMS

    return tuple(_p(name, sig, task, tests) for name, sig, task, tests in FRESH_PROBLEMS)

BUGFIX_PROMPT = "This function has a bug. Find and fix it:\n```python\n{code}\n```"


def bugfix_eval(lm, *, max_new_tokens: int = 200, log: Log | None = None) -> tuple[float, list[str]]:
    """fix@1: share of buggy functions whose greedy "fixed" version passes the tests."""
    tests = _ALL_TESTS
    fixed: list[str] = []
    for signature, code in BUGGY:
        name = signature.split("(")[0]
        answer = lm.chat([("user", BUGFIX_PROMPT.format(code=code))], max_new_tokens=max_new_tokens, temperature=0)
        ok = run_tests(extract_code(answer), tests[name])
        if ok:
            fixed.append(name)
        if log:
            log(f"  {'FIXED' if ok else 'fail '}  {name}")
    return len(fixed) / len(BUGGY), fixed


_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:```|$)", re.DOTALL)


def extract_code(answer: str) -> str:
    """The first fenced code block, or the answer from its first `def` onwards."""
    match = _CODE_BLOCK.search(answer)
    if match:
        return match.group(1).strip()
    idx = answer.find("def ")
    return answer[idx:].strip() if idx >= 0 else answer.strip()


def run_tests(code: str, tests: str, timeout: float = 5.0) -> bool:
    program = f"{code}\n\n{tests}\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, "-I", str(path)], cwd=tmp, capture_output=True,
                                  timeout=timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return False
    return proc.returncode == 0


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


@dataclass
class FunctionalResult:
    pass_at_1: float
    pass_at_k: float | None
    k: int
    solved: list[str] = field(default_factory=list)
    samples: dict[str, list[str]] = field(default_factory=dict)


def functional_eval(lm, *, samples: int = 1, temperature: float = 0.8, max_new_tokens: int = 200,
                    problems: tuple[Problem, ...] = PROBLEMS, seed: int = 0, log: Log | None = None
                    ) -> FunctionalResult:
    """pass@1 from a greedy answer; pass@k from ``samples`` sampled answers when samples > 1."""
    torch.manual_seed(seed)
    greedy_passes = 0
    per_problem: list[tuple[int, int]] = []
    solved: list[str] = []
    record: dict[str, list[str]] = {}
    for prob in problems:
        answer = lm.chat([("user", prob.prompt)], max_new_tokens=max_new_tokens, temperature=0)
        ok = run_tests(extract_code(answer), prob.tests)
        greedy_passes += ok
        record[prob.name] = [answer]
        if ok:
            solved.append(prob.name)
        if samples > 1:
            correct = 0
            for _ in range(samples):
                sampled = lm.chat([("user", prob.prompt)], max_new_tokens=max_new_tokens, temperature=temperature)
                correct += run_tests(extract_code(sampled), prob.tests)
            per_problem.append((samples, correct))
        if log:
            log(f"  {'PASS' if ok else 'fail'}  {prob.name}")
    k = min(samples, 5) if samples > 1 else 1
    pk = (sum(pass_at_k(n, c, k) for n, c in per_problem) / len(per_problem)) if per_problem else None
    return FunctionalResult(greedy_passes / len(problems), pk, k, solved, record)


@torch.no_grad()
def bits_per_byte(model, tokenizer, texts: list[str], *, device: str = "cpu", max_tokens: int = 200_000) -> float:
    """Average bits per UTF-8 byte the model needs to encode ``texts``."""
    model.eval()
    block = model.cfg.block_size
    total_nll = 0.0
    total_bytes = 0
    budget = max_tokens
    for text in texts:
        ids = tokenizer.encode(text, allow_special=False)
        for start in range(0, len(ids) - 1, block):
            chunk = ids[start: start + block + 1]
            if len(chunk) < 2 or budget <= 0:
                continue
            x = torch.tensor([chunk[:-1]], device=device)
            y = torch.tensor([chunk[1:]], device=device)
            logits, _ = model(x, y)
            nll = torch.nn.functional.cross_entropy(logits[0].float(), y[0], reduction="sum").item()
            total_nll += nll
            total_bytes += len(tokenizer.decode(chunk[1:]).encode("utf-8"))
            budget -= len(chunk) - 1
        if budget <= 0:
            break
    if total_bytes == 0:
        raise ValueError("no evaluation text")
    return total_nll / math.log(2) / total_bytes


def load_heldout_texts(sources: list[Path], max_chars: int = 600_000) -> list[str]:
    from ycode.lm.data import read_code_files

    texts, size = [], 0
    for _, text in read_code_files(sources):
        texts.append(text)
        size += len(text)
        if size >= max_chars:
            break
    return texts
