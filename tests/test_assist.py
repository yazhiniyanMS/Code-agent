"""Answers without an API: Python documentation lookups and checked function writing."""

from ycode.lm.assist import (
    CHECKED_NOTE,
    DOC_SOURCE_NOTE,
    UNVERIFIED_NOTE,
    Assistant,
    defines,
    docs_answer,
    requested_function,
)


def test_docs_answers_for_builtins_stdlib_and_keywords():
    assert "Return the number of items" in docs_answer("What does the len() function do?")
    assert "json.loads(" in docs_answer("how do I use json.loads")
    assert "os.path.join(" in docs_answer("Explain os.path.join")
    assert "Yield tuples" in docs_answer("what does `zip` do?")
    assert "(Python keyword)" in docs_answer("what is yield?")
    both = docs_answer("what is the difference between a list and a tuple")
    assert "mutable sequence" in both and "immutable sequence" in both
    assert docs_answer("What does len() do?").endswith(DOC_SOURCE_NOTE)


def test_docs_only_for_documentation_questions():
    assert docs_answer("Write a function that uses len()") is None  # a coding request, not a docs one
    assert docs_answer("hello there") is None
    assert docs_answer("what does my_project.helpers do?") is None  # never imports the user's code
    assert docs_answer("what does _thread do?") is None


def test_requested_function_and_defines():
    assert requested_function("Write a Python function is_even(n) that returns True if n is even.") == "is_even"
    assert requested_function("Write a function called `slugify(text)`") == "slugify"
    assert requested_function("How does sorting work?") is None
    assert defines("def is_even(n):\n    return n % 2 == 0\n", "is_even")
    assert not defines("def other(n):\n    return n\n", "is_even")
    assert not defines("def is_even(n) return", "is_even")


class ScriptedLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.temperatures = []

    def chat(self, turns, *, temperature=0.7, **_kw):
        self.temperatures.append(temperature)
        return self.replies.pop(0)


def test_assistant_returns_first_candidate_defining_the_function():
    lm = ScriptedLM(["```python\ndef wrong_name(s):\n    return s\n```",
                     "```python\ndef reverse_string(s):\n    return s[::-1]\n```"])
    streamed = []
    answer = Assistant(lm, candidates=4).answer(
        [("user", "Write a function reverse_string(s) that returns s reversed.")], on_text=streamed.append)
    assert "return s[::-1]" in answer and CHECKED_NOTE in answer
    assert "".join(streamed) == answer
    assert lm.temperatures == [0.2, 0.7]  # nearly greedy first, then sampling


def test_assistant_flags_unverified_code():
    lm = ScriptedLM(["def nope(x): pass"] * 3)
    answer = Assistant(lm, candidates=3).answer([("user", "Write a function add(a, b).")])
    assert answer.endswith(UNVERIFIED_NOTE) and len(lm.temperatures) == 3


def test_assistant_uses_docs_before_the_model():
    lm = ScriptedLM([])
    answer = Assistant(lm).answer([("user", "What does the len() function do?")])
    assert "Return the number of items" in answer and lm.temperatures == []
