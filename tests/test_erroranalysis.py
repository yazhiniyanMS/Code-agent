"""Error analysis: per-assertion diagnosis, failure categories, run-to-run comparison."""

import pytest

pytest.importorskip("torch")

from ycode.lm.erroranalysis import analyze, classify, compare, diagnose, parse_tests  # noqa: E402
from ycode.lm.evaluate import BUGGY, PROBLEMS  # noqa: E402
from ycode.lm.references import FRESH_REFERENCE, REFERENCE  # noqa: E402


def test_parse_tests_understands_every_assert_form():
    kinds = [k for *_, k in parse_tests("assert f(1) == 2\nassert abs(f(2) - 0.5) < 1e-9\nassert f(0)\nassert not f(3)")]
    assert kinds == ["eq", "approx", "truth", "truth"]
    assert parse_tests("assert not f(3)")[0][2] == "False"


def test_diagnose_reports_expected_and_actual_per_assertion():
    diag = diagnose("def f(x):\n    return x + 1", "assert f(1) == 2\nassert f(2) == 5")
    assert [a["ok"] for a in diag["asserts"]] == [True, False]
    assert diag["asserts"][1]["actual"] == "3" and diag["asserts"][1]["expected"] == "5"
    assert diagnose("def f(:\n  pass", "assert f()")["error"][0] == "SyntaxError"
    assert diagnose("def f():\n    while True:\n        pass", "assert f()", timeout=2)["error"][0] == "Timeout"


@pytest.mark.parametrize("name, code, category", [
    ("f", "", "formatting"),
    ("f", "deff(x):\n    return x", "formatting"),
    ("f", "def g(x):\n    return x", "instruction_following"),
    ("to_upper", "def to_upper(s):\n    return s.lower()", "instruction_following"),  # to_lower's body
    ("f", "def f(x):\n    return helper(x)", "hallucination"),
    ("f", "def f(x):\n    return x.toupper()", "hallucination"),
    ("f", "def f(x):\n    return [][x]", "coding"),
    ("seconds_to_minutes", "def seconds_to_minutes(seconds):\n    return seconds * 61", "factual"),
    ("f", "def f(x):\n    return x * 7 - 100", "reasoning"),
])
def test_classify_puts_each_failure_in_one_category(name, code, category):
    tests = f"assert {name}(2) == 3"
    assert classify(name, code, diagnose(code, tests))[0] == category


def test_handing_the_bug_back_is_instruction_following():
    sig, buggy = BUGGY[0]
    name = sig.split("(")[0]
    tests = {p.name: p.tests for p in PROBLEMS}[name]
    category, reason = classify(name, buggy, diagnose(buggy, tests), buggy=buggy)
    assert category == "instruction_following" and "unchanged" in reason


def test_analysis_and_comparison_of_two_runs():
    rows = [{"suite": "write", "name": p.name, "prompt": p.prompt, "answer": "", "pass": p.name == "add",
             "code": REFERENCE[p.name] if p.name == "add" else f"def {p.name}(x):\n    return x"} for p in PROBLEMS[:3]]
    rows.append({"suite": "write", "name": "triple", "prompt": "", "answer": "", "pass": True,
                 "code": FRESH_REFERENCE["triple"]})
    first = analyze(rows, "bench")
    assert first["points"] == 2 and first["of"] == 4
    assert sum(first["lost_by_category"].values()) == 2
    assert first["groups"]["fresh"] == {"passed": 1, "total": 1, "lost": {}}
    rows[1]["pass"], rows[1]["code"] = True, REFERENCE[rows[1]["name"]]
    diff = compare(first, analyze(rows, "bench"))
    assert diff["points"] == [2, 3] and diff["gained"] == [rows[1]["name"]] and diff["lost"] == []


def test_every_reference_passes_its_tests():
    from ycode.lm.evaluate import fresh_problems, run_tests

    assert set(FRESH_REFERENCE) == {p.name for p in fresh_problems()}
    for p in fresh_problems():
        assert run_tests(FRESH_REFERENCE[p.name], p.tests), p.name


def test_training_generators_never_import_the_references():
    from pathlib import Path

    import ycode.lm as lm

    for name in ("basics.py", "compose.py", "skills.py", "data.py", "devset.py"):
        assert "references" not in (Path(lm.__file__).parent / name).read_text()
