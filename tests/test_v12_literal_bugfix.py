"""V12: bug fixes whose fix restores a string literal (the class round 54's training mix lacks)."""
import random
import re

from ycode.lm import basics, rounddata

R1_OPTIONS = {"idioms_families": "v3", "idioms_select": "v9_grammar_train", "idioms_write": 2, "basics_write": 0,
              "basics_bugfix": 0, "compose_write": 0, "compose_bugfix": 0, "contrast": 0, "novel_write": 0,
              "replay": 0}
LITERAL = re.compile(r"'[^']*'|\"[^\"]*\"")


def _code(text: str) -> str:
    return re.search(r"```python\n(.*?)\n```", text, re.S).group(1)


def test_every_literal_bug_is_real_and_its_fix_is_the_verified_reference():
    examples = basics.literal_bugfix_examples(random.Random(1), per_bug=1)
    assert len(examples) >= 25  # 31 distinct swaps over 13 of the 20 concepts that use literals
    concepts = basics.training_concepts()
    restored = set()
    for e in examples:
        assert e.prompt.startswith(basics.BUGFIX_PROMPT.split("{code}")[0])
        buggy, fixed = _code(e.prompt), _code(e.response)
        diff = [(b, f) for b, f in zip(buggy.splitlines(), fixed.splitlines()) if b != f]
        assert len(diff) == 1 and len(buggy.splitlines()) == len(fixed.splitlines())  # one line changes
        b, f = diff[0]
        assert LITERAL.sub("S", b) == LITERAL.sub("S", f)  # ... and only in its string literals
        # whatever the function is called (names are sometimes neutralised), it is its concept's reference
        name = re.search(r"^def (\w+)\(", fixed, re.M).group(1)
        concept = next(c for c in concepts if fixed.endswith(c.body))
        tests = basics._rename_tests(concept.tests, name)
        assert basics._passes(fixed, tests) and not basics._passes(buggy, tests)
        restored.update(set(LITERAL.findall(f)) - set(LITERAL.findall(b)))
    assert {"''", "' '", "'_'"} <= restored  # the right literal differs by task: no single answer to copy


def test_no_literal_bugfix_example_uses_an_evaluation_name():
    held_out = basics.held_out_names()
    for e in basics.literal_bugfix_examples(random.Random(2), per_bug=3):
        assert not set(re.findall(r"def (\w+)\(", e.response)) & held_out


def test_the_option_adds_examples_without_changing_the_rest_of_the_round():
    base, base_tags = rounddata.round_examples_tagged(1302, {**rounddata.DEFAULTS, **R1_OPTIONS})
    more, more_tags = rounddata.round_examples_tagged(1302, {**rounddata.DEFAULTS, **R1_OPTIONS, "literal_bugfix": 3})
    assert more[:len(base)] == base and more_tags[:len(base)] == base_tags
    extra = more_tags[len(base):]
    assert len(extra) == len(more) - len(base) >= 75 and set(extra) == {"retention:bugfix"}
    assert rounddata.source_group(extra[0]) == "retention"
