"""V13: the string-operations curriculum is correct, covers its eight groups and their boundary cases, and
cannot leak an evaluation item into training."""
import random
import re

from ycode.lm import basics, strings
from ycode.lm.data import InstructionExample

LITERAL = re.compile(r"'((?:[^'\\]|\\.)*)'")


def _literals(concept) -> list[str]:
    return LITERAL.findall(concept.tests)


def test_every_reference_passes_its_tests_and_every_test_set_has_an_empty_input():
    for c in strings.STRING_CONCEPTS:
        assert basics._passes(c.source(), basics._rename_tests(c.tests, c.name)), c.name
        assert "''" in c.tests or "[]" in c.tests, f"{c.name}: no empty-string (or empty-list) case"
        assert c.tests.count("assert") >= 3, f"{c.name}: fewer than 3 tests"


def test_the_groups_cover_the_eight_operations_and_the_boundary_cases():
    assert list(strings.GROUPS) == ["replace", "split_join", "strip", "filter_transform", "substring", "case",
                                    "composition", "minimal_pairs"]
    assert all(len(group) >= 5 for group in strings.GROUPS.values())
    assert len(strings.BY_NAME) == len(strings.STRING_CONCEPTS) == 47  # no name used twice
    literals = [s for c in strings.STRING_CONCEPTS for s in _literals(c)]
    assert sum(bool(s.strip()) and s != s.strip() for s in literals) >= 10  # leading or trailing spaces around text
    assert sum(bool(re.search(r"(.)\1\1", s)) for s in literals) >= 5  # a character repeated three times
    assert sum(any(p in s for p in ".,!?;:") for s in literals) >= 10  # punctuation


def test_no_name_is_an_evaluation_or_existing_training_name():
    held = basics.held_out_names() | basics.dev_names()
    taken = {n for c in basics.CONCEPTS for n in (c.name, *c.aliases)}
    assert not {c.name for c in strings.STRING_CONCEPTS} & (held | taken)


def test_no_reference_comes_close_to_computing_an_evaluation_item():
    from ycode.lm import leakage

    functions = leakage.answer_functions(f"```python\n{c.source()}\n```" for c in strings.STRING_CONCEPTS)
    assert len(functions) == len(strings.STRING_CONCEPTS)
    # agreement 0.8, stricter than the dataset hygiene check's 1.0 (benchmark, dev, unseen-v1 and unseen-v2 items;
    # only their reference solutions run, never the model)
    assert leakage.behaviour_leaks(functions, near=0.8) == []


def test_no_reference_duplicates_an_existing_training_concept():
    for s in strings.STRING_CONCEPTS:
        for t in basics.CONCEPTS:
            if s.params.count(",") != t.params.count(","):
                continue
            both = (basics._passes(s.source(t.name), basics._rename_tests(t.tests, t.name))
                    and basics._passes(t.source(s.name), basics._rename_tests(s.tests, s.name)))
            assert not both, f"{s.name} passes {t.name}'s tests and the other way round"


def test_minimal_pairs_share_most_of_their_wording_and_differ_in_behaviour():
    words = lambda text: set(re.findall(r"[a-z]+", text.lower()))  # noqa: E731
    assert len(strings.MINIMAL_PAIRS) >= 10
    for a, b in strings.MINIMAL_PAIRS:
        ca, cb = strings.BY_NAME[a], strings.BY_NAME[b]
        wa, wb = words(ca.task), words(cb.task)
        assert len(wa & wb) / len(wa | wb) >= 0.5, (a, b)
        assert not basics._passes(ca.source(b), basics._rename_tests(cb.tests, b)), (a, b)
        assert not basics._passes(cb.source(a), basics._rename_tests(ca.tests, a)), (b, a)
    paired = {n for pair in strings.MINIMAL_PAIRS for n in pair}
    assert {c.name for c in strings.GROUPS["minimal_pairs"]} <= paired


def test_generated_examples_are_checked_under_their_final_names_and_reproducible():
    made = strings.string_examples(random.Random(7), write_per_concept=3, bugfix_per_concept=2)
    assert made == strings.string_examples(random.Random(7), write_per_concept=3, bugfix_per_concept=2)
    tags = [tag for _, tag in made]
    assert len({(ex.prompt, ex.response) for ex, _ in made}) == len(made)  # no exact repeats
    writes = [t for t in tags if t.startswith("strings:write:")]
    assert {t.rsplit(":", 1)[1] for t in writes} == set(strings.BY_NAME)  # every concept is written
    assert len(writes) >= 2.5 * len(strings.STRING_CONCEPTS)  # of 3 asked per concept, minus exact repeats
    assert any(t.startswith("strings:bugfix:") for t in tags)
    for ex, tag in made:
        assert strings.check_example(ex, strings.BY_NAME[tag.rsplit(":", 1)[1]])
    neutral = [ex for ex, tag in made if f"def {tag.rsplit(':', 1)[1]}(" not in ex.response]
    assert neutral  # some answers were renamed (basics.neutralize), and they were checked under the new name
    wrong = InstructionExample("Write `strip_start(text)`.", "```python\ndef g(text):\n    return text.rstrip()\n```")
    assert not strings.check_example(wrong, strings.BY_NAME["strip_start"])


def test_the_round_option_adds_tagged_examples_that_pass_the_dataset_hygiene_check():
    from ycode.lm.datacheck import clean_examples
    from ycode.lm.rounddata import PRODUCTION, round_examples_tagged

    none = {k: 0 for k in PRODUCTION}  # (novel exercises keep one bug fix each even at novel_write 0)
    examples, tags = round_examples_tagged(1301, {**none, "strings_write": 2, "strings_bugfix": 1})
    mine = [ex for ex, tag in zip(examples, tags) if tag.startswith("strings:")]
    assert len(mine) >= 2 * len(strings.STRING_CONCEPTS)
    kept, removed = clean_examples(mine, dedup=True, contradictions=True, held_out=True)
    assert removed == [], [(r.reason, r.detail) for r in removed]
    # the string stream is the round's own: the other examples of a round do not change with it
    base, _ = round_examples_tagged(1301, {"basics_write": 1, **{k: 0 for k in PRODUCTION if k != "basics_write"}})
    both, _ = round_examples_tagged(1301, {"basics_write": 1, **{k: 0 for k in PRODUCTION if k != "basics_write"},
                                           "strings_write": 1})
    assert both[:len(base)] == base


def test_every_concept_gets_bug_fixes_and_every_bug_is_real():
    from ycode.lm.evaluate import extract_code

    made = strings.string_examples(random.Random(1302), write_per_concept=1, bugfix_per_concept=2)
    fixes = [(ex, strings.BY_NAME[tag.rsplit(":", 1)[1]]) for ex, tag in made if ":bugfix:" in tag]
    assert {c.name for _, c in fixes} == set(strings.BY_NAME)  # STRING_MUTATIONS reach every concept
    for ex, concept in fixes:
        buggy = extract_code(ex.prompt)
        name = strings._name_of(buggy)
        assert basics._parses(buggy) and not basics._passes(buggy, basics._rename_tests(concept.tests, name)), buggy
        assert strings.check_example(ex, concept)
    # the production generators keep their own mutations
    assert basics.bugfix_examples.__defaults__[-1] is basics._TEXT_MUTATIONS
    assert not set(strings.STRING_MUTATIONS) & set(basics._TEXT_MUTATIONS)
