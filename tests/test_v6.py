"""V6: idiom curriculum, skeleton-level held-out exclusion, unseen-v2 and the items-lost rule."""

import pytest

from ycode.lm import idioms
from ycode.lm.leakage import held_out_items, held_out_skeletons, skeleton


def test_skeleton_identifies_near_twins_and_separates_compositions():
    inc = "def count_increases(numbers):\n    return sum(1 for a, b in zip(numbers, numbers[1:]) if b > a)"
    dec = "def f(xs):\n    '''doc'''\n    return sum(1 for p, q in zip(xs, xs[1:]) if q < p)"
    assert skeleton(inc) == skeleton(dec)  # other names, operator and docstring: a near-twin
    assert skeleton("def f(s):\n    return s.upper()") == skeleton("def g(t):\n    return t.lower()")
    assert skeleton("def f(a, b):\n    return max(a, b)") == skeleton("def f(a, b):\n    return min(a, b)")
    assert skeleton("def f(x):\n    return x * -3") == skeleton("def g(h):\n    return h * 3600")
    # an idiom inside another operation is a different program
    assert skeleton("def f(items):\n    return items[1:]") != skeleton("def f(items):\n    return sum(items[1:])")
    assert skeleton("not python") is None


@pytest.fixture(scope="module")
def programs():
    return idioms.candidate_programs()


@pytest.fixture(scope="module")
def kept(programs):
    return idioms.held_out_check(programs)[0]


def test_every_curriculum_program_is_verified_by_execution(programs):
    # candidate_programs raises if an implementation fails, implementations disagree or the output is constant
    assert len(programs) > 200
    assert all(len(p.inputs) >= 20 for p in programs)
    assert len({p.family for p in programs}) == len(idioms.FAMILIES)


def test_kept_programs_are_not_held_out_by_name_skeleton_or_behaviour(kept):
    from ycode.lm.basics import held_out_names
    from ycode.lm.leakage import behaviour_leaks

    items = held_out_items()
    names = held_out_names() | {i.name for i in items}
    skeletons = held_out_skeletons(items)
    assert kept and not {p.name for p in kept} & names
    for p in kept:
        for i in range(len(p.bodies)):
            assert skeleton(p.source(i)) not in skeletons, p.name
    functions = {p.source(0): (p.name, len([x for x in p.params.split(",") if x.strip()])) for p in kept}
    assert behaviour_leaks(functions, items, near=1.0) == []


def test_held_out_check_gives_way_to_unseen2(programs, kept):
    from ycode.lm.unseen2 import unseen2_references

    v2 = {skeleton(src) for src in unseen2_references().values()}
    assert not [p.name for p in kept for i in range(len(p.bodies)) if skeleton(p.source(i)) in v2]


def test_idiom_examples_answers_are_the_verified_programs(kept):
    import re

    examples = idioms.idiom_examples(5, write_per_program=1, bugfix_per_program=1, programs=kept)
    bodies = {skeleton(p.source(i)) for p in kept for i in range(len(p.bodies))}
    writes = [e for e in examples if not e.prompt.startswith("This function has a bug")]
    fixes = [e for e in examples if e.prompt.startswith("This function has a bug")]
    assert len(writes) == len(kept) and len(fixes) > 0.8 * len(kept)
    for e in examples:
        code = re.search(r"```python\n(.*?)\n```", e.response, re.S).group(1)
        assert skeleton(code) in bodies  # renamed or not, the answer is one of the verified implementations
    for e in fixes:
        buggy = re.search(r"```python\n(.*?)\n```", e.prompt, re.S).group(1)
        assert buggy != re.search(r"```python\n(.*?)\n```", e.response, re.S).group(1)
    assert len({e.prompt.split("`")[0] for e in writes}) >= 3  # several prompt templates


def test_round_data_defaults_are_unchanged_and_idioms_are_opt_in():
    from ycode.lm import rounddata

    assert rounddata.DEFAULTS["idioms_write"] == 0 and rounddata.DEFAULTS["idioms_bugfix"] == 0
    zero = dict(basics_write=0, basics_bugfix=0, compose_write=0, compose_bugfix=0, contrast=0, novel_write=0)
    base = len(rounddata.round_examples(3, {**rounddata.DEFAULTS, **zero}))
    assert len(rounddata.round_examples(3, {**rounddata.DEFAULTS, **zero, "idioms_write": 1})) > base + 200


def test_unseen2_is_a_held_out_split():
    from ycode.lm.basics import _passes, held_out_names
    from ycode.lm.evalsets import eval_items, manifest
    from ycode.lm.unseen2 import unseen2_names, unseen2_problems, unseen2_references

    refs = unseen2_references()
    problems = unseen2_problems()
    assert len(problems) == 71
    assert all(_passes(refs[p.name], p.tests, timeout=2.0) for p in problems)
    assert all(p.tests.count("assert") >= 4 for p in problems)
    assert unseen2_names() <= held_out_names()
    assert {i.name for i in held_out_items() if i.split == "unseen2"} == unseen2_names()
    items = eval_items("unseen2")
    assert len(items) == 71 and not any(i.contaminated for i in items)
    assert manifest()["unseen2"]["items"] == 71


def test_items_lost_rule_ignores_failures_that_only_change_type():
    from ycode.lm.experiment import compare

    def run(passed, categories, dev_clean):
        block = {"passed": passed, "categories": categories, "contaminated": [], "clean": dev_clean, "points": dev_clean,
                 "lost_by_category": {c: list(categories.values()).count(c) for c in ("coding", "reasoning")}}
        return {"dev": block, "unseen": {**block, "clean": 0, "passed": [], "categories": {},
                                         "lost_by_category": {"coding": 0, "reasoning": 0}}}

    base = {s: run(["write:a"], {"write:b": "reasoning", "write:c": "reasoning", "write:d": "reasoning"}, 1) for s in (1, 2)}
    # the candidate also passes a, and fails b, c, d with a different failure type: nothing was lost
    same = {s: run(["write:a"], {"write:b": "coding", "write:c": "coding", "write:d": "coding"}, 1) for s in (1, 2)}
    assert compare(same, base, "unseen_clean", "items_lost")["category_change"]["coding"] == 0
    assert compare(same, base, "unseen_clean", "failure_counts")["category_change"]["coding"] == 3  # the v5.1 rule
    # a candidate that loses a: one item lost, in its failure category
    lost = {s: run([], {"write:a": "coding", "write:b": "reasoning", "write:c": "reasoning", "write:d": "reasoning"}, 0)
            for s in (1, 2)}
    assert compare(lost, base, "unseen_clean", "items_lost")["category_change"]["coding"] == 1


def test_sibling_repairs_show_plausible_wrong_code_and_fix_it(kept):
    import re

    examples = idioms.idiom_examples(11, write_per_program=0, bugfix_per_program=0, contrast_per_program=1, programs=kept)
    assert len(examples) > 0.7 * len(kept)
    from collections import Counter

    behaviours = Counter(p.name for p in {p.source(0): p for p in kept}.values())
    by_name = {p.name: p for p in kept if behaviours[p.name] == 1}  # some names cover variants told apart by the task
    checked = 0
    for e in examples:
        wrong = re.search(r"```python\n(.*?)\n```", e.prompt, re.S).group(1)
        fixed = re.search(r"```python\n(.*?)\n```", e.response, re.S).group(1)
        name = re.match(r"def (\w+)\(", fixed).group(1)
        assert '"""' in wrong and wrong.split("\n")[:2] == fixed.split("\n")[:2]  # same signature and docstring
        if name in by_name:  # not renamed: compare behaviour with the program
            p = by_name[name]
            want = idioms._run(p.source(0), name, list(p.inputs))
            assert idioms._run(fixed, name, list(p.inputs)) == want
            assert idioms._run(wrong, name, list(p.inputs)) != want
            checked += 1
    assert checked > 0.5 * len(examples)


def test_worked_examples_in_prompts_are_correct(kept):
    import re

    examples = idioms.idiom_examples(12, write_per_program=2, bugfix_per_program=0, io_fraction=1.0, programs=kept)
    with_io = [e for e in examples if "For example, " in e.prompt]
    assert len(with_io) > 0.7 * len(examples)
    for e in with_io[:150]:
        ns: dict = {}
        exec(re.search(r"```python\n(.*?)\n```", e.response, re.S).group(1), ns)
        for call, out in re.findall(r"`(\w+\(.*?\))` returns `(.*?)`", e.prompt.split("For example, ")[1]):
            assert eval(call, ns) == eval(out)


def test_probe_sets_stay_on_curriculum_v1():
    assert len(idioms.FAMILIES_V1) == 76 and len(idioms.FAMILIES) > 76
    assert all(f.key in idioms.FAMILIES_V1 for f in idioms.FAMILIES[:76])
