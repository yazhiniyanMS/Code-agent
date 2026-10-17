"""V7: neighbour diagnostic set, contrast-token weights and neighbour-pair examples."""

import json
import re
from pathlib import Path

import pytest

from ycode.lm import idioms

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tok():
    from ycode.lm.tokenizer import BPETokenizer

    path = Path("/home/user/lmwork/data/tokenizer.json")
    if not path.exists():
        pytest.skip("production tokenizer not available")
    return BPETokenizer.load(path)


def test_contrast_weights_mark_only_the_deciding_tokens(tok):
    from ycode.lm.data import BODY_WEIGHT, InstructionExample, encode_example

    prompt = "Write a Python function `pick(items)` that returns the fourth item of the list."
    plain = InstructionExample(prompt, "```python\ndef pick(items):\n    return items[3]\n```")
    ids, base = encode_example(tok, plain, weighted=True)
    contrast = InstructionExample(plain.prompt, plain.response, "def pick(items):\n    return items[4]", 20)
    ids2, weighted = encode_example(tok, contrast, weighted=True)
    assert ids2 == ids
    changed = [i for i, (a, b) in enumerate(zip(base, weighted)) if a != b]
    assert changed and all(weighted[i] == 20 for i in changed)
    assert "3" in tok.decode([ids[i] for i in changed]) and "4" not in tok.decode([ids[i] for i in changed])
    assert all(base[i] == BODY_WEIGHT for i in changed)
    # weight 0 (the control) leaves the usual weights
    assert encode_example(tok, InstructionExample(plain.prompt, plain.response, contrast.contrast, 0), weighted=True)[1] == base


@pytest.fixture(scope="module")
def clean_programs():
    return idioms.held_out_check(idioms.candidate_programs(family_list=idioms.families("v3")))[0]


def test_pair_examples_contrast_a_sibling_and_match_the_control(clean_programs):
    treated = idioms.pair_examples(5, per_program=1, weight=20, programs=clean_programs)
    control = idioms.pair_examples(5, per_program=1, weight=0, programs=clean_programs)
    assert len(treated) > 200 and len(treated) == len(control)
    assert [(e.prompt, e.response, e.contrast) for e in treated] == [(e.prompt, e.response, e.contrast) for e in control]
    assert {e.contrast_weight for e in treated} == {20} and {e.contrast_weight for e in control} == {0}
    by_source = {}
    for p in clean_programs:
        for i in range(len(p.bodies)):
            by_source[p.source(i, "f")] = p
    for e in treated[:120]:
        code = re.search(r"```python\n(.*?)\n```", e.response, re.S).group(1)
        name = re.match(r"def (\w+)\(", code).group(1)
        assert re.match(rf"def {name}\(", e.contrast)  # the contrast is renamed with the answer
        p = by_source[re.sub(rf"\b{name}\b", "f", code)]  # the answer is a verified program
        want = idioms._run(code, name, list(p.inputs))
        assert idioms._run(e.contrast, name, list(p.inputs)) != want  # the sibling computes something else
        assert code != e.contrast


def test_pair_programs_avoid_held_out_and_neighbour_skeletons(clean_programs):
    from ycode.lm.leakage import held_out_items, held_out_skeletons, skeleton

    data = json.loads((ROOT / "ycode" / "lm" / "neighbours_v1.json").read_text())
    avoid = idioms.neighbour_skeletons()
    assert len(data["tasks"]) == 38 and avoid
    held = held_out_skeletons(held_out_items())
    for e in idioms.pair_examples(6, per_program=1, weight=20, programs=clean_programs):
        code = re.search(r"```python\n(.*?)\n```", e.response, re.S).group(1)
        assert skeleton(code) not in avoid and skeleton(code) not in held


def test_neighbour_set_is_frozen_and_verified():
    from ycode.lm.evaluate import run_tests

    data = json.loads((ROOT / "ycode" / "lm" / "neighbours_v1.json").read_text())
    groups = [t["group"] for t in data["tasks"]]
    assert groups.count("same_wording") == groups.count("new_wording") == 19
    for t in data["tasks"]:
        assert run_tests(t["reference"], t["tests"])
        assert not run_tests(t["wrong"], t["tests"])  # the neighbour's behaviour fails
        assert set(t["prompts"]) == {"original", "io", "constraint", "distractor"}
        assert all(p.startswith(t["prompts"]["original"]) for p in t["prompts"].values())


def test_k_family_selection_trains_only_constant_families_and_leaves_family_new_untrained():
    import sys

    from ycode.lm import rounddata

    progs = idioms.k_family_programs("v1")
    keys = {p.family for p in progs}
    fams = {f.key: f for f in idioms.families("v1")}
    assert len(keys) == 12 and all(all(set(v) == {"k", "kw"} for v in fams[k].variants) for k in keys)
    sys.path.insert(0, str(ROOT / "experiments" / "v6"))
    from probe_learning import family_new_programs

    trained = {p.source(i, "f") for p in progs for i in range(len(p.bodies))}
    assert not any(p.source(i, "f") in trained for p in family_new_programs() for i in range(len(p.bodies)))
    with pytest.raises(ValueError):
        rounddata._idiom_examples(1, {**rounddata.DEFAULTS, "idioms_write": 1, "idioms_select": "nope"})
