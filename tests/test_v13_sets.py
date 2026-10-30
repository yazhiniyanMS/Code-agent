"""V13's string-operation evaluation sets: reproducible, scored as experiments/v7/prompt_diagnostic.py scores them, and
the held-out set is held out from every training concept and distinct from every evaluation item."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ycode.lm import basics, strings

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / "experiments" / "v13" / "sets"


def _module(path: Path, name: str):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sets():
    return {name: json.loads((SETS / f"{name}.json").read_text()) for name in ("strings_trained", "strings_heldout")}


def test_the_committed_sets_are_what_the_builder_makes(sets):
    build = _module(ROOT / "experiments" / "v13" / "build_sets.py", "v13_build_sets")
    assert json.loads(json.dumps(build.build())) == sets


def test_every_task_has_40_probes_and_the_trained_set_is_the_curriculum(sets):
    assert [t["name"] for t in sets["strings_trained"]["tasks"]] == [c.name for c in strings.STRING_CONCEPTS]
    assert len(sets["strings_heldout"]["tasks"]) == 32
    for data in sets.values():
        assert all(len(t["probe_args"]) == len(t["probe_expected"]) == 40 for t in data["tasks"])


def test_the_held_out_tasks_are_held_out_from_training_and_distinct_from_evaluation_items(sets):
    from ycode.lm import leakage

    held = basics.held_out_names() | basics.dev_names()
    trained = list(strings.STRING_CONCEPTS) + list(basics.CONCEPTS)
    taken = {n for c in trained for n in (c.name, *c.aliases)}
    tasks = sets["strings_heldout"]["tasks"]
    assert not {t["name"] for t in tasks} & (held | taken)
    sys.path.insert(0, str(ROOT / "experiments" / "v7"))
    from neighbors import run

    for t in tasks:  # no training function computes a held-out task: it differs on at least one of the 40 probes
        arity = len([p for p in t["params"].split(",") if p.strip()])
        args = [eval(a) for a in t["probe_args"]]  # noqa: S307 - repr strings of our own probe tuples
        for c in trained:
            if len([p for p in c.params.split(",") if p.strip()]) == arity:
                got = [repr(v) if kind == "ok" else None for kind, v in run(c.source(t["name"]), t["name"], args)]
                assert got != t["probe_expected"], (t["name"], c.name)
    functions = leakage.answer_functions(f"```python\n{t['reference']}\n```" for t in tasks)
    assert leakage.behaviour_leaks(functions, near=1.0) == []  # not a copy of a benchmark, dev or unseen item


def test_the_scorer_marks_each_reference_strict_and_each_neighbour_wrong(sets):
    diagnostic = _module(ROOT / "experiments" / "v7" / "prompt_diagnostic.py", "v7_prompt_diagnostic")
    for data in sets.values():
        for t in data["tasks"]:
            assert diagnostic.outputs(t["reference"], t["name"], t["probe_args"], literal=True) == t["probe_expected"]
    for t in sets["strings_heldout"]["tasks"]:
        got = diagnostic.outputs(t["wrong"], t["name"], t["test_inputs"])
        assert got == t["wrong_outputs"] and got != t["expected"], t["name"]  # the neighbour fails a test
