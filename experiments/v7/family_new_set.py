"""V7 positive-control set: new members of trained curriculum families (unused constants).

usage: family_new_set.py OUT.json [--trained | --k-families]
(--trained: 30 programs the curriculum runs trained on, with their training task text: can the
models write what they were trained on?)
The learning probe's ``family_new`` programs (idiom-curriculum v1 families parameterised by k, at
constants no variant of the family uses; no run trained on them) as write tasks in the neighbour
set's format, so prompt_diagnostic.py can score them: the task's own prompt, tests on six of the
program's inputs (the smallest ones first: boundary cases), and as the "wrong" code a trained sibling
of the same family (a constant the curriculum used). Held-out matches are dropped (counts only).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "v6"))

from probe_learning import family_new_programs  # noqa: E402

from ycode.lm import idioms  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1])
    trained, _ = idioms.held_out_check([p for p in idioms.candidate_programs() if p.family in idioms.FAMILIES_V1])
    if "--k-families" in sys.argv:  # every trained program of the constant-parameterised families (L1's training set)
        seen = {}
        for p in idioms.k_family_programs("v1"):
            seen.setdefault(p.source(0, "f"), p)
        kept_new, dropped = list(seen.values()), {}
    elif "--trained" in sys.argv:  # the in-distribution control: 30 programs the v1 curriculum runs trained on
        import random

        seen = {}
        for p in trained:
            seen.setdefault(p.source(0, "f"), p)
        kept_new, dropped = random.Random(7).sample(sorted(seen.values(), key=lambda p: p.name), 30), {}
    else:
        kept_new, dropped = idioms.held_out_check(family_new_programs())
    by_family: dict[str, list] = {}
    for p in trained:
        by_family.setdefault(p.family, []).append(p)
    tasks = []
    for p in kept_new:
        inputs = sorted(p.inputs, key=lambda a: len(repr(a)))
        tests = inputs[:3] + inputs[len(inputs) // 2: len(inputs) // 2 + 3]
        want = idioms._run(p.source(0), p.name, tests)
        sibling = next((q for q in by_family.get(p.family, []) if q.params == p.params
                        and repr(idioms._run(q.source(0, p.name), p.name, tests)) != repr(want)), None)
        if sibling is None or want is None:
            continue
        wrong = sibling.source(0, p.name)
        got = idioms._run(wrong, p.name, tests)
        tasks.append({"group": "family_new", "name": p.name, "params": p.params, "neighbour": sibling.name,
                      "same_as": None, "prompts": {"original": f"Write a Python function `{p.name}({p.params})` that {p.tasks[0]}"},
                      "tests": "\n".join(f"assert {p.name}({', '.join(repr(x) for x in a)}) == {v!r}" for a, v in zip(tests, want)),
                      "test_inputs": [list(a) for a in tests], "reference": p.source(0), "wrong": wrong,
                      "expected": [repr(v) for v in want], "wrong_outputs": [repr(v) for v in got]})
    kind = "k_trained" if "--k-families" in sys.argv else "trained" if "--trained" in sys.argv else "family_new"
    version = {"k_trained": "k-trained-v1", "trained": "trained-v1", "family_new": "family-new-v1"}[kind]
    for t in tasks:
        t["group"] = kind
    out.write_text(json.dumps({"version": version, "variants": ["original"], "tasks": tasks}, indent=1))
    print(len(kept_new) + len(dropped), "programs;", len(dropped), "held-out matches dropped;", len(tasks), "tasks written")


if __name__ == "__main__":
    main()
