"""V8 evaluation sets: V7's task sets with 40 probe inputs per task, so every answer is scored for exact
functional correctness on all of them (``strict``), not only on its handful of tests.

usage: build_sets.py            (writes experiments/v8/sets/{k_trained,family_new,neighbours}.json)

* k_trained: the 36 distinct programs of the 12 constant-parameterised v1 curriculum families (V7 L1/L2's
  training set, and the curriculum of V8's replay runs); probes are the program's own 40 inputs;
* family_new: constants of those families that no run trains on (the V6 probe's family_new, held-out
  matches dropped); probes as above;
* neighbours: V7's frozen neighbour set (ycode/lm/neighbours_v1.json, unchanged prompts and tests);
  probes are the test arguments followed by typed variations and seeded random values
  (ycode.lm.leakage.probe_inputs) on which the reference runs, the first 40.
Expected outputs come from the reference. Inputs and outputs are stored as Python repr strings.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "v6"))
sys.path.insert(0, str(ROOT / "experiments" / "v7"))

from neighbors import run  # noqa: E402

from ycode.lm import idioms  # noqa: E402

PROBES = 40


def _probe_fields(reference: str, name: str, candidates: list) -> dict:
    outs = run(reference, name, candidates)
    args = [a for a, (kind, _) in zip(candidates, outs) if kind == "ok"][:PROBES]
    want = [repr(v) for _, (kind, v) in zip(candidates, outs) if kind == "ok"][:PROBES]
    return {"probe_args": [repr(tuple(a)) for a in args], "probe_expected": want}


def program_tasks(programs, group: str, siblings_from) -> list[dict]:
    by_family: dict[str, list] = {}
    for q in siblings_from:
        by_family.setdefault(q.family, []).append(q)
    tasks = []
    for p in programs:
        inputs = sorted(p.inputs, key=lambda a: len(repr(a)))
        tests = inputs[:3] + inputs[len(inputs) // 2: len(inputs) // 2 + 3]
        want = idioms._run(p.source(0), p.name, tests)
        sibling = next((q for q in by_family.get(p.family, []) if q.params == p.params
                        and repr(idioms._run(q.source(0, p.name), p.name, tests)) != repr(want)), None)
        wrong = sibling.source(0, p.name) if sibling else None
        got = idioms._run(wrong, p.name, tests) if wrong else None
        tasks.append({"group": group, "family": p.family, "name": p.name, "params": p.params,
                      "neighbour": sibling.name if sibling else None, "same_as": None,
                      "prompts": {"original": f"Write a Python function `{p.name}({p.params})` that {p.tasks[0]}"},
                      "tests": "\n".join(f"assert {p.name}({', '.join(repr(x) for x in a)}) == {v!r}" for a, v in zip(tests, want)),
                      "test_inputs": [list(a) for a in tests], "reference": p.source(0), "wrong": wrong,
                      "expected": [repr(v) for v in want], "wrong_outputs": [repr(v) for v in got] if got else None,
                      **_probe_fields(p.source(0), p.name, list(p.inputs))})
    return tasks


def distinct(programs):
    seen = {}
    for p in programs:
        seen.setdefault(p.source(0, "f"), p)
    return list(seen.values())


def main() -> None:
    from probe_learning import family_new_programs as fnew

    from ycode.lm.leakage import HeldOutItem, probe_inputs

    out = ROOT / "experiments" / "v8" / "sets"
    k = distinct(idioms.k_family_programs("v1"))
    sets = {"k_trained": program_tasks(k, "k_trained", k)}
    new, dropped = idioms.held_out_check(fnew())
    sets["family_new"] = program_tasks(new, "family_new", k)
    nb = json.loads((ROOT / "ycode" / "lm" / "neighbours_v1.json").read_text())
    tasks = []
    for t in nb["tasks"]:
        arity = len([x for x in t["params"].split(",") if x.strip()])
        cands = probe_inputs(HeldOutItem("neighbours", t["name"], arity, t["tests"], t["reference"]), limit=120)
        tasks.append({**t, **_probe_fields(t["reference"], t["name"], cands)})
    sets["neighbours"] = tasks
    for name, ts in sets.items():
        version = {"k_trained": "k-trained-v2", "family_new": "family-new-v2", "neighbours": "neighbours-v1-probes"}[name]
        variants = nb["variants"] if name == "neighbours" else ["original"]
        (out / f"{name}.json").write_text(json.dumps({"version": version, "variants": variants, "probes": PROBES,
                                                      "tasks": ts}, indent=1))
        counts = [len(t["probe_args"]) for t in ts]
        print(name, len(ts), "tasks; probes per task min", min(counts), "max", max(counts),
              "; held-out matches dropped" if name == "family_new" else "", len(dropped) if name == "family_new" else "")


if __name__ == "__main__":
    main()
