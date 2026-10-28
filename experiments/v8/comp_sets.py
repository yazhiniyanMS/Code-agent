"""V8 stage 3: freeze the composition split and build its test sets (40 probe inputs per task).

usage: comp_sets.py      (writes ycode/lm/v8_comp_split.json and experiments/v8/sets/c_{more,comp_only,test}.json)

* train: the COMP arm's programs; more: the MORE arm's programs (60 of train);
* c_more: the 60 programs both arms train on (COMP ~3 examples each, MORE ~18);
* c_comp_only: 60 of the programs only COMP trains on (for MORE they are new combinations);
* c_test: 60 combinations no arm trains on, each of whose selection, transform and aggregate occurs
  in training (systematic generalisation to new combinations of known parts).
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "v8"))

from build_sets import program_tasks  # noqa: E402

from ycode.lm import compositions as C  # noqa: E402


def main() -> None:
    kept, counts = C.composition_programs()
    split = C.composition_split(kept)
    rest = [p for p in split["train"] if p not in split["more"]]
    comp_only = random.Random(9).sample(rest, 60)
    names = {k: [p.name for p in v] for k, v in split.items()}
    names["comp_only_sample"] = [p.name for p in comp_only]
    out = {"version": "v8-comp-1", "counts": {**counts, **{k: len(v) for k, v in names.items()}}, **names}
    (ROOT / "ycode" / "lm" / "v8_comp_split.json").write_text(json.dumps(out, indent=1))
    for name, ps in (("c_more", split["more"]), ("c_comp_only", comp_only), ("c_test", split["test"])):
        tasks = program_tasks(ps, name, kept)
        (ROOT / "experiments" / "v8" / "sets" / f"{name}.json").write_text(json.dumps(
            {"version": f"{name}-v1", "variants": ["original"], "probes": 40, "tasks": tasks}, indent=1))
    print(json.dumps(out["counts"]))


if __name__ == "__main__":
    main()
