"""V10 R2: freeze the grammar v3 split and build its test sets (40 probe inputs per task).

usage: grammar3_sets.py     (writes ycode/lm/v10_grammar3_split.json and experiments/v10/sets/c3_{test,trained}.json)

* train: R2's programs (every kept combination except the test and those held back, per source);
* c3_test: 60 combinations not trained (12 per source) whose view and aggregate each occur in that
  source's training programs;
* c3_trained: 40 of the training programs (8 per source), to measure reproduction.
The split file records the hygiene counts and the held-out-family tasks grammar v3 can express
(``expressible_t_families``), measured before any training.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "v8"))

from build_sets import program_tasks  # noqa: E402

from ycode.lm import compositions3 as G  # noqa: E402


def main() -> None:
    kept, report = G.grammar_programs()
    split = G.grammar_split(kept)
    names = {k: [p.name for p in v] for k, v in split.items()}
    out = {"version": "v10-grammar3-1", "report": report, "counts": {k: len(v) for k, v in names.items()}, **names}
    (ROOT / "ycode" / "lm" / "v10_grammar3_split.json").write_text(json.dumps(out, indent=1))
    (ROOT / "experiments" / "v10" / "sets").mkdir(exist_ok=True)
    for name, ps in (("c3_test", split["test"]), ("c3_trained", split["trained_sample"])):
        tasks = program_tasks(ps, name, kept)
        (ROOT / "experiments" / "v10" / "sets" / f"{name}.json").write_text(json.dumps(
            {"version": f"{name}-v1", "variants": ["original"], "probes": 40, "tasks": tasks}, indent=1))
    print(json.dumps({**out["counts"], "kept_by_source": report["kept_by_source"],
                      "expressible": report["expressible_t_families"]}))


if __name__ == "__main__":
    main()
