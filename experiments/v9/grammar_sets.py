"""V9-1: freeze the grammar v2 split and build its test sets (40 probe inputs per task).

usage: grammar_sets.py      (writes ycode/lm/v9_grammar_split.json and experiments/v9/sets/c2_{test,trained}.json)

* train: the C2 arm's programs (350, per domain: ycode.lm.compositions2.TRAIN_PER_DOMAIN);
* c2_test: 60 combinations not trained (15 per domain) whose selection, transform and aggregate each
  occur in that domain's training programs;
* c2_trained: 40 of the training programs (10 per domain), to measure reproduction.
The split file also records the hygiene counts and the held-out-family tasks grammar v2 can express
(``expressible_t_families``), measured before any training.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "v8"))

from build_sets import program_tasks  # noqa: E402

from ycode.lm import compositions2 as G  # noqa: E402


def main() -> None:
    kept, report = G.grammar_programs()
    split = G.grammar_split(kept)
    names = {k: [p.name for p in v] for k, v in split.items()}
    out = {"version": "v9-grammar-1", "report": report, "counts": {k: len(v) for k, v in names.items()}, **names}
    (ROOT / "ycode" / "lm" / "v9_grammar_split.json").write_text(json.dumps(out, indent=1))
    for name, ps in (("c2_test", split["test"]), ("c2_trained", split["trained_sample"])):
        tasks = program_tasks(ps, name, kept)
        (ROOT / "experiments" / "v9" / "sets" / f"{name}.json").write_text(json.dumps(
            {"version": f"{name}-v1", "variants": ["original"], "probes": 40, "tasks": tasks}, indent=1))
    print(json.dumps({**out["counts"], "kept_by_domain": report["kept_by_domain"],
                      "expressible": report["expressible_t_families"]}))


if __name__ == "__main__":
    main()
