"""V13 gates G1-G5 (experiments/v13/PLAN.md, section 4) for one run, against S1 of the same seed.

usage: gates.py RUN_DIR SEED      (prints a JSON verdict; RUN_DIR holds the run's result.json and answers)

* G1 strings improve:        strings_heldout strict >= S1's + 3
* G2 clean evaluation kept:  combined clean dev + unseen-v1 >= S1's - 1
* G3 no category regresses:  fewer than 3 of S1's clean official passes lost in any failure category (V6's items_lost
                             rule, ycode.lm.experiment.compare, with S1 of the same seed as the reference)
* G4 robust-clean kept:      robust-clean dev + unseen-v1 >= S1's - 1
* G5 unrelated families:     neighbours, family_new, t_families, c2_test strict each >= S1's - 2

S1's dev, unseen-v1 and task-set records are V12's (experiments/v12/results/runs/S1/SEED); its string-set records are
V13's baseline (experiments/v13/results/baseline/S1_SEED). A gate whose sets a run has not evaluated yet is "not
measured", never a pass.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S1 = ROOT / "experiments" / "v12" / "results" / "runs" / "S1"
BASELINE = ROOT / "experiments" / "v13" / "results" / "baseline"
FAMILIES = ("neighbours", "family_new", "t_families", "c2_test")


def _strict(rec: dict, name: str) -> int | None:
    s = (rec.get("sets") or {}).get(name)
    return None if s is None else s["scores"]["original"]["strict"]


def _clean_passes(rec: dict, which: str) -> set:
    dirty = set(rec[which].get("contaminated", []))
    return {k for k in rec[which]["passed"] if k not in dirty}


def gates(run: dict, s1: dict, s1_strings: dict) -> dict:
    out = {}
    got, base = _strict(run, "strings_heldout"), _strict(s1_strings, "strings_heldout")
    out["G1"] = None if got is None else {"pass": got >= base + 3, "run": got, "s1": base}
    if run.get("dev") and run.get("unseen"):
        combined = run["dev"]["clean"] + run["unseen"]["clean"]
        ref = s1["dev"]["clean"] + s1["unseen"]["clean"]
        out["G2"] = {"pass": combined >= ref - 1, "run": combined, "s1": ref}
        lost, cats = [], Counter()
        for which in ("dev", "unseen"):
            for k in sorted(_clean_passes(s1, which) - _clean_passes(run, which)):
                lost.append(f"{which}:{k}")
                cats[run[which]["categories"].get(k, "?")] += 1
        out["G3"] = {"pass": all(v < 3 for v in cats.values()), "lost": lost, "by_category": dict(cats)}
        robust = run["dev"]["robust_clean"] + run["unseen"]["robust_clean"]
        ref_robust = s1["dev"]["robust_clean"] + s1["unseen"]["robust_clean"]
        out["G4"] = {"pass": robust >= ref_robust - 1, "run": robust, "s1": ref_robust}
    else:
        out["G2"] = out["G3"] = out["G4"] = None
    fam = {f: (_strict(run, f), _strict(s1, f)) for f in FAMILIES}
    out["G5"] = (None if any(r is None for r, _ in fam.values())
                 else {"pass": all(r >= s - 2 for r, s in fam.values()), **{f: {"run": r, "s1": s} for f, (r, s) in fam.items()}})
    measured = [g for g in out.values() if g is not None]
    out["verdict"] = ("fails" if any(not g["pass"] for g in measured)
                      else "passes" if len(measured) == 5 else "passes so far (not all gates measured)")
    return out


def main(argv: list[str]) -> None:
    run_dir, seed = Path(argv[0]), argv[1]
    run = json.loads((run_dir / "result.json").read_text())
    s1 = json.loads((S1 / seed / "result.json").read_text())
    s1_strings = json.loads((BASELINE / f"S1_{seed}" / "result.json").read_text())
    print(json.dumps(gates(run, s1, s1_strings), indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
