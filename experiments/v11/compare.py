"""V11: round 54, C2, R1 (and R6, R7) against every V11 run, with V10's comparison code unchanged.

usage: compare.py          (prints markdown tables: scores, forgetting by category, cost and identity)

The tables, scoring and the items_lost category rule are experiments/v10/compare.py's; this only adds
experiments/v11/results/runs/* to its list of models.
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("v10_compare", ROOT / "experiments" / "v10" / "compare.py")
v10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v10)
_models = v10.models
KEEP = ("Round 54", "C2", "R1 1301", "R6 1301", "R7 1301")


def models() -> list[dict]:
    out = [m for m in _models() if m["name"] in KEEP]
    for f in sorted((ROOT / "experiments" / "v11" / "results" / "runs").glob("*/*/result.json")):
        out.append({"name": f"V11-{f.parts[-3]} {f.parts[-2]}", "dir": f.parent, "rec": json.loads(f.read_text()),
                    "sets": {s: v10.strict(f.parent / f"set_{s}.jsonl") for s in v10.SETS}})
    return out


v10.models = models

if __name__ == "__main__":
    v10.main()
