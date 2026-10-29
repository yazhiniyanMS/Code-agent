"""V12: round 54, C2, R1 and V11-A against every V12 run, with V10's comparison code unchanged.

usage: compare.py          (prints markdown tables: scores, forgetting by category, cost and identity)

The tables, scoring and the items_lost category rule are experiments/v10/compare.py's (through
experiments/v11/compare.py); this only changes the list of models.
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("v11_compare", ROOT / "experiments" / "v11" / "compare.py")
v11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v11)
KEEP = ("Round 54", "C2", "R1 1301", "V11-A 1302")


def models() -> list[dict]:
    out = [m for m in v11.models() if m["name"] in KEEP]
    for f in sorted((ROOT / "experiments" / "v12" / "results" / "runs").glob("*/*/result.json")):
        out.append({"name": f"V12-{f.parts[-3]} {f.parts[-2]}", "dir": f.parent, "rec": json.loads(f.read_text()),
                    "sets": {s: v11.v10.strict(f.parent / f"set_{s}.jsonl") for s in v11.v10.SETS}})
    return out


v11.v10.models = models

if __name__ == "__main__":
    v11.v10.main()
