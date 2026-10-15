"""Failure loop, step 1: root causes of a checkpoint's failures on the DEVELOPMENT tasks.

usage: failure_loop.py NAME DEV_ANSWERS.jsonl UNSEEN_V1_ANSWERS.jsonl [OUT.json]
Development tasks only: dev (near trained concepts) and unseen-v1 (whose failures were read to
design the curriculum). The fixed holdout (unseen-v2) is never analysed here.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ycode.lm.erroranalysis import load_answers  # noqa: E402
from ycode.lm.failures import ROOT_CAUSES, classify_answers  # noqa: E402

name, dev, unseen = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
out = {"model": name}
for split, path in (("dev", dev), ("unseen", unseen)):
    r = classify_answers(load_answers(path), split)
    clean = [f for f in r["items"] if not f["contaminated"]]
    out[split] = {"failures_clean": len(clean), "root_causes": dict(Counter(f["root_cause"] for f in clean)),
                  "items": [{k: f[k] for k in ("suite", "name", "root_cause", "class", "evidence")} for f in clean]}
total = Counter()
for split in ("dev", "unseen"):
    total.update(out[split]["root_causes"])
out["total"] = {c: total.get(c, 0) for c in ROOT_CAUSES}
print(name, "| dev clean failures", out["dev"]["failures_clean"], "| unseen-v1 failures", out["unseen"]["failures_clean"])
print("  root causes (dev + unseen-v1):", {c: n for c, n in out["total"].items() if n})
if len(sys.argv) > 4:
    Path(sys.argv[4]).write_text(json.dumps(out, indent=1))
