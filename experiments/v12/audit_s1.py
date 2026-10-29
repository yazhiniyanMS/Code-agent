"""V12-S1's promotion audit (PLAN.md: V11's audit of every promoted candidate's gained items).

For each seed: every clean item gained against round 54 (dev and unseen-v1, official passes), whether
its code also passes the robust tests, and whether its name occurs anywhere in the S1 stage's training
tokens. The dataset is rebuilt from the run's recorded seed and options and must match the run's manifest
byte for byte (sft_tokens.bin, sft_mask.bin, sft_index.npy), so the tokens searched are the ones trained on.
The runs' own hygiene check (name, AST skeleton and behaviour against every evaluation item) is in each
run's hygiene.json.

usage: audit_s1.py OUT.json   (needs the replay data and tokenizer under /home/user/lmwork/data)
"""
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ycode.lm import evalsets  # noqa: E402
from ycode.lm.rounddata import write_round_dataset  # noqa: E402
from ycode.lm.tokenizer import BPETokenizer  # noqa: E402

RUNS = ROOT / "experiments" / "v12" / "results" / "runs" / "S1"
ANCHOR = ROOT / "experiments" / "v51" / "results" / "m4" / "anchor"  # round 54's saved answers
DATA = Path("/home/user/lmwork/data")


def verdicts(path: Path, split: str) -> dict:
    answers = [json.loads(line) for line in path.read_text().splitlines()]
    return {f"{split}:{i['suite']}:{i['name']}": i for i in evalsets.score(answers, split)["items"]}


def training_text(run: Path, work: Path) -> str:
    manifest = json.loads((run / "manifest.json").read_text())
    rebuilt = write_round_dataset(work, manifest["seed"], DATA / "tokenizer.json", replay_dir=DATA,
                                  options=manifest["options"], log=lambda *_: None)
    if rebuilt["files"] != manifest["files"]:
        raise SystemExit(f"{run}: the rebuilt dataset differs from the one trained on")
    tokens = np.fromfile(work / "sft_tokens.bin", dtype=np.uint16)
    return BPETokenizer.load(DATA / "tokenizer.json").decode(tokens.tolist())


def main(out: str) -> None:
    base = {**verdicts(ANCHOR / "dev.jsonl", "dev"), **verdicts(ANCHOR / "unseen.jsonl", "unseen")}
    report = {}
    for run in sorted(p for p in RUNS.iterdir() if (p / "result.json").is_file()):
        model = {**verdicts(run / "dev.jsonl", "dev"), **verdicts(run / "unseen.jsonl", "unseen")}
        with tempfile.TemporaryDirectory(dir="/dev/shm") as work:
            text = training_text(run, Path(work))
        gains = []
        for key, v in sorted(model.items()):
            if v["official"] and not v["contaminated"] and not base[key]["official"]:
                name = key.rsplit(":", 1)[1]
                gains.append({"item": key, "robust": v["robust"],
                              "name_in_training_tokens": len(re.findall(rf"\b{re.escape(name)}\b", text))})
        hygiene = json.loads((run / "hygiene.json").read_text())
        report[run.name] = {"gains": gains, "hygiene_removed": len(hygiene),
                            "weak_passes": sorted(k for k, v in model.items() if v["official"] and not v["robust"]),
                            "training_tokens_checked": len(text)}
        print(run.name, json.dumps(report[run.name], indent=1))
    Path(out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
