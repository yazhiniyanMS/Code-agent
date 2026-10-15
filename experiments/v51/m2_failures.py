"""Phase 2: classify every unseen-task failure of a checkpoint (no training).

usage: m2_failures.py DEV_ANSWERS.jsonl UNSEEN_ANSWERS.jsonl OUT_DIR
Answers come from `ycode-lm eval --answers-out` (evaluation is deterministic, so saved answers of
a checkpoint are its answers). Writes OUT_DIR/m2_failures.json (every failure with prompt,
expected behaviour, generated answer, test outcome, task group, class, evidence) and
OUT_DIR/m2_failures.md (the classes ranked).
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ycode.lm.erroranalysis import load_answers  # noqa: E402
from ycode.lm.evalsets import score  # noqa: E402
from ycode.lm.failures import CLASSES, classify_answers  # noqa: E402


def main() -> None:
    dev_path, unseen_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for split, path in (("dev", dev_path), ("unseen", unseen_path)):
        answers = load_answers(path)
        report[split] = {"scores": {k: v for k, v in score(answers, split).items() if k != "items"},
                         **classify_answers(answers, split)}
    (out / "m2_failures.json").write_text(json.dumps(report, indent=1))
    clean = [f for split in ("dev", "unseen") for f in report[split]["items"] if not f["contaminated"]]
    ranked = Counter(f["class"] for f in clean)
    lines = ["# Unseen-task failures of the round-54 checkpoint", "",
             f"Clean dev: {report['dev']['scores']['clean']}/{report['dev']['scores']['clean_of']} passed; "
             f"unseen set: {report['unseen']['scores']['clean']}/{report['unseen']['scores']['clean_of']} passed "
             f"(robust {report['unseen']['scores']['robust']}/{report['unseen']['scores']['robust_of']}).", "",
             "| Class | Clean dev | Unseen | Total | Share |", "| --- | --- | --- | --- | --- |"]
    by_split = {s: Counter(f["class"] for f in report[s]["items"] if not f["contaminated"]) for s in ("dev", "unseen")}
    for cls, n in sorted(((c, ranked[c]) for c in CLASSES), key=lambda x: -x[1]):
        lines.append(f"| {cls} | {by_split['dev'][cls]} | {by_split['unseen'][cls]} | {n} | {n / max(1, len(clean)):.0%} |")
    lines += ["", "## Every clean failure", ""]
    for f in clean:
        lines.append(f"- `{f['split']}:{f['suite']}:{f['name']}`: **{f['class']}**: {f['evidence'][:160]}")
    (out / "m2_failures.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
