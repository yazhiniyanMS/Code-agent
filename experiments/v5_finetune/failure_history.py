"""Backfill the failure store from every evaluated round and report which errors cost the most points.

Reads /home/user/lmwork/loop/{bench,dev}{N}.jsonl (the answers of every round), runs the error
analysis on each, appends one record per failed item to /home/user/lmwork/analysis/failures.jsonl,
and writes per-round category totals (rounds.json) and an aggregate report (report.md).

Note: the "body of a training function" check uses today's training generators, so for early
rounds a few copies of functions added later (skill families, whole-list maps) may be labelled
differently than they would have been at the time.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/user/Code-agent")
from ycode.lm.cli import EVAL_SETTINGS  # noqa: E402
from ycode.lm.erroranalysis import CATEGORIES, analyze, failure_rows, load_answers, lost_points_table  # noqa: E402

L = Path("/home/user/lmwork/loop")
OUT = Path("/home/user/lmwork/analysis")


def main() -> None:
    if "--render" in sys.argv:  # re-render the report from an existing store
        rounds = json.loads((OUT / "rounds.json").read_text())
        failures = [json.loads(x) for x in (OUT / "failures.jsonl").read_text().splitlines()]
        (OUT / "report.md").write_text(render(rounds, failures))
        print((OUT / "report.md").read_text())
        return
    OUT.mkdir(parents=True, exist_ok=True)
    store = OUT / "failures.jsonl"
    store.write_text("")
    rounds = []
    files = sorted(L.glob("*.jsonl"), key=lambda p: (int(re.sub(r"\D", "", p.stem) or 0), p.stem))
    for path in files:
        m = re.fullmatch(r"(bench|dev)(\d+)", path.stem)
        if not m:
            continue
        which, r = m[1], int(m[2])
        answers = load_answers(path)
        if not answers:
            continue
        result = analyze(answers, which)
        meta = {"run": f"round{r}-{which}", "round": r, "answers_file": str(path),
                "model": f"round {r} checkpoint", "model_step": answers[0].get("model_step"),
                "model_sha256": None, "settings": answers[0].get("settings", EVAL_SETTINGS)}
        rows = failure_rows(result, meta)
        with store.open("a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        rounds.append({"round": r, "set": which, "points": result["points"], "of": result["of"],
                       "lost_by_category": result["lost_by_category"],
                       "groups": {g: v["passed"] for g, v in result["groups"].items()}})
        print(f"round {r} {which}: {result['points']}/{result['of']} "
              f"{ {k: v for k, v in result['lost_by_category'].items() if v} }", flush=True)
    (OUT / "rounds.json").write_text(json.dumps(rounds, indent=1))
    failures = [json.loads(x) for x in store.read_text().splitlines()]
    report = render(rounds, failures)
    (OUT / "report.md").write_text(report)
    print(report)


def render(rounds: list[dict], failures: list[dict]) -> str:
    out = ["# Where YCode-LM v5 loses points", ""]
    for which, label in (("bench", "Benchmark (65 items; never trained on, never used to choose)"),
                         ("dev", "Dev set (62 items; held out of training, used to accept rounds)")):
        rs = [r for r in rounds if r["set"] == which]
        fs = [f for f in failures if f["set"] == which]
        if not rs:
            continue
        recent = [r for r in rs if r["round"] >= max(x["round"] for x in rs) - 9]
        recent_rounds = {r["round"] for r in recent}
        out += [f"## {label}", "",
                f"{len(rs)} evaluated rounds ({rs[0]['round']}-{rs[-1]['round']}), {len(fs)} failed answers in all.", "",
                f"| Category | Points lost, all rounds | Share | Per evaluation, rounds {min(recent_rounds)}-"
                f"{max(recent_rounds)} ({len(recent)} evaluations) |", "| --- | --- | --- | --- |"]
        total = Counter(f["category"] for f in fs)
        last = Counter(f["category"] for f in fs if f["round"] in recent_rounds)
        for c in sorted(CATEGORIES, key=lambda c: -total[c]):
            out.append(f"| {c} | {total[c]} | {total[c] / max(1, len(fs)):.0%} | {last[c] / max(1, len(recent)):.1f} |")
        table = lost_points_table([f for f in fs if f["round"] in recent_rounds])
        out += ["", f"Most-failed items in those {len(recent)} evaluations (failures by category; latest failure):", ""]
        latest = {}
        for f in fs:
            latest[(f["group"], f["name"])] = f
        for item in table["items"][:15]:
            last_failure = latest[(item["group"], item["name"])]
            counts = ", ".join(f"{c} {n}" for c, n in item["categories"].items())
            out.append(f"- `{item['name']}` ({item['group']}): failed {item['failures']}/{len(recent)} rounds "
                       f"({counts}); latest: {last_failure['category']}, {last_failure['reason']}")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    main()
