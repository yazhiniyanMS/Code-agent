"""Experiment A report: 3 identical evaluations of one checkpoint."""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/home/user/Code-agent-next")  # the error-analysis pipeline (evaluator itself is unchanged)
from ycode.lm.erroranalysis import analyze, load_answers

E = Path("/home/user/lmwork/expA")
manifest = json.loads((E / "manifest.json").read_text())
out = {"manifest": manifest, "sets": {}}
for which, tag in (("dev", "dev"), ("bench", "both")):
    runs = [load_answers(E / f"run{i}_{tag}.jsonl") for i in (1, 2, 3)]
    totals = [sum(r["pass"] for r in run) for run in runs]
    keyed = [{(r["suite"], r["name"]): r for r in run} for run in runs]
    same_verdict = sum(all(k[(s, n)]["pass"] == keyed[0][(s, n)]["pass"] for k in keyed) for s, n in keyed[0])
    same_text = sum(all(k[(s, n)]["answer"] == keyed[0][(s, n)]["answer"] for k in keyed) for s, n in keyed[0])
    analyses = [analyze(run, which) for run in runs]
    cats = [a["lost_by_category"] for a in analyses]
    groups = [{g: v["passed"] for g, v in a["groups"].items()} for a in analyses]
    out["sets"][which] = {
        "totals": totals, "of": len(runs[0]), "mean": statistics.mean(totals), "stdev": statistics.pstdev(totals),
        "per_question_identical_verdict": f"{same_verdict}/{len(keyed[0])}",
        "per_question_identical_answer_text": f"{same_text}/{len(keyed[0])}",
        "groups_per_run": groups, "lost_by_category_per_run": cats,
    }
    (E / f"analysis_{which}_run1.json").write_text(json.dumps(analyses[0], indent=1))
hist = json.loads(Path("/home/user/lmwork/loop/state2.json").read_text())["history"]
rej = [h for h in hist if not h.get("accepted", h.get("kept"))]
out["rejection_rate"] = {"rounds": len(hist), "rejected": len(rej), "rate": round(len(rej) / len(hist), 3),
                         "rejected_rounds": [h["round"] for h in rej]}
(E / "report.json").write_text(json.dumps(out, indent=1))
for which, s in out["sets"].items():
    print(f"{which}: totals {s['totals']} / {s['of']}  mean {s['mean']:.2f}  stdev {s['stdev']:.2f}  "
          f"identical verdicts {s['per_question_identical_verdict']}  identical answers {s['per_question_identical_answer_text']}")
    print(f"   groups run1: {s['groups_per_run'][0]}")
    print(f"   lost by category run1: {s['lost_by_category_per_run'][0]}")
print(f"rejection rate: {out['rejection_rate']['rejected']}/{out['rejection_rate']['rounds']} rounds "
      f"= {out['rejection_rate']['rate']:.0%}  (rounds {out['rejection_rate']['rejected_rounds']})")
