"""Phase 4: per-run and per-arm results of the control-vs-treatment experiment.

usage: m4_report.py SPEC.json [OUT.md]
Reads every finished run (<out>/<arm>/<seed>/result.json) and writes a markdown report with,
per arm and seed: official, clean and robust scores with denominators (dev, unseen, benchmark),
the primary metric, unseen-task loss and accuracy, training and validation loss, hallucinations
and invalid outputs, training tokens, wall-clock time and peak memory; then mean, standard
deviation, the paired comparison with its 95% interval and verdict, per-category changes, and
the regression rate against the starting checkpoint (items it passed that a run fails).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ycode.lm.erroranalysis import load_answers  # noqa: E402
from ycode.lm.evalsets import score  # noqa: E402
from ycode.lm.experiment import _mean_sd, summarize  # noqa: E402

# The starting checkpoint's answers. The unseen set was evaluated on this run's host; the dev and
# benchmark answers come from the AMX host (the same checkpoint and decoding; see the report).
ANCHOR = {"dev": Path("/home/user/lmwork/expA/run1_dev.jsonl"),
          "unseen": Path("/home/user/lmwork/v51/m2/unseen_anchor.jsonl"),
          "bench": Path("/home/user/lmwork/loop/bench54.jsonl")}


def anchor_passes() -> dict[str, set]:
    out = {}
    for split, path in ANCHOR.items():
        if path.is_file():
            s = score(load_answers(path), split)
            out[split] = {(i["suite"], i["name"]) for i in s["items"] if i["official"]}
    return out


def run_passes(run_dir: Path, split: str) -> tuple[set, set] | None:
    """(passed, answered) items of one run's answers file."""
    path = run_dir / f"{split}.jsonl"
    if not path.is_file():
        return None
    s = score(load_answers(path), split)
    return ({(i["suite"], i["name"]) for i in s["items"] if i["official"]},
            {(i["suite"], i["name"]) for i in s["items"] if i["answered"]})


def fmt(values: list[float], digits: int = 1) -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    mean, sd = _mean_sd(values)
    return f"{mean:.{digits}f}" + (f" ± {sd:.{digits}f}" if len(values) > 1 else "")


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text())
    out_md = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    root = Path(spec["out"])
    summary = summarize(spec, root)
    base = anchor_passes()
    rows = []
    for arm in spec["arms"]:
        for seed in spec["seeds"]:
            run_dir = root / arm["name"] / str(seed)
            path = run_dir / "result.json"
            if not path.is_file():
                continue
            r = json.loads(path.read_text())
            d = r.get("diagnose", {})
            regress = {}
            for split in ("dev", "unseen", "bench"):
                got = run_passes(run_dir, split)
                if got is not None and split in base:
                    now, answered = got
                    before = base[split] & answered  # only items this run was asked
                    if before:
                        regress[split] = (len(before - now), len(before))
            rows.append({
                "arm": arm["name"], "seed": seed,
                "primary": r["dev"]["clean"] + r["unseen"]["clean"],
                "dev": (r["dev"]["points"], r["dev"]["of"]), "dev_clean": (r["dev"]["clean"], r["dev"]["clean_of"]),
                "dev_robust": (r["dev"]["robust"], r["dev"]["robust_of"]),
                "unseen": (r["unseen"]["points"], r["unseen"]["of"]), "unseen_robust": (r["unseen"]["robust"], r["unseen"]["robust_of"]),
                "bench": (r["bench"]["points"], r["bench"]["of"]), "bench_clean": (r["bench"]["clean"], r["bench"]["clean_of"]),
                "bench_robust": (r["bench"]["robust"], r["bench"]["robust_of"]),
                "loss_unseen_dev": d.get("unseen_dev_clean", {}).get("token_loss"),
                "loss_unseen_new": d.get("unseen_new", {}).get("token_loss"),
                # code-only unseen loss with references in the arm's own answer format (fair across formats;
                # recorded from runs A1302/B1302 on: earlier runs lack it)
                "code_loss_unseen_new": d.get("unseen_new", {}).get(
                    "code_token_loss_spec_first" if arm.get("data", {}).get("spec_first") else "code_token_loss_plain"),
                "code_loss_unseen_dev": d.get("unseen_dev_clean", {}).get(
                    "code_token_loss_spec_first" if arm.get("data", {}).get("spec_first") else "code_token_loss_plain"),
                "loss_train_rows": d.get("train", {}).get("token_loss"), "loss_val_rows": d.get("heldout_rows", {}).get("token_loss"),
                "loss_train_log": r["training"].get("loss_last10_mean"),
                "hallucinations": sum(r[s]["hallucinations"] for s in ("dev", "unseen", "bench")),
                "invalid": sum(r[s]["invalid_outputs"] for s in ("dev", "unseen", "bench")),
                "tokens": r["training"].get("tokens"), "train_min": r["training"]["minutes"], "total_min": r["minutes_total"],
                "train_rss": r["training"]["peak_rss_mb"],
                "eval_rss": max(r.get(f"{s}_run", {}).get("peak_rss_mb", 0) for s in ("dev", "unseen", "bench")),
                "regress": regress, "classes": r["unseen"].get("failure_classes_clean", {}),
                "dev_classes": r["dev"].get("failure_classes_clean", {}),
            })
    of_primary = 44 + 53
    lines = [f"### Per run ({spec['name']})", "",
             f"Primary = clean dev + unseen, out of {of_primary}. Losses are answer-token cross-entropy.", "",
             "| Arm | Seed | Primary /97 | Dev /62 | Dev clean /44 | Dev robust /62 | Unseen /53 | Unseen robust /53 | "
             "Bench /65 | Bench clean /61 | Bench robust /65 | Loss unseen dev / new | Loss train rows / val rows | "
             "Code loss unseen dev / new (own format) | Train loss (last 10 steps) | Halluc. | Invalid | Train tokens | "
             "Train min | Total min | Peak MB train / eval |",
             "| " + " | ".join(["---"] * 21) + " |"]
    for x in rows:
        lines.append(
            f"| {x['arm']} | {x['seed']} | {x['primary']} | {x['dev'][0]} | {x['dev_clean'][0]} | {x['dev_robust'][0]} | "
            f"{x['unseen'][0]} | {x['unseen_robust'][0]} | {x['bench'][0]} | {x['bench_clean'][0]} | {x['bench_robust'][0]} | "
            f"{x['loss_unseen_dev']} / {x['loss_unseen_new']} | {x['loss_train_rows']} / {x['loss_val_rows']} | "
            f"{x['code_loss_unseen_dev']} / {x['code_loss_unseen_new']} | {x['loss_train_log']} | {x['hallucinations']} | {x['invalid']} | {x['tokens']} | {x['train_min']:.0f} | "
            f"{x['total_min']:.0f} | {x['train_rss']} / {x['eval_rss']} |")
    lines += ["", "### Per arm (mean ± standard deviation over seeds)", "",
              "| Arm | Seeds | Primary /97 | Dev clean /44 | Unseen /53 | Unseen robust | Bench /65 | Bench robust | "
              "Loss unseen new | Val loss | Halluc. | Invalid | Regression vs start (dev / unseen / bench) |",
              "| " + " | ".join(["---"] * 13) + " |"]
    for arm in spec["arms"]:
        xs = [x for x in rows if x["arm"] == arm["name"]]
        if not xs:
            continue
        reg = {s: [x["regress"][s][0] / x["regress"][s][1] for x in xs if s in x["regress"]] for s in ("dev", "unseen", "bench")}
        lines.append(
            f"| {arm['name']} | {len(xs)} | {fmt([x['primary'] for x in xs])} | {fmt([x['dev_clean'][0] for x in xs])} | "
            f"{fmt([x['unseen'][0] for x in xs])} | {fmt([x['unseen_robust'][0] for x in xs])} | {fmt([x['bench'][0] for x in xs])} | "
            f"{fmt([x['bench_robust'][0] for x in xs])} | {fmt([x['loss_unseen_new'] for x in xs], 3)} | "
            f"{fmt([x['loss_val_rows'] for x in xs], 3)} | {fmt([x['hallucinations'] for x in xs])} | "
            f"{fmt([x['invalid'] for x in xs])} | {fmt(reg['dev'], 2)} / {fmt(reg['unseen'], 2)} / {fmt(reg['bench'], 2)} |")
    lines += ["", "### Paired comparison", "",
              "| Comparison | Paired diffs (primary) | Mean | 95% CI | Category change (dev+unseen, + = more lost) | Verdict |",
              "| --- | --- | --- | --- | --- | --- |"]
    for c in summary["comparisons"]:
        changed = {k: v for k, v in c["category_change"].items() if v}
        lines.append(f"| {c['arm']} vs {c['against']} | {c['diffs']} | {c['mean']} | {c.get('ci95')} | {changed or 'none'} | "
                     f"**{c['verdict']}** |")
    lines += ["", "### Unseen-task failure classes (clean), per run", ""]
    for x in rows:
        lines.append(f"- {x['arm']} {x['seed']}: unseen {x['classes']}; dev {x['dev_classes']}")
    text = "\n".join(lines) + "\n"
    if out_md:
        out_md.write_text(text)
    (root / "m4_report.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=1, default=list))
    print(text)


if __name__ == "__main__":
    main()
