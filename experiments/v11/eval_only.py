"""V11: evaluate a checkpoint that was not trained by the runner (e.g. an interpolation) exactly as the runner
evaluates a trained one, and write a result.json in the runner's format.

usage: eval_only.py SPEC MODEL_DIR RUN_DIR --arm ARM --seed SEED --init-from PATH [--note TEXT] [--build-json F]

* The evaluation section of SPEC (dev, unseen-v1, bench, task sets) and the very commands of
  ycode.lm.experiment.run_one: `ycode.lm.cli eval` for dev / unseen / bench, experiments/v7/prompt_diagnostic.py
  for the task sets, scored with ycode.lm.experiment.scores and set_scores. unseen-v2 is never evaluated.
* Resumable after a container restart: each finished evaluation leaves RUN_DIR/.done_<name>, and a rerun
  skips it (its answers file is kept).
* --build-json: a JSON record of how MODEL_DIR was made (sources, weights, minutes, peak RAM), stored under
  "training" so the comparison tables have its cost and identity.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from ycode.lm.experiment import EVAL_DEFAULTS, run_measured, scores, set_scores
    from ycode.lm.train import assert_finite_weights, file_sha256, load_checkpoint

    ap = argparse.ArgumentParser()
    ap.add_argument("spec", type=Path)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--init-from", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--build-json", type=Path, default=None)
    args = ap.parse_args()
    spec = json.loads(args.spec.read_text())
    evaluation = {**EVAL_DEFAULTS, **spec.get("eval", {})}
    assert not evaluation.get("unseen2"), "unseen-v2 is never evaluated in V11"
    run, cur = args.run_dir, args.model_dir
    run.mkdir(parents=True, exist_ok=True)
    say = lambda m: print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {args.arm} {args.seed}: {m}", flush=True)

    import torch

    model, _, _ = load_checkpoint(cur, dtype=torch.bfloat16)
    assert_finite_weights(model)
    del model
    build = json.loads(args.build_json.read_text()) if args.build_json else {}
    sha = file_sha256(cur / "model.pt")
    record = {"arm": args.arm, "seed": args.seed, "smoke": False, "note": args.note, "eval": evaluation,
              "anchor": spec["anchor"], "anchor_sha256": spec.get("anchor_sha256"), "init_from": args.init_from,
              "training": {"steps": 0, "tokens": 0, "minutes": build.get("minutes", 0.0),
                           "peak_rss_mb": build.get("peak_rss_mb", 0), "checkpoint_sha256": sha, "build": build},
              "kept_model": {"path": str(cur / "model.pt"), "sha256": sha}}
    for which, problems in (("dev", "dev"), ("unseen", "unseen"), ("bench", "both")):
        if not evaluation.get(which):
            continue
        answers, done = run / f"{which}.jsonl", run / f".done_{which}"
        if not done.is_file():
            say(f"evaluating {which}")
            answers.unlink(missing_ok=True)
            cmd = [sys.executable, "-u", "-m", "ycode.lm.cli", "eval", "--model", str(cur), "--problems", problems,
                   "--device", "cpu", "--answers-out", str(answers)]
            r = run_measured(cmd, run / f"{which}.log")
            if r["exit"] != 0:
                raise RuntimeError(f"{which} evaluation failed; see {run / f'{which}.log'}")
            done.write_text(json.dumps(r))
        record[f"{which}_run"] = json.loads(done.read_text())
        record[which] = scores(answers, which)
    for ts in evaluation.get("sets") or []:
        rows, done = run / f"set_{ts['name']}.jsonl", run / f".done_set_{ts['name']}"
        if not done.is_file():
            say(f"task set {ts['name']}")
            rows.unlink(missing_ok=True)
            cmd = [sys.executable, str(ROOT / "experiments" / "v7" / "prompt_diagnostic.py"), str(cur), str(rows),
                   "--set", str(ROOT / ts["set"]), "--variants", ts.get("variants", "original")]
            r = run_measured(cmd, run / f"set_{ts['name']}.log")
            if r["exit"] != 0:
                raise RuntimeError(f"task set {ts['name']} failed; see {run / ('set_' + ts['name'] + '.log')}")
            done.write_text(json.dumps(r))
        record.setdefault("sets", {})[ts["name"]] = {"run": json.loads(done.read_text()), **set_scores(rows)}
    record["minutes_total"] = round(record["training"]["minutes"]
                                    + sum(record.get(f"{w}_run", {}).get("minutes", 0) for w in ("dev", "unseen", "bench"))
                                    + sum(v["run"].get("minutes", 0) for v in record.get("sets", {}).values()), 1)
    (run / "result.json").write_text(json.dumps(record, indent=1))
    d = record.get("dev", {})
    say(f"dev clean {d.get('clean')}/{d.get('clean_of')}, unseen clean {record.get('unseen', {}).get('clean')}, "
        f"bench {record.get('bench', {}).get('points')}; wrote {run / 'result.json'}")


if __name__ == "__main__":
    main()
