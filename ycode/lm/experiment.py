"""Controlled fine-tuning experiments: arms x seeds from one checkpoint, scored identically.

A spec (JSON) names the starting checkpoint, one training budget, the evaluation, the seeds and
the arms. Each arm changes only its data options (``ycode.lm.rounddata.DEFAULTS``) or, if it says
so, a training setting. Every run gets the same checkpoint, steps, learning rate, batch, layers and
decoding; only the arm's own change and the seed differ. Each run records:

* data manifest and hygiene report (what was removed and why), training metrics and record
  (config, checkpoint hashes, peak memory, minutes), evaluation runtime and peak memory;
* dev and benchmark scores: official (the unchanged harness), clean (without items the starting
  checkpoint saw in equivalent form, ``leakage.CONTAMINATED_ITEMS``) and robust (passes that also
  agree with the reference solution on probe inputs: the items' own tests are short);
* points lost per error category (``erroranalysis``), hallucinations, invalid outputs;
* answer loss on training rows, held-out rows and unseen tasks (``diagnose``).

``summarize`` reports mean and standard deviation per arm and compares arms seed by seed. A
change is **retained** only if its mean paired gain on the decision metric clears a one-sided 95%
t-test over seeds and no error category loses 2 or more points on average; **rejected** if it
loses by the same margin or damages a category; otherwise **inconclusive**. One lucky seed cannot
retain anything.

usage: python -m ycode.lm.experiment SPEC.json [--smoke] [--summarize]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # subprocesses run this checkout's ycode, wherever we start
# One-sided 95% critical values of Student's t for n - 1 degrees of freedom.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812}
TRAIN_DEFAULTS = {"steps": 72, "lr": 1.5e-5, "batch": 4, "train_layers": 24, "warmup": 2, "precision": "bf16",
                  "init_optim": True, "seq_len": None, "average": None}
EVAL_DEFAULTS = {"dev": True, "unseen": True, "unseen2": False, "bench": True, "limit": None, "diagnose": 32}
DISK_NEEDED_GB = 8
SMOKE = {"train": {"steps": 2}, "eval": {"limit": 4, "diagnose": 4}}


def run_measured(cmd: list[str], log_path: Path, cwd: Path | None = None) -> dict:
    """Run a command, logging its output; return exit code, minutes and peak resident memory (MB)."""
    t0 = time.time()
    peak = [0]
    with log_path.open("a") as log:
        env = {**os.environ, "PYTHONPATH": str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        proc = subprocess.Popen(cmd, cwd=cwd or REPO, stdout=log, stderr=subprocess.STDOUT, env=env)

        def watch() -> None:
            status = Path(f"/proc/{proc.pid}/status")
            while proc.poll() is None:
                try:
                    for line in status.read_text().splitlines():
                        if line.startswith("VmHWM:"):
                            peak[0] = max(peak[0], int(line.split()[1]) // 1024)
                except OSError:
                    pass
                time.sleep(2)

        thread = threading.Thread(target=watch, daemon=True)
        thread.start()
        code = proc.wait()
        thread.join(timeout=5)
    return {"exit": code, "minutes": round((time.time() - t0) / 60, 2), "peak_rss_mb": peak[0]}


def scores(answers_path: Path, which: str, *, classify: bool = True) -> dict:
    """Official, clean and robust scores (ycode.lm.evalsets, explicit denominators), the v5 error
    categories, and the v5.1 failure classes (ycode.lm.failures) of one answers file."""
    from ycode.lm.erroranalysis import analyze, load_answers
    from ycode.lm.evalsets import score

    answers = load_answers(answers_path)
    result = analyze(answers, which)
    s = score(answers, which)
    out = {
        "points": s["official"], "of": s["official_of"], "clean": s["clean"], "clean_of": s["clean_of"],
        "robust": s["robust"], "robust_of": s["robust_of"], "robust_clean": s["robust_clean"],
        "weak_passes": s["weak_passes"], "fingerprint": s["fingerprint"], "eval_version": s["eval_version"],
        "groups": {g: v["passed"] for g, v in result["groups"].items()},
        "lost_by_category": result["lost_by_category"],
        "hallucinations": result["lost_by_category"].get("hallucination", 0),
        "invalid_outputs": result["lost_by_category"].get("formatting", 0),
        "passed": sorted(f"{i['suite']}:{i['name']}" for i in result["items"] if i["pass"]),
        # v5 error category of every failed item (for the items-lost category rule)
        "categories": {f"{i['suite']}:{i['name']}": i.get("category") for i in result["items"] if not i["pass"]},
        "contaminated": sorted(f"{i['suite']}:{i['name']}" for i in s["items"] if i["contaminated"]),
    }
    if classify and which in ("dev", "unseen", "unseen2"):
        from ycode.lm.failures import classify_answers

        failures = classify_answers(answers, which)
        out["failure_classes"] = dict(failures["ranked"])
        out["failure_classes_clean"] = dict(failures["ranked_clean"])
        out["failures"] = [{k: f[k] for k in ("suite", "name", "contaminated", "class", "evidence")}
                           for f in failures["items"]]
    return out


def _metrics_tail(path: Path, n: int = 10) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []
    if not rows:
        return {}
    last = rows[-n:]
    return {"steps": rows[-1]["step"], "loss_last10_mean": round(sum(r["loss"] for r in last) / len(last), 4),
            "loss_first": rows[0]["loss"], "grad_norm_last10_mean": round(
                sum(r["grad_norm"] or 0 for r in last) / len(last), 4), "tok_per_s": rows[-1]["tok_per_s"],
            "answer_tokens": rows[-1]["answer_tokens"]}


def run_one(spec: dict, arm: dict, seed: int, out: Path, *, smoke: bool = False, log=print) -> dict:
    from ycode.lm.rounddata import write_round_dataset
    from ycode.lm.train import assert_finite_weights, load_checkpoint, verify_checkpoint

    run_dir = out / arm["name"] / str(seed)
    result_path = run_dir / "result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    train = {**TRAIN_DEFAULTS, **spec.get("train", {}), **arm.get("train", {}), **(SMOKE["train"] if smoke else {})}
    evaluation = {**EVAL_DEFAULTS, **spec.get("eval", {}), **(SMOKE["eval"] if smoke else {})}
    record: dict = {"arm": arm["name"], "seed": seed, "smoke": smoke, "data_options": arm.get("data", {}),
                    "train": train, "eval": evaluation, "anchor": spec["anchor"], "anchor_sha256": spec["anchor_sha256"]}

    data, cur = run_dir / "data", run_dir / "model"
    t0 = time.time()
    if (data / "manifest.json").is_file() and (cur / "model.pt").is_file():
        # an interrupted run (e.g. a container restart): its data is reused and training resumes
        log(f"{arm['name']} seed {seed}: resuming from the data and checkpoint already written")
        manifest = json.loads((data / "manifest.json").read_text())
    else:
        log(f"{arm['name']} seed {seed}: building data {arm.get('data', {})}")
        shutil.rmtree(data, ignore_errors=True)
        shutil.rmtree(cur, ignore_errors=True)
        manifest = write_round_dataset(data, seed, spec["tokenizer"], replay_dir=spec.get("replay_dir"),
                                       options=arm.get("data", {}), log=lambda m: log(f"  {m}"))
    record["data"] = {k: v for k, v in manifest.items() if k != "options"}
    record["data"]["minutes"] = round((time.time() - t0) / 60, 2)
    shutil.copy(data / "hygiene.json", run_dir / "hygiene.json")
    shutil.copy(data / "manifest.json", run_dir / "manifest.json")

    free_gb = shutil.disk_usage(run_dir).free / 1e9
    if free_gb < DISK_NEEDED_GB:  # a checkpoint is ~7 GB; failing at the first save wastes the whole run
        raise RuntimeError(f"only {free_gb:.1f} GB free under {run_dir}; training needs {DISK_NEEDED_GB} GB "
                           "(remove leftover checkpoints, e.g. of an interrupted smoke run)")
    log(f"{arm['name']} seed {seed}: training {train['steps']} steps")
    cmd = [sys.executable, "-m", "ycode.lm.cli", "sft", "--data", str(data), "--init-from", spec["anchor"],
           "--out", str(cur), "--steps", str(train["steps"]), "--warmup", str(train["warmup"]),
           "--batch-size", str(train["batch"]), "--optimizer", "lion", "--lr", str(train["lr"]),
           "--train-layers", str(train["train_layers"]), "--grad-checkpoint", "--eval-interval", "100000",
           "--eval-iters", "0", "--seed", str(seed), "--device", "cpu", "--precision", train["precision"]]
    cmd += ["--init-optim"] if train["init_optim"] else []
    # checkpoint halfway (in place: one copy on disk) so a restart costs at most half a run
    cmd += ["--resume", "--save-interval", str(max(1, train["steps"] // 2)), "--save-in-place"]
    cmd += ["--seq-len", str(train["seq_len"])] if train["seq_len"] else []
    record["training"] = run_measured(cmd, run_dir / "train.log")
    if record["training"]["exit"] != 0:
        raise RuntimeError(f"{arm['name']} seed {seed}: training failed (exit {record['training']['exit']}); "
                           f"see {run_dir / 'train.log'}")
    if train["average"]:
        avg = subprocess.run([sys.executable, str(Path(spec["average_script"])), spec["anchor"], str(cur),
                              str(train["average"])], capture_output=True, text=True)
        if avg.returncode != 0:
            raise RuntimeError(f"averaging failed: {avg.stderr[-300:]}")
    record["training"].update(_metrics_tail(cur / "metrics.jsonl"))
    summary = json.loads((cur / "train_log.json").read_text())["summary"]
    record["training"].update({k: summary.get(k) for k in ("empty_batches", "checkpoint_sha256", "init_sha256",
                                                            "tokens", "answer_tokens")})
    shutil.copy(cur / "metrics.jsonl", run_dir / "metrics.jsonl")
    shutil.copy(cur / "train_log.json", run_dir / "train_log.json")
    if not train["average"]:
        verify_checkpoint(cur)  # the file evaluated is the file training wrote
    import torch

    model, _, _ = load_checkpoint(cur, dtype=torch.bfloat16)  # half the memory; NaN/Inf survive the cast
    assert_finite_weights(model)
    del model

    for which, problems in (("dev", "dev"), ("unseen", "unseen"), ("unseen2", "unseen2"), ("bench", "both")):
        if not evaluation.get(which):
            continue
        log(f"{arm['name']} seed {seed}: evaluating {which}")
        answers = run_dir / f"{which}.jsonl"
        cmd = [sys.executable, "-u", "-m", "ycode.lm.cli", "eval", "--model", str(cur), "--problems", problems,
               "--device", "cpu", "--answers-out", str(answers)]
        cmd += ["--limit", str(evaluation["limit"])] if evaluation["limit"] else []
        record[f"{which}_run"] = run_measured(cmd, run_dir / f"{which}.log")
        if record[f"{which}_run"]["exit"] != 0:
            raise RuntimeError(f"{which} evaluation failed; see {run_dir / f'{which}.log'}")
        record[which] = scores(answers, which)
    if evaluation["diagnose"]:
        log(f"{arm['name']} seed {seed}: answer loss on train / held-out / unseen")
        diag = run_dir / "diagnose.json"
        record["diagnose_run"] = run_measured([sys.executable, "-m", "ycode.lm.diagnose", str(cur), str(data),
                                               "--n", str(evaluation["diagnose"]), "--trained-on", "--no-all-dev",
                                               "--json", str(diag)],
                                              run_dir / "diagnose.log")
        if diag.is_file():
            record["diagnose"] = json.loads(diag.read_text())
    record["minutes_total"] = round(sum(record.get(k, {}).get("minutes", 0) for k in
                                        ("data", "training", "dev_run", "unseen_run", "unseen2_run", "bench_run",
                                         "diagnose_run")), 1)
    result_path.write_text(json.dumps(record, indent=1))
    shutil.rmtree(cur, ignore_errors=True)  # 7 GB per checkpoint: kept only while scoring
    shutil.rmtree(data, ignore_errors=True)
    dev = record.get("dev", {})
    log(f"{arm['name']} seed {seed}: dev {dev.get('points')}/{dev.get('of')} (clean {dev.get('clean')}/"
        f"{dev.get('clean_of')}), bench {record.get('bench', {}).get('points')}, {record['minutes_total']} min")
    return record


# ------------------------------------------------------------------ summary


def _mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else float("nan")
    return mean, sd


METRIC_SETS = {"unseen_clean": ("dev", "unseen"),  # v5.1 primary: clean dev (/44) + unseen-v1 (/53)
               "all_clean": ("dev", "unseen", "unseen2")}  # V6 primary: + unseen-v2 (/71), out of 168


def _metric(record: dict, metric: str) -> float | None:
    if metric in METRIC_SETS:
        if not all(record.get(w) for w in METRIC_SETS[metric]):
            return None
        return sum(record[w]["clean"] for w in METRIC_SETS[metric])
    which, _, kind = metric.partition("_")
    block = record.get(which)
    if not block:
        return None
    return block["points"] if not kind else block[kind]


ITEMS_LOST_LIMIT = 3  # mean clean items lost per category (items_lost rule); seed-to-seed churn is ~1 item


def compare(a_runs: dict[int, dict], b_runs: dict[int, dict], metric: str,
            category_rule: str = "failure_counts") -> dict:
    """Arm A against arm B, seed by seed, on ``metric`` (e.g. "dev_clean")."""
    seeds = sorted(set(a_runs) & set(b_runs))
    diffs = [_metric(a_runs[s], metric) - _metric(b_runs[s], metric) for s in seeds]
    n = len(diffs)
    mean, sd = _mean_sd(diffs)
    whiches = METRIC_SETS.get(metric, (metric.split("_")[0],))
    cats = a_runs[seeds[0]][whiches[0]]["lost_by_category"] if seeds else {}
    if category_rule == "items_lost":
        # per category: clean items the baseline (b) passes and the candidate (a) fails, by the
        # candidate's failure category; failures that only change type do not count
        def lost_items(a: dict, b: dict, c: str) -> int:
            count = 0
            for w in whiches:
                dirty = set(a[w].get("contaminated", []))
                count += sum(1 for k in set(b[w]["passed"]) - set(a[w]["passed"])
                             if k not in dirty and a[w]["categories"].get(k) == c)
            return count

        damage = {c: round(sum(lost_items(a_runs[s], b_runs[s], c) for s in seeds) / n, 2) for c in cats} if n else {}
        limit = ITEMS_LOST_LIMIT
    else:  # "failure_counts" (v5.1): change in failures per category, which also moves when a failure changes type
        def lost(run: dict, c: str) -> int:
            return sum(run[w]["lost_by_category"][c] for w in whiches)

        damage = {c: round(sum(lost(a_runs[s], c) - lost(b_runs[s], c) for s in seeds) / n, 2) for c in cats} if n else {}
        limit = 2
    if n < 2:
        verdict, margin = "inconclusive (fewer than 2 paired seeds)", None
    else:
        margin = T95.get(n - 1, 1.8) * (sd if sd > 0 else 0.0) / math.sqrt(n)
        if any(v >= limit for v in damage.values()) or mean < -margin or (sd == 0 and mean < 0):
            verdict = "rejected"
        elif mean > margin and mean > 0:
            verdict = "retained"
        else:
            verdict = "inconclusive"
    ci = None
    if n >= 2:
        half = T975.get(n - 1, 2.0) * sd / math.sqrt(n)
        ci = [round(mean - half, 2), round(mean + half, 2)]
    return {"seeds": seeds, "diffs": diffs, "mean": None if not n else round(mean, 2),
            "sd": None if n < 2 else round(sd, 2), "margin": None if margin is None else round(margin, 2), "ci95": ci,
            "category_change": damage, "category_rule": category_rule, "verdict": verdict}


def summarize(spec: dict, out: Path) -> dict:
    runs: dict[str, dict[int, dict]] = {}
    for arm in spec["arms"]:
        for seed in spec["seeds"]:
            path = out / arm["name"] / str(seed) / "result.json"
            if path.is_file():
                runs.setdefault(arm["name"], {})[seed] = json.loads(path.read_text())
    metric = spec.get("metric", "dev_clean")
    arms = {}
    for name, by_seed in runs.items():
        rs = list(by_seed.values())
        row = {"seeds": sorted(by_seed)}
        for m in ("unseen_clean", "all_clean", "dev", "dev_clean", "dev_robust", "unseen", "unseen_robust", "unseen2",
                  "bench", "bench_clean", "bench_robust"):
            vals = [v for r in rs if (v := _metric(r, m)) is not None]
            mean, sd = _mean_sd(vals)
            row[m] = {"values": vals, "mean": round(mean, 2), "sd": None if math.isnan(sd) else round(sd, 2)}
        for which in ("dev", "unseen", "unseen2", "bench"):
            cats = [r[which]["lost_by_category"] for r in rs if which in r]
            if cats:
                row[f"{which}_lost_by_category"] = {c: round(sum(x[c] for x in cats) / len(cats), 2) for c in cats[0]}
                row[f"{which}_hallucinations"] = [r[which]["hallucinations"] for r in rs if which in r]
                row[f"{which}_invalid_outputs"] = [r[which]["invalid_outputs"] for r in rs if which in r]
                classes = [r[which].get("failure_classes_clean") for r in rs if r.get(which, {}).get("failure_classes_clean") is not None]
                if classes:
                    keys = sorted({k for c in classes for k in c})
                    row[f"{which}_failure_classes_clean"] = {k: round(sum(c.get(k, 0) for c in classes) / len(classes), 2)
                                                             for k in keys}
        row["train_minutes"] = _mean_sd([r["training"]["minutes"] for r in rs])[0]
        row["total_minutes"] = _mean_sd([r["minutes_total"] for r in rs])[0]
        row["train_peak_rss_mb"] = max(r["training"]["peak_rss_mb"] for r in rs)
        row["eval_peak_rss_mb"] = max(max(r.get("dev_run", {}).get("peak_rss_mb", 0), r.get("bench_run", {}).get(
            "peak_rss_mb", 0)) for r in rs)
        row["removed_rows"] = [r["data"].get("removed", {}) for r in rs]
        diags = [r["diagnose"] for r in rs if "diagnose" in r]
        if diags:
            row["answer_loss"] = {split: round(sum(d[split]["token_loss"] for d in diags) / len(diags), 4)
                                  for split in ("train", "heldout_rows", "unseen_tasks", "unseen_dev_clean", "unseen_new")
                                  if all(split in d for d in diags)}
        row["train_tokens"] = [r["training"].get("tokens") for r in rs]
        arms[name] = row
    comparisons = [{"arm": a, "against": b, **compare(runs[a], runs[b], metric, spec.get("category_rule", "failure_counts"))}
                   for a, b in spec.get("comparisons", []) if a in runs and b in runs]
    return {"name": spec["name"], "metric": metric, "arms": arms, "comparisons": comparisons}


def render(summary: dict) -> str:
    primary = summary["metric"]
    lines = [f"# Experiment {summary['name']}", "",
             f"Decision metric: `{primary}`" + (" = clean dev (/44) + unseen-concept set (/53), out of 97: concepts the "
                                                "starting checkpoint never saw, in any form." if primary == "unseen_clean" else "."),
             "Benchmark scores are reported only. Mean ± standard deviation over seeds.", "",
             "| Arm | Seeds | Primary | Dev clean /44 | Unseen /53 | Unseen robust | Dev official /62 | Bench official /65 | "
             "Bench clean /61 | Bench robust /65 | Halluc. dev+unseen | Invalid dev+unseen | Train min | Peak RSS MB |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]

    def ms(cell):
        if cell is None or not cell["values"]:
            return "-"
        return f"{cell['mean']:.1f} ± {cell['sd']:.1f}" if cell["sd"] is not None else f"{cell['mean']:.1f}"

    for name, r in summary["arms"].items():
        hall = [a + b for a, b in zip(r.get("dev_hallucinations", []), r.get("unseen_hallucinations", []))]
        inval = [a + b for a, b in zip(r.get("dev_invalid_outputs", []), r.get("unseen_invalid_outputs", []))]
        lines.append(f"| {name} | {len(r['seeds'])} | {ms(r.get(primary))} | {ms(r.get('dev_clean'))} | {ms(r.get('unseen'))} | "
                     f"{ms(r.get('unseen_robust'))} | {ms(r.get('dev'))} | {ms(r.get('bench'))} | {ms(r.get('bench_clean'))} | "
                     f"{ms(r.get('bench_robust'))} | {hall} | {inval} | {r['train_minutes']:.0f} | {r['train_peak_rss_mb']} |")
    lines += ["", "| Arm | Answer loss: train / held-out (validation) / unseen dev clean / unseen new | "
              "Unseen-task failure classes (clean, mean per seed) |", "| --- | --- | --- |"]
    for name, r in summary["arms"].items():
        al = r.get("answer_loss", {})
        losses = " / ".join(str(al.get(k, "-")) for k in ("train", "heldout_rows", "unseen_dev_clean", "unseen_new"))
        classes = {k: v for k, v in (r.get("unseen_failure_classes_clean") or {}).items()}
        lines.append(f"| {name} | {losses} | {classes or '-'} |")
    lines += ["", "| Comparison | Paired diffs | Mean | 95% CI | One-sided 95% margin | Category change "
              "(+ = more points lost) | Verdict |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for c in summary["comparisons"]:
        changed = {k: v for k, v in c["category_change"].items() if v}
        lines.append(f"| {c['arm']} vs {c['against']} | {c['diffs']} | {c['mean']} | {c.get('ci95')} | {c['margin']} | "
                     f"{changed or 'none'} | **{c['verdict']}** |")
    return "\n".join(lines) + "\n"


def _anchor_hash(anchor: Path, out: Path, verify) -> str:
    """The starting checkpoint's SHA-256, cached next to the results (hashing 7 GB takes minutes);
    recomputed if the file's size or modification time changed."""
    stat = (anchor / "model.pt").stat()
    key = f"{anchor.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    cache = out / "anchor.sha256.json"
    if cache.is_file() and (cached := json.loads(cache.read_text())).get("key") == key:
        return cached["sha256"]
    digest = verify(anchor)
    cache.write_text(json.dumps({"key": key, "sha256": digest}))
    return digest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run or summarize a controlled fine-tuning experiment.")
    ap.add_argument("spec", type=Path)
    ap.add_argument("--smoke", action="store_true", help="2 training steps, 4 items per test set, first seed only")
    ap.add_argument("--summarize", action="store_true", help="only rebuild the summary from finished runs")
    ap.add_argument("--arms", help="comma-separated arm names to run (default: all)")
    args = ap.parse_args(argv)
    spec = json.loads(args.spec.read_text())
    out = Path(spec["out"]) / ("smoke" if args.smoke else "")
    out.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with (out / "experiment.log").open("a") as f:
            f.write(line + "\n")

    if not args.summarize:
        from ycode.lm.train import verify_checkpoint

        spec.setdefault("anchor_sha256", _anchor_hash(Path(spec["anchor"]), out, verify_checkpoint))
        wanted = set(args.arms.split(",")) if args.arms else None
        seeds = spec["seeds"][:1] if args.smoke else spec["seeds"]
        # seed-major order: every arm gets its first seed before any arm gets a second
        for seed in seeds:
            for arm in spec["arms"]:
                if wanted is None or arm["name"] in wanted:
                    run_one(spec, arm, seed, out, smoke=args.smoke, log=log)
    summary = summarize({**spec, "seeds": spec["seeds"][:1] if args.smoke else spec["seeds"]}, out)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "summary.md").write_text(render(summary))
    log("summary written to " + str(out / "summary.md"))
    print(render(summary))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
