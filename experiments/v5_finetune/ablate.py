"""Ablation runs: does each proposed change actually help on held-out prompts?

Every arm starts from the same checkpoint, trains the same number of steps on data built with the
same seed (unless the arm varies the seed), and is evaluated with the same prompts, scoring and
decoding (dev set for decisions, benchmark reported). Each arm changes ONE thing. Two baseline arms
with different seeds give the noise floor: a change counts only if it beats the baseline by more
than the seed-to-seed difference, without losing points in another category.

usage: ablate.py SPEC.json     (resumable: finished arms are skipped)
Arm weights are deleted after evaluation (disk); results go to <out>/<arm>/.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

W = Path("/home/user/lmwork")
REPO = Path("/home/user/Code-agent")
sys.path.insert(0, str(REPO))


def log(out: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (out / "ablation.log").open("a") as f:
        f.write(line + "\n")


def run(cmd, logfile: Path) -> int:
    with logfile.open("a") as f:
        return subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode


BUILD = r'''
import sys
sys.path.insert(0, "/home/user/Code-agent")
from pathlib import Path
import ycode.lm.compose as compose
from ycode.lm.basics import basics_examples, write_basics_dataset
seed, out, skills = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == "1"
compose.INCLUDE_SKILLS = skills
examples = (basics_examples(seed, write_per_concept=6, bugfix_per_concept=4)
            + compose.compose_examples(seed, per_exercise=3) + compose.compose_bugfix_examples(seed, per_exercise=1)
            + compose.contrast_examples(seed) + compose.novel_examples(seed))
write_basics_dataset(out, "/home/user/lmwork/data/tokenizer.json", replay_dir="/home/user/lmwork/data", replay=200,
                     seed=seed, examples=examples, log=print)
'''


def arm_done(d: Path) -> bool:
    return (d / "result.json").is_file()


def run_arm(spec: dict, arm: dict, out: Path) -> dict:
    d = out / arm["name"]
    d.mkdir(parents=True, exist_ok=True)
    data, cur = d / "data", out / "cur"
    log(out, f"{arm['name']}: building data (seed {arm['seed']}, skills {arm.get('skills', True)})")
    shutil.rmtree(data, ignore_errors=True)
    if run([sys.executable, "-c", BUILD, str(arm["seed"]), str(data), "1" if arm.get("skills", True) else "0"],
           d / "build.log") != 0:
        raise RuntimeError(f"{arm['name']}: data build failed")
    shutil.rmtree(cur, ignore_errors=True)
    log(out, f"{arm['name']}: training {spec['steps']} steps from {spec['anchor']}")
    t0 = time.time()
    code = run([sys.executable, "-m", "ycode.lm.cli", "sft", "--data", str(data), "--init-from", spec["anchor"],
                "--out", str(cur), "--steps", str(spec["steps"]), "--warmup", "2", "--batch-size", "4",
                "--optimizer", "lion", "--lr", str(arm.get("lr", 1.5e-5)), "--train-layers", "24",
                "--grad-checkpoint", "--eval-interval", "100000", "--eval-iters", "0", "--seed", str(arm["seed"]),
                "--init-optim", "--device", "cpu", "--precision", spec.get("precision", "bf16")], d / "train.log")
    if code != 0:
        raise RuntimeError(f"{arm['name']}: training failed (exit {code})")
    train_minutes = (time.time() - t0) / 60
    if arm.get("average"):
        out_avg = subprocess.run([sys.executable, str(W / "average.py"), spec["anchor"], str(cur), str(arm["average"])],
                                 capture_output=True, text=True)
        log(out, f"{arm['name']}: {out_avg.stdout.strip() or out_avg.stderr[-200:]}")
    result = {"arm": arm, "train_minutes": round(train_minutes, 1)}
    for which, problems in (("dev", "dev"), ("bench", "both")):
        answers = d / f"{which}.jsonl"
        run([sys.executable, "-u", "-m", "ycode.lm.cli", "eval", "--model", str(cur), "--problems", problems,
             "--device", "cpu", "--answers-out", str(answers)], d / f"{which}.log")
        run([sys.executable, "-m", "ycode.lm.cli", "analyze", str(answers), "--set", which,
             "--json", str(d / f"{which}.analysis.json")], d / f"{which}.analysis.txt")
        a = json.loads((d / f"{which}.analysis.json").read_text())
        result[which] = {"points": a["points"], "of": a["of"], "groups": a["groups"],
                         "lost_by_category": a["lost_by_category"],
                         "passed": sorted(f"{i['suite']}:{i['name']}" for i in a["items"] if i["pass"])}
    (d / "result.json").write_text(json.dumps(result, indent=1))
    shutil.rmtree(cur, ignore_errors=True)
    shutil.rmtree(data, ignore_errors=True)
    log(out, f"{arm['name']}: dev {result['dev']['points']}/{result['dev']['of']}, "
             f"bench {result['bench']['points']}/{result['bench']['of']}")
    return result


def summarize(spec: dict, out: Path) -> str:
    results = {a["name"]: json.loads((out / a["name"] / "result.json").read_text())
               for a in spec["arms"] if arm_done(out / a["name"])}
    base = results.get(spec["baseline"])
    lines = [f"Ablation {spec['name']} (anchor {spec['anchor']}, {spec['steps']} steps per arm)", ""]
    header = f"{'arm':14} {'dev':>6} {'bench':>6}  dev groups / bench groups"
    lines.append(header)
    for name, r in results.items():
        dg = " ".join(f"{k} {v['passed']}/{v['total']}" for k, v in r["dev"]["groups"].items())
        bg = " ".join(f"{k} {v['passed']}/{v['total']}" for k, v in r["bench"]["groups"].items())
        lines.append(f"{name:14} {r['dev']['points']:>6} {r['bench']['points']:>6}  {dg} | {bg}")
    if base:
        lines.append("")
        noise = [r for n, r in results.items() if n != spec["baseline"] and n.startswith("base")]
        floor = max((abs(r["dev"]["points"] - base["dev"]["points"]) for r in noise), default=None)
        lines.append(f"noise floor (dev, baseline seed vs seed): {floor}")
        for name, r in results.items():
            if name == spec["baseline"] or name.startswith("base"):
                continue
            d_dev = r["dev"]["points"] - base["dev"]["points"]
            worse = {c: r["dev"]["lost_by_category"][c] - base["dev"]["lost_by_category"][c]
                     for c in r["dev"]["lost_by_category"]
                     if r["dev"]["lost_by_category"][c] > base["dev"]["lost_by_category"][c]}
            keep = floor is not None and d_dev > floor and not any(v >= 2 for v in worse.values())
            gained = sorted(set(r["dev"]["passed"]) - set(base["dev"]["passed"]))
            lost = sorted(set(base["dev"]["passed"]) - set(r["dev"]["passed"]))
            lines.append(f"{name}: dev {d_dev:+d} vs baseline; categories losing more points: {worse or 'none'}; "
                         f"{'KEEP' if keep else 'do not keep'}")
            lines.append(f"   dev gained {gained}")
            lines.append(f"   dev lost   {lost}")
    return "\n".join(lines)


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text())
    out = W / "ablations" / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    for arm in spec["arms"]:
        if not arm_done(out / arm["name"]):
            run_arm(spec, arm, out)
    report = summarize(spec, out)
    (out / "summary.txt").write_text(report)
    log(out, "DONE\n" + report)


if __name__ == "__main__":
    main()
