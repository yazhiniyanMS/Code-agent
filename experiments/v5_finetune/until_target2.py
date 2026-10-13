"""Keep fine-tuning YCode-LM v5 in rounds until the benchmark targets are met (recipe 2).

Changes from until_target.py (see the plateau diagnosis):
- Checkpoints are chosen on a held-out dev set (ycode.lm.devset), not on the benchmark. The
  benchmark (standard 30 + fresh 20 + 15 bug fixes) is run every round for reporting only.
- Each round continues from the latest accepted weights ("anchor"). A round is accepted unless
  its dev score drops (dev >= anchor_dev - 1 and dev >= max_dev - 3), so training is not thrown away
  on benchmark noise.
- No automatic layer ladder; train_layers, steps and precision live in state.json.
- Every answer is logged (loop/dev{r}.jsonl, loop/bench{r}.jsonl) so failures can be inspected.
The loop stops when the anchor's own benchmark meets every target. State lives in state2.json.
"""

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

W = Path("/home/user/lmwork")
L = W / "loop"
REPO = Path("/home/user/Code-agent")
STATE = L / "state2.json"
LOG = L / "loop.log"
# Raised on 2026-10-11 from 20/15/10 (45 of 65) to a total of 55 of 65, split by suite size.
TARGETS = {"standard": 25, "fresh": 17, "fixed": 13}
TOTAL_TARGET = 55

sys.path.insert(0, str(REPO))


def log(msg: str) -> None:
    with LOG.open("a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def run(cmd: list[str], out: Path) -> int:
    with out.open("a") as f:
        return subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode


def save_state(state: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE)


def last_row(path: Path) -> dict | None:
    lines = [line for line in path.read_text().splitlines() if line.startswith("[{")] if path.is_file() else []
    return json.loads(lines[-1])[0] if lines else None


def evaluate(model: Path, problems: str, tag: str) -> dict | None:
    """Run `ycode-lm eval` once (cached in loop/{tag}.log); returns the result row."""
    out = L / f"{tag}.log"
    row = last_row(out)
    if row is None:
        out.write_text("")
        run([sys.executable, "-u", "-m", "ycode.lm.cli", "eval", "--model", str(model), "--problems", problems,
             "--verbose", "--device", "cpu", "--answers-out", str(L / f"{tag}.jsonl")], out)
        row = last_row(out)
    return row


def bench_score(row: dict) -> dict:
    from ycode.lm.evaluate import PROBLEMS

    standard = {p.name for p in PROBLEMS}
    solved = set(row.get("solved", []))
    return {"standard": len(solved & standard), "fresh": len(solved - standard), "fixed": len(row.get("fixed", []))}


def dev_score(row: dict) -> int:
    return len(row.get("solved", [])) + len(row.get("fixed", []))


def met(s: dict | None) -> bool:
    return (s is not None and all(s[k] >= v for k, v in TARGETS.items())
            and sum(s[k] for k in TARGETS) >= TOTAL_TARGET)


def build_data(r: int, out: Path) -> int:
    """In a fresh process (build_round.py), so generator edits apply without restarting the driver."""
    return run([sys.executable, str(W / "build_round.py"), str(r), str(out)], L / "build.log")


def wait_if_paused() -> None:
    """`touch loop/PAUSE` holds the loop at the next round boundary (for code changes)."""
    if (L / "PAUSE").exists():
        log("paused (loop/PAUSE exists)")
        while (L / "PAUSE").exists():
            time.sleep(30)
        log("resumed")


def choose_precision(state: dict) -> str:
    """bf16 only on CPUs with AMX/AVX512-BF16 hardware *and* when measured faster; cached per CPU.
    Containers have moved between hosts with and without bf16 units."""
    flags = Path("/proc/cpuinfo").read_text()
    if "amx_bf16" not in flags and "avx512_bf16" not in flags:
        return "fp32"
    model = next((line.split(":", 1)[1].strip() for line in flags.splitlines() if line.startswith("model name")), "?")
    key = f"{model} amx={'amx_bf16' in flags}"
    cache_path = L / "precision.json"
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    if key not in cache:
        out = subprocess.run([sys.executable, str(W / "precbench.py"), "--quick"], capture_output=True,
                             text=True).stdout
        m = re.search(r"fp32 ([\d.]+)s\s+bf16 ([\d.]+)s", out)
        choice = "bf16" if m and float(m.group(2)) <= 0.77 * float(m.group(1)) else "fp32"
        cache[key] = {"choice": choice, "bench": out.strip()}
        cache_path.write_text(json.dumps(cache, indent=2))
        log(f"precision benchmark on {key}: {out.strip()} -> {choice}")
    return cache[key]["choice"]


def policy() -> dict:
    """loop/policy.json, re-read every round: acceptance tolerances and steps per round."""
    path = L / "policy.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def analyze_round(r: int, which: str, tag: str, anchor_tag: str | None) -> str:
    """Error analysis of this round's answers (categories, expected vs actual), compared item by item with
    the anchor's answers on the same prompts. Writes loop/{tag}.analysis.json; returns a one-line summary."""
    args = [sys.executable, "-m", "ycode.lm.cli", "analyze", str(L / f"{tag}.jsonl"), "--set", which,
            "--json", str(L / f"{tag}.analysis.json")]
    if anchor_tag and (L / f"{anchor_tag}.jsonl").is_file():
        args += ["--compare", str(L / f"{anchor_tag}.jsonl")]
    run(args, L / f"{tag}.analysis.txt")
    try:
        a = json.loads((L / f"{tag}.analysis.json").read_text())
    except (OSError, ValueError):
        return "analysis failed"
    lost = ", ".join(f"{c} {n}" for c, n in sorted(a["lost_by_category"].items(), key=lambda kv: -kv[1]) if n)
    cmp = a.get("comparison")
    diff = f"; vs anchor +{len(cmp['gained'])} -{len(cmp['lost'])}" if cmp else ""
    return f"{which} {a['points']}/{a['of']} lost: {lost}{diff}"


def round_record(r: int, state: dict) -> None:
    """What is needed to reproduce this round (written before training)."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    info = json.loads((Path(state["anchor"]) / "info.json").read_text())
    rec = {"round": r, "anchor": state["anchor"], "anchor_round": state.get("anchor_round"),
           "anchor_step": info.get("step"), "anchor_model_bytes": (Path(state["anchor"]) / "model.pt").stat().st_size,
           "data_seed": 1000 + r, "code_commit": commit, "build_mix": (W / "build_round.py").read_text(),
           "steps": state["steps"], "lr": state.get("lr", 1.5e-5), "train_layers": state["train_layers"],
           "precision": state["precision"], "policy": policy(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (L / f"round{r}.json").write_text(json.dumps(rec, indent=1))


def dev_parts(r: int) -> dict:
    """Dev sub-scores: devB = held-out basics concepts (new ideas), devC = held-out compose slots
    (one word from training), dfix = bug fixes, dcopied = fixes that handed the bug back."""
    from ycode.lm.devset import DEV_CONCEPT_NAMES

    rows = [json.loads(line) for line in (L / f"dev{r}.jsonl").read_text().splitlines()]
    writes = [x for x in rows if x["suite"] == "write"]
    fixes = [x for x in rows if x["suite"] == "bugfix"]
    return {"devB": sum(x["pass"] for x in writes if x["name"] in DEV_CONCEPT_NAMES),
            "devC": sum(x["pass"] for x in writes if x["name"] not in DEV_CONCEPT_NAMES),
            "dfix": sum(x["pass"] for x in fixes), "dcopied": sum(x.get("copied", False) for x in fixes)}


def main() -> None:
    state = json.loads(STATE.read_text())
    state["precision"] = choose_precision(state)
    save_state(state)
    while not met(state.get("anchor_bench")):
        r = state["round"] + 1
        data, cur = L / f"data{r}", L / "cur"
        if state["stage"] == "data":
            wait_if_paused()
            log(f"round {r}: building data (seed {1000 + r})")
            shutil.rmtree(data, ignore_errors=True)
            if build_data(r, data) != 0:
                log(f"round {r}: data build failed (see loop/build.log); stopping")
                return
            state["stage"] = "train"
            save_state(state)
        if state["stage"] == "train":
            state["steps"] = int(policy().get("steps", state["steps"]))
            round_record(r, state)
            log(f"round {r}: training {state['steps']} steps from {Path(state['anchor']).name} "
                f"(top {state['train_layers']} blocks, {state['precision']})")
            for attempt in range(3):  # a killed run (e.g. out of memory, exit -9) is retried
                code = run([sys.executable, "-m", "ycode.lm.cli", "sft", "--data", str(data),
                            "--init-from", state["anchor"], "--resume", "--out", str(cur),
                            "--steps", str(state["steps"]), "--warmup", "2", "--batch-size", "4",
                            "--optimizer", "lion", "--lr", str(state.get("lr", 1.5e-5)),
                            "--train-layers", str(state["train_layers"]), "--grad-checkpoint",
                            "--eval-interval", "100000", "--eval-iters", "0", "--seed", str(1000 + r),
                            # Continue the anchor's Lion momentum; checkpoint mid-round in place so a
                            # crash costs half a round and the save needs no second copy on disk.
                            "--init-optim", "--save-interval", str(state["steps"] // 2), "--save-in-place",
                            "--device", "cpu", "--precision", state["precision"]], L / "train.log")
                if code >= 0:
                    break
                log(f"round {r}: training was killed (signal {-code}); retrying ({attempt + 1}/3)")
                time.sleep(60)
            if code != 0:
                log(f"round {r}: training failed (exit {code}); stopping")
                return
            weight = policy().get("average_weight")
            if weight:  # smooth round-to-round noise: blend the new weights with the anchor's
                out = subprocess.run([sys.executable, str(W / "average.py"), state["anchor"], str(cur), str(weight)],
                                     capture_output=True, text=True)
                log(f"round {r}: {out.stdout.strip() or out.stderr.strip()[-300:]}")
                if out.returncode != 0:
                    log(f"round {r}: averaging failed; stopping")
                    return
            state["stage"] = "eval"
            save_state(state)
        if state["stage"] == "eval":
            wait_if_paused()
            dev = evaluate(cur, "dev", f"dev{r}")
            if dev is None:
                log(f"round {r}: evaluation failed; stopping")
                return
            d, parts = dev_score(dev), dev_parts(r)
            log(f"round {r}: " + analyze_round(r, "dev", f"dev{r}", f"dev{state.get('anchor_round')}"))
            a_parts = state.get("anchor_sub", parts)
            # Accept when the dev total holds up, or when new-concept + bug-fix items clearly improve
            # at a small cost in near-copy items (devC).
            tol_anchor, tol_max = policy().get("tol_anchor", 3), policy().get("tol_max", 3)
            accept = ((d >= state["anchor_dev"] - tol_anchor and d >= state["max_dev"] - tol_max)
                      or (parts["devB"] + parts["dfix"] >= a_parts["devB"] + a_parts["dfix"] + 2
                          and d >= state["anchor_dev"] - 3))
            s = None
            if accept:  # the benchmark is only reported, for accepted rounds
                bench = evaluate(cur, "both", f"bench{r}")
                if bench is None:
                    log(f"round {r}: benchmark failed; stopping")
                    return
                s = bench_score(bench)
                log(f"round {r}: " + analyze_round(r, "bench", f"bench{r}", f"bench{state.get('anchor_round')}"))
            log(f"round {r}: dev {d} {parts} (anchor {state['anchor_dev']}, max {state['max_dev']}) -> "
                f"{'ACCEPTED' if accept else 'rejected'}"
                + (f"; benchmark standard {s['standard']}/30, fresh {s['fresh']}/20, fixed {s['fixed']}/15, "
                   f"total {sum(s.values())}/65" if s else ""))
            state["history"].append({"round": r, "dev": d, **parts, **(s or {}), "accepted": accept})
            if accept:
                anchor = L / "anchor"
                old = Path(state["anchor"])
                shutil.rmtree(anchor.with_name("anchor.new"), ignore_errors=True)
                shutil.move(str(cur), str(anchor.with_name("anchor.new")))
                if old.resolve() != (L / "best").resolve():  # keep the r6 checkpoint as a fallback
                    shutil.rmtree(old, ignore_errors=True)
                shutil.rmtree(anchor, ignore_errors=True)
                shutil.move(str(anchor.with_name("anchor.new")), str(anchor))
                state.update(anchor=str(anchor), anchor_dev=d, anchor_bench=s, anchor_round=r, anchor_sub=parts)
                state["max_dev"] = max(state["max_dev"], d)
            else:
                shutil.rmtree(cur, ignore_errors=True)
            shutil.rmtree(data, ignore_errors=True)
            state["round"] = r
            state["stage"] = "data"
            save_state(state)
    log(f"TARGETS MET by round {state['anchor_round']}: {state['anchor_bench']}")


if __name__ == "__main__":
    main()
