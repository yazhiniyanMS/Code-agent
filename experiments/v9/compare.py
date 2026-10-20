"""V9-1: the C2 run against round 54 and the earlier runs, on identical item sets.

usage: compare.py [RUNS_DIR]     (default experiments/v9/results/runs; prints markdown tables)

* dev, unseen-v1 and benchmark: official, clean and robust scores with denominators. Round 54's
  benchmark is its saved V5.1 answers (experiments/v51/results/m4/anchor) rescored with the current
  scorer; V6 runs are their archived records (the only earlier runs with a benchmark score).
* task sets: strict (correct on all 40 probe inputs). Held-out families are split into the 5 tasks
  grammar v2 can express (ycode/lm/v9_grammar_split.json, measured before training) and the 61 it
  cannot. Paired with round 54 by task, with an exact sign test.
"""
import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "experiments" / "v9" / "results" / "runs"
BASES = (ROOT / "experiments" / "v8" / "results" / "base", ROOT / "experiments" / "v9" / "results" / "base")


def strict_rows(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    # keyed by position: a set can hold two tasks with one name (t_families: star_bars)
    return {(i, r["name"]): bool(r.get("strict")) for i, r in enumerate(x for x in rows if x["variant"] == "original")}


def sign_p(a: int, b: int) -> float:
    n = a + b
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(a, b) + 1)) / 2 ** n) if n else 1.0


def fmt(s: dict | None, key: str, of: str) -> str:
    return f"{s[key]}/{s[of]}" if s else "-"


def main() -> None:
    from ycode.lm.experiment import scores

    anchor = ROOT / "experiments" / "v51" / "results" / "m4" / "anchor"
    rows = [("Round 54", {w: scores(anchor / f"{w}.jsonl", w) for w in ("dev", "unseen", "bench")})]
    for version in ("v6", "v7", "v8"):
        for f in sorted((ROOT / "experiments" / version / "results" / "runs").glob("*/*/result.json")):
            rows.append((f"{version.upper()} {f.parts[-3]} {f.parts[-2]}", json.loads(f.read_text())))
    for f in sorted(RUNS.glob("*/*/result.json")):
        r = json.loads(f.read_text())
        rows.append((f"**V9 {f.parts[-3]} {f.parts[-2]}**", r))

    print("| Model | Dev clean | Unseen-v1 clean | Dev + unseen | Bench official | Bench clean | Bench robust |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for name, r in rows:
        d, u, b = r.get("dev"), r.get("unseen"), r.get("bench")
        total = d["clean"] + u["clean"] if d and u else "-"
        bo = f"{b.get('points', b.get('official'))}/{b.get('of', b.get('official_of'))}" if b else "-"
        print(f"| {name} | {fmt(d, 'clean', 'clean_of')} | {fmt(u, 'clean', 'clean_of')} | {total} | {bo} | "
              f"{fmt(b, 'clean', 'clean_of')} | {fmt(b, 'robust', 'robust_of') if b and 'robust_of' in b else '-'} |")

    split = json.loads((ROOT / "ycode" / "lm" / "v9_grammar_split.json").read_text())
    expressible = set(split["report"]["expressible_t_families"])
    print("\n| Set | Round 54 | " + " | ".join(f.parts[-3] + " " + f.parts[-2] for f in sorted(RUNS.glob("*/*/result.json")))
          + " | Only run / only round 54 (p) |")
    print("| --- | --- | --- | --- |")
    for name, base_file in (("c2_test", "round54_c2_test.jsonl"), ("c2_trained", "round54_c2_trained.jsonl"),
                            ("t_families", "round54_t_families.jsonl"), ("neighbours", "round54_neighbours.jsonl")):
        base = next((b for b in (strict_rows(d / base_file) for d in BASES) if b is not None), None)
        for run_dir in sorted(p.parent for p in RUNS.glob("*/*/result.json")):
            got = strict_rows(run_dir / f"set_{name}.jsonl")
            if got is None:
                continue
            groups = [("", lambda k: True)]
            if name == "t_families":
                groups = [(" (all)", lambda k: True), (" (expressible in grammar v2)", lambda k: k[1] in expressible),
                          (" (not expressible)", lambda k: k[1] not in expressible)]
            for label, keep in groups:
                keys = [k for k in got if keep(k)]
                g = sum(got[k] for k in keys)
                if base is not None and len(base) == len(got):
                    b = sum(base[k] for k in keys)
                    only_run = sum(got[k] and not base[k] for k in keys)
                    only_base = sum(base[k] and not got[k] for k in keys)
                    print(f"| {name}{label} | {b}/{len(keys)} | {g}/{len(keys)} | {only_run} / {only_base} "
                          f"(p = {sign_p(only_run, only_base):.3f}) |")
                else:
                    print(f"| {name}{label} | - | {g}/{len(keys)} | - |")
    run_dirs = sorted(p.parent for p in RUNS.glob("*/*/result.json"))
    by_domain(run_dirs)
    expressible_items(anchor, run_dirs)


def by_domain(run_dirs) -> None:
    """Grammar sets per domain: round 54 against each run."""
    from ycode.lm import compositions2 as G

    G.all_combinations()  # fills PARTS
    print("\n| Set | Domain | Round 54 | " + " | ".join(d.parts[-2] + " " + d.parts[-1] for d in run_dirs) + " |")
    print("| --- | --- | --- | " + " | ".join("---" for _ in run_dirs) + " |")
    for name in ("c2_test", "c2_trained"):
        base = strict_rows(BASES[1] / f"round54_{name}.jsonl")
        runs = [strict_rows(d / f"set_{name}.jsonl") for d in run_dirs]
        for domain in G.DOMAINS:
            keys = [k for k in base if G.PARTS[k[1]][0] == domain]
            cells = " | ".join(f"{sum(r[k] for k in keys)}/{len(keys)}" if r else "-" for r in runs)
            print(f"| {name} | {domain} | {sum(base[k] for k in keys)}/{len(keys)} | {cells} |")


def expressible_items(anchor: Path, run_dirs) -> None:
    """Clean dev, unseen-v1 and benchmark items that a grammar v2 combination computes (each excluded
    from training by the held-out check), against the other clean items."""
    from ycode.lm import compositions2 as G
    from ycode.lm import idioms

    _, dropped = idioms.held_out_check(G.all_combinations())
    items = {why.split(" ", 1)[1] for why in dropped.values() if why.startswith("computes ")}  # "split:item"
    print("\nGrammar v2 computes: " + ", ".join(sorted(items)))
    print("\n| Set | Items | Round 54 | " + " | ".join(d.parts[-2] + " " + d.parts[-1] for d in run_dirs)
          + " | Gained / lost (first run) |")
    print("| --- | --- | --- | " + " | ".join("---" for _ in run_dirs) + " | --- |")
    for which in ("dev", "unseen", "bench"):
        load = lambda f: {(r["suite"], r["name"]): bool(r["pass"]) for r in map(json.loads, f.read_text().splitlines())}  # noqa: E731
        base = load(anchor / f"{which}.jsonl")
        runs = [load(d / f"{which}.jsonl") for d in run_dirs]
        contaminated = set(json.loads((run_dirs[0] / "result.json").read_text())[which].get("contaminated", []))
        clean = [k for k in runs[0] if f"{k[0]}:{k[1]}" not in contaminated]
        split = {"dev": "dev", "unseen": "unseen", "bench": "bench"}[which]
        for label, keep in (("expressible", lambda k: f"{split}:{k[1]}" in items),
                            ("other", lambda k: f"{split}:{k[1]}" not in items)):
            keys = [k for k in clean if keep(k)]
            cells = " | ".join(str(sum(r[k] for k in keys)) for r in runs)
            gained = sorted(f"{a}:{b}" for a, b in keys if runs[0][(a, b)] and not base.get((a, b)))
            lost = sorted(f"{a}:{b}" for a, b in keys if base.get((a, b)) and not runs[0][(a, b)])
            print(f"| {which} {label} | {len(keys)} | {sum(base.get(k, False) for k in keys)} | {cells} | "
                  f"{', '.join(gained) or '-'} / {', '.join(lost) or '-'} |")


if __name__ == "__main__":
    main()
