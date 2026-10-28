"""V10: every candidate against round 54 and C2 on identical item sets, under PLAN.md's rules.

usage: compare.py          (reads experiments/v10/results; prints markdown tables)

* Selection: dev clean /44, unseen-v1 clean /53, combined /97, family_new /24 (strict on 40 inputs).
* Reported: neighbours /38, held-out families /66 (split by what grammar v2 or v3 can express, measured
  before training), reproduction (c2_trained, c3_trained), held-out combinations (c2_test, c3_test),
  the benchmark (official / clean / robust), cost and identity.
* Forgetting: clean dev and unseen-v1 items that round 54 (or C2) passes and the candidate fails,
  with the items_lost category rule (ycode.lm.experiment.compare: >= 3 in a category fails).
Round 54's rows: its saved V5.1 answers (dev, unseen, bench) and V8/V9/V10 base task-set rows.
C2's rows: V9-1's record (bit-identical checkpoint, rebuilt in R0) plus R0's family_new.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RES = ROOT / "experiments" / "v10" / "results"
SETS = ("family_new", "neighbours", "t_families", "c2_trained", "c2_test", "c3_trained", "c3_test")
BASE_ROWS = {"family_new": "experiments/v8/results/base/round54_family_new.jsonl",
             "neighbours": "experiments/v8/results/base/round54_neighbours.jsonl",
             "t_families": "experiments/v8/results/base/round54_t_families.jsonl",
             "c2_trained": "experiments/v9/results/base/round54_c2_trained.jsonl",
             "c2_test": "experiments/v9/results/base/round54_c2_test.jsonl",
             "c3_trained": "experiments/v10/results/base/round54_c3_trained.jsonl",
             "c3_test": "experiments/v10/results/base/round54_c3_test.jsonl"}
C2_ROWS = {**{s: f"experiments/v9/results/runs/C2/1301/set_{s}.jsonl" for s in ("neighbours", "t_families", "c2_trained", "c2_test")},
           "family_new": "experiments/v10/results/runs/C2/1301/set_family_new.jsonl",
           "c3_trained": "experiments/v10/results/base/C2_c3_trained.jsonl",
           "c3_test": "experiments/v10/results/base/C2_c3_test.jsonl"}


def strict(path: Path) -> list[tuple[str, bool]] | None:
    if not path.is_file():
        return None
    return [(r["name"], bool(r.get("strict"))) for r in map(json.loads, path.read_text().splitlines())
            if r.get("variant", "original") == "original"]


def answers(path: Path) -> dict:
    return {f"{r['suite']}:{r['name']}": bool(r["pass"]) for r in map(json.loads, path.read_text().splitlines())}


def models() -> list[dict]:
    from ycode.lm.experiment import scores

    anchor = ROOT / "experiments" / "v51" / "results" / "m4" / "anchor"
    c2 = json.loads((ROOT / "experiments" / "v9" / "results" / "runs" / "C2" / "1301" / "result.json").read_text())
    out = [{"name": "Round 54", "dir": anchor, "rec": {w: scores(anchor / f"{w}.jsonl", w) for w in ("dev", "unseen", "bench")},
            "sets": {s: strict(ROOT / p) for s, p in BASE_ROWS.items()}},
           {"name": "C2", "dir": ROOT / "experiments" / "v9" / "results" / "runs" / "C2" / "1301", "rec": c2,
            "sets": {s: strict(ROOT / p) for s, p in C2_ROWS.items()}}]
    for f in sorted(RES.glob("runs/*/*/result.json")):
        if f.parts[-3] == "C2":
            continue
        r = json.loads(f.read_text())
        out.append({"name": f"{f.parts[-3]} {f.parts[-2]}", "dir": f.parent, "rec": r,
                    "sets": {s: strict(f.parent / f"set_{s}.jsonl") for s in SETS}})
    return out


def clean_passes(m: dict, which: str, contaminated: set) -> set:
    return {k for k, v in answers(m["dir"] / f"{which}.jsonl").items() if v and k not in contaminated}


def main() -> None:
    ms = models()
    ref = ms[2]["rec"] if len(ms) > 2 else ms[1]["rec"]
    dirty = {w: set(ref[w].get("contaminated", [])) for w in ("dev", "unseen", "bench")}
    expressible = set(json.loads((ROOT / "ycode" / "lm" / "v9_grammar_split.json").read_text())["report"]["expressible_t_families"])
    expressible |= set(json.loads((ROOT / "ycode" / "lm" / "v10_grammar3_split.json").read_text())["report"]["expressible_t_families"])

    def cell(rows, keep=lambda n: True):
        if rows is None:
            return "-"
        rs = [v for n, v in rows if keep(n)]
        return f"{sum(rs)}/{len(rs)}"

    print("| Model | Dev /44 | Unseen-v1 /53 | Combined /97 | family_new /24 | Neighbours /38 | Held-out families /66 "
          "(expressible 7 · other 59) | Reproduction c2 /40 · c3 /40 | Held-out combinations c2 /60 · c3 /60 | "
          "Bench official · clean · robust |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for m in ms:
        r, s = m["rec"], m["sets"]
        d, u, b = r["dev"]["clean"], r["unseen"]["clean"], r.get("bench")
        bench = f"{b.get('points', b.get('official'))} · {b['clean']} · {b['robust']}" if b else "-"
        tf = s.get("t_families")
        print(f"| {m['name']} | {d} | {u} | **{d + u}** | {cell(s.get('family_new'))} | {cell(s.get('neighbours'))} | "
              f"{cell(tf)} ({cell(tf, lambda n: n in expressible)} · {cell(tf, lambda n: n not in expressible)}) | "
              f"{cell(s.get('c2_trained'))} · {cell(s.get('c3_trained'))} | {cell(s.get('c2_test'))} · {cell(s.get('c3_test'))} | "
              f"{bench} |")

    print("\n| Model | Lost against round 54 (by category) | Gained against round 54 | Lost / gained against C2 | "
          "Category gate (>= 3 fails) |")
    print("| --- | --- | --- | --- | --- |")
    base = {w: clean_passes(ms[0], w, dirty[w]) for w in ("dev", "unseen")}
    c2p = {w: clean_passes(ms[1], w, dirty[w]) for w in ("dev", "unseen")}
    for m in ms[1:]:
        lost, gained, lost_c2, gained_c2, cats = [], [], [], [], Counter()
        for w in ("dev", "unseen"):
            p = clean_passes(m, w, dirty[w])
            for k in sorted(base[w] - p):
                lost.append(f"{w}:{k}")
                cats[m["rec"][w]["categories"].get(k, "?")] += 1
            gained += [f"{w}:{k}" for k in sorted(p - base[w])]
            lost_c2 += [f"{w}:{k}" for k in sorted(c2p[w] - p)]
            gained_c2 += [f"{w}:{k}" for k in sorted(p - c2p[w])]
        gate = "fails" if any(v >= 3 for v in cats.values()) else "passes"
        print(f"| {m['name']} | {len(lost)}: {', '.join(lost) or '-'} ({dict(cats)}) | {len(gained)}: {', '.join(gained) or '-'} | "
              f"{len(lost_c2)} / {len(gained_c2)} | {gate} |")

    print("\n| Run | Start | Trained tokens | Steps | Train min | Total min | Peak RAM MB | Checkpoint sha256 | Packed content sha256 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for m in ms[1:]:
        r, t = m["rec"], m["rec"]["training"]
        packed = (RES / "runs" / "C2" / "1301" if m["name"] == "C2" else m["dir"]) / "packed_content_sha256.txt"
        print(f"| {m['name']} | {Path(r['init_from']).parts[-3] if r.get('init_from') else 'round 54'} | "
              f"{t['tokens']:,} | {t['steps']} | {t['minutes']:.0f} | {r['minutes_total']:.0f} | {t['peak_rss_mb']} | "
              f"`{t['checkpoint_sha256'][:12]}` | {('`' + packed.read_text().strip()[:12] + '`') if packed.is_file() else '-'} |")


if __name__ == "__main__":
    main()
