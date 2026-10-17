"""Summarize prompt-diagnostic runs (prompt_diagnostic.py outputs).

usage: diag_summary.py OUT.json RUN.jsonl [RUN.jsonl ...]

Per model, group (same_wording / new_wording / all) and prompt variant: pass count, count of
answers that compute the neighbour, mean preference margin (nats) and the share of tasks where the
model prefers the reference; for each variant against the original prompt, the tasks gained and
lost (paired, same tasks) with an exact two-sided sign test, and the margin's paired change.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def sign_test(gained: int, lost: int) -> float:
    n = gained + lost
    if n == 0:
        return 1.0
    k = min(gained, lost)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def summarize(rows: list[dict]) -> dict:
    by = defaultdict(dict)
    for r in rows:
        by[r["variant"]][r["name"]] = r
    groups = {r["name"]: r["group"] for r in rows}
    out = {}
    for group in ("same_wording", "new_wording", "all"):
        names = sorted(n for n, g in groups.items() if group == "all" or g == group)
        block = {}
        base = by.get("original", {})
        for variant, rs in by.items():
            sel = [rs[n] for n in names if n in rs]
            if not sel:
                continue
            entry = {"tasks": len(sel), "margin_mean": round(sum(r["margin"] for r in sel) / len(sel), 3),
                     "prefers_reference": sum(r["margin"] > 0 for r in sel)}
            if "pass" in sel[0]:
                entry.update({"pass": sum(r["pass"] for r in sel), "wrong": sum(r["wrong"] for r in sel)})
            if variant != "original" and base:
                paired = [(base[n], rs[n]) for n in names if n in rs and n in base]
                if "pass" in sel[0]:
                    gained = sum(not a["pass"] and b["pass"] for a, b in paired)
                    lost = sum(a["pass"] and not b["pass"] for a, b in paired)
                    entry.update({"gained": gained, "lost": lost, "sign_test_p": round(sign_test(gained, lost), 4),
                                  "wrong_change": sum(b["wrong"] for _, b in paired) - sum(a["wrong"] for a, _ in paired)})
                d = [b["margin"] - a["margin"] for a, b in paired]
                up = sum(x > 0 for x in d)
                entry.update({"margin_change_mean": round(sum(d) / len(d), 3), "margin_up": up,
                              "margin_down": sum(x < 0 for x in d),
                              "margin_sign_p": round(sign_test(up, sum(x < 0 for x in d)), 4)})
            block[variant] = entry
        out[group] = block
    return out


def main() -> None:
    result = {}
    for path in sys.argv[2:]:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        name = Path(path).stem
        result[name] = summarize(rows)
        print("==", name)
        for group, block in result[name].items():
            for variant, e in block.items():
                print(f"  {group:13s} {variant:10s}", {k: v for k, v in e.items()})
    Path(sys.argv[1]).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
