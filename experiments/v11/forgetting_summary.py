"""V11: summarise a forgetting-score file (experiments/v11/forgetting_scores.py) by kind of example.

usage: forgetting_summary.py SCORES.json [--draw]

Groups round 54's production examples by tag (write / bug fix) and, for bug fixes, by the kind of edit
the fix makes (a flipped comparison, a changed literal, other), with C2's mean loss rise on each, and
lists the most-forgotten examples. With --draw, also compares the replay the spec's arm draws for its own
curriculum (computed as rounddata.write_round_dataset does) uniformly (R1) and by forgetting score
(V11-B), using rounddata's own sampler.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
COMPARISONS = {"<", ">", "<=", ">=", "==", "!="}


def edit_kind(prompt: str, answer: str) -> str:
    """What a bug fix changes: 'comparison' (a comparison operator), 'literal' (a string or number),
    or 'other'; from the token difference between the buggy code and the fix."""
    buggy = re.search(r"```python\n(.*?)\n```", prompt, re.S)
    fixed = re.search(r"```python\n(.*?)\n```", answer, re.S)
    if not (buggy and fixed):
        return "other"
    split = lambda s: re.findall(r"<=|>=|==|!=|'[^']*'|\"[^\"]*\"|\d+|\w+|\S", s)
    a, b = split(buggy.group(1)), split(fixed.group(1))
    changed = [t for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
               if tag != "equal" for t in a[i1:i2] + b[j1:j2]]
    if changed and all(t in COMPARISONS for t in changed):
        return "comparison"
    if changed and all(t[0] in "'\"" or t.isdigit() for t in changed):
        return "literal"
    return "other"


def main() -> None:
    from ycode.lm import rounddata

    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("--draw", action="store_true")
    args = ap.parse_args()
    s = json.loads(Path(args.scores).read_text())
    spec = json.loads(Path(s["spec"]).read_text())
    arm = next(a for a in spec["arms"] if a["name"] == s["arm"])
    o = {**rounddata.DEFAULTS, **arm["data"]}
    production = rounddata._legacy_examples(s["seed"], {**o, **rounddata.PRODUCTION})
    assert len(production) == s["n"] and [e.prompt[:120] for e in production] == s["prompts_head"]
    kinds = [("bugfix:" + edit_kind(e.prompt, e.response)) if t == "legacy:bugfix" else "write"
             for e, t in zip(production, s["tags"])]
    groups = defaultdict(list)
    for k, f, b, a in zip(kinds, s["forgetting"], s["loss_before"], s["loss_after"]):
        groups[k].append((f, b, a))
    print(f"seed {s['seed']}: {s['n']} examples; before = {s['before']}, after = {s['after']}\n")
    print("| Kind | Examples | Loss round 54 | Loss C2 | Mean rise | Share rising | Share of weight |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    total_w = sum(s["weights"])
    for k in sorted(groups, key=lambda k: -len(groups[k])):
        g = groups[k]
        w = sum(s["weights"][i] for i, kk in enumerate(kinds) if kk == k)
        print(f"| {k} | {len(g)} | {sum(x[1] for x in g) / len(g):.3f} | {sum(x[2] for x in g) / len(g):.3f} | "
              f"{sum(x[0] for x in g) / len(g):+.3f} | {sum(x[0] > 0 for x in g) / len(g):.0%} | {w / total_w:.1%} |")
    allf = s["forgetting"]
    print(f"| all | {len(allf)} | {sum(s['loss_before']) / len(allf):.3f} | {sum(s['loss_after']) / len(allf):.3f} | "
          f"{sum(allf) / len(allf):+.3f} | {sum(f > 0 for f in allf) / len(allf):.0%} | 100% |")
    print("\nMost forgotten (loss rise, kind, prompt start):")
    for i in sorted(range(len(allf)), key=lambda i: -allf[i])[:15]:
        print(f"  {allf[i]:+.3f}  {kinds[i]:18s} {s['prompts_head'][i][:100]!r}")
    if args.draw:
        from ycode.lm.data import encode_example
        from ycode.lm.tokenizer import BPETokenizer

        tok = BPETokenizer.load(spec["tokenizer"])
        own = sum(len(encode_example(tok, e, weighted=True)[0]) for e in rounddata.round_examples_tagged(s["seed"], o)[0])
        logs = []
        idx = {(e.prompt, e.response): i for i, e in enumerate(production)}
        for label, weights in (("uniform (R1)", None), ("by forgetting (V11-B)", args.scores)):
            o2 = {**o, "legacy_weights": weights}
            picked, tags, _ = rounddata._legacy_sample(s["seed"], o2, own, spec["tokenizer"], spec.get("replay_dir"),
                                                       logs.append)
            drawn = [idx[(e.prompt, e.response)] for e in picked]
            ks = [kinds[i] for i in drawn]
            counts = {k: ks.count(k) for k in sorted(set(kinds))}
            print(f"\nreplay drawn {label}: {len(picked)} examples, {counts}, "
                  f"mean rise of those drawn {sum(allf[i] for i in drawn) / len(drawn):+.3f}")


if __name__ == "__main__":
    main()
