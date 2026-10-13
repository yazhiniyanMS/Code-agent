"""Recover round 53's rejection reason and re-score its saved answers under the same conditions."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/home/user/Code-agent")
from ycode.lm.devset import dev_bugs, dev_problems
from ycode.lm.evaluate import run_tests

L = Path("/home/user/lmwork/loop")
line = next(ln for ln in (L / "loop.log").read_text().splitlines() if "round 53: dev" in ln)
m = re.search(r"round 53: dev (\d+) (\{.*?\}) \(anchor (\d+), max (\d+)\) -> (\w+)", line)
d, parts, anchor_dev, max_dev, verdict = int(m[1]), eval(m[2]), int(m[3]), int(m[4]), m[5]
# Policy in force for round 53 (loop/policy.json before 2026-10-13 15:17): tol_anchor 3, tol_max 3.
tol_anchor, tol_max = 3, 3
rule_a = d >= anchor_dev - tol_anchor
rule_b = d >= max_dev - tol_max
print(f"logged: {line[22:]}")
print(f"rule 1: dev {d} >= anchor {anchor_dev} - {tol_anchor} = {anchor_dev - tol_anchor}: {rule_a}")
print(f"rule 2: dev {d} >= best ever {max_dev} - {tol_max} = {max_dev - tol_max}: {rule_b}")
print(f"=> {'accepted' if rule_a and rule_b else 'rejected'} (logged: {verdict}); the best-ever dev 40 came from round 50")

tests = {("write", p.name): p.tests for p in dev_problems()}
tests.update({("bugfix", s.split("(")[0]): t for s, _, t in dev_bugs()})
rows = [json.loads(x) for x in (L / "dev53.jsonl").read_text().splitlines()]
same = sum(run_tests(r["code"], tests[(r["suite"], r["name"])]) == r["pass"] for r in rows)
rescored = sum(run_tests(r["code"], tests[(r["suite"], r["name"])]) for r in rows)
print(f"re-scored the {len(rows)} saved round-53 dev answers with the same harness: {rescored} pass "
      f"(logged {d}); {same}/{len(rows)} items get the same verdict")
anchor = {(r["suite"], r["name"]): r["pass"] for r in map(json.loads, (L / "dev52.jsonl").read_text().splitlines())}
now = {(r["suite"], r["name"]): r["pass"] for r in rows}
print("vs its anchor (round 52): gained", sorted(f"{k[0]}:{k[1]}" for k in now if now[k] and not anchor[k]),
      "lost", sorted(f"{k[0]}:{k[1]}" for k in now if anchor[k] and not now[k]))
