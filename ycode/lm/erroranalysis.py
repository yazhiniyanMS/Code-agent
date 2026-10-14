"""Why did each benchmark item fail, and which kinds of error cost the most points?

For every failed answer (from `ycode-lm eval --answers-out`), the candidate code is run against each
test assertion separately (in a subprocess, with a timeout), so a report shows the expected value,
the actual value or exception, and the reference solution. Each failure is put in one category:

  formatting             no code, a syntax error, or a broken def line
  instruction_following  wrong function name or parameters, the bug handed back unchanged, or the
                         body of a different (familiar) training function
  hallucination          calls a function, method or attribute that does not exist
  coding                 crashes (TypeError, IndexError, timeout, ...)
  factual                runs, but gets a domain fact wrong (a formula, which letters are vowels, ...)
  reasoning              runs, but computes the wrong thing

`compare()` diffs two runs item by item (same prompts, same scoring, same decoding settings).
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

CATEGORIES = ("formatting", "instruction_following", "hallucination", "coding", "factual", "reasoning")

# Items whose solution hinges on a fact rather than on program logic.
FACT_ITEMS = {
    "celsius_to_fahrenheit": "the formula F = C * 9 / 5 + 32",
    "seconds_to_minutes": "a minute has 60 seconds",
    "count_vowels": "the vowels are a, e, i, o, u",
    "starts_with_vowel": "the vowels are a, e, i, o, u",
    "is_prime": "what a prime number is",
    "fibonacci": "the Fibonacci sequence (0, 1, 1, 2, ...)",
    "factorial": "what a factorial is (0! = 1)",
    "area_of_square": "the area of a square is side * side",
}

_HARNESS = r'''
import json, sys
spec = json.loads(sys.stdin.read())
out = {"defined": [], "error": None, "asserts": []}
ns = {}
try:
    exec(compile(spec["code"], "<answer>", "exec"), ns)
    out["defined"] = sorted(k for k, v in ns.items() if callable(v) and not k.startswith("__"))
except BaseException as e:
    out["error"] = [type(e).__name__, str(e)[:200]]
if out["error"] is None:
    for line, call, expected, kind in spec["asserts"]:
        rec = {"assert": line, "call": call}
        try:
            actual = eval(call, ns)
            rec["actual"] = repr(actual)[:200]
            if kind == "approx":
                want = eval(expected, ns)
                rec["expected"], rec["ok"] = repr(want), abs(actual - want) < 1e-9
            elif kind == "eq":
                want = eval(expected, ns)
                rec["expected"], rec["ok"] = repr(want)[:200], actual == want
            elif kind == "is":
                want = eval(expected, ns)
                rec["expected"], rec["ok"] = repr(want)[:200], actual is want
            else:
                rec["expected"], rec["ok"] = expected, bool(actual) == (expected == "True")
        except BaseException as e:
            rec["error"], rec["ok"] = [type(e).__name__, str(e)[:200]], False
        out["asserts"].append(rec)
print("@@RESULT@@" + json.dumps(out))
'''


def parse_tests(tests: str) -> list[tuple[str, str, str, str]]:
    """(assert line, call expression, expected expression, kind) for each `assert` in a test block."""
    out = []
    for line in tests.splitlines():
        line = line.strip()
        if not line.startswith("assert "):
            continue
        try:
            node = ast.parse(line).body[0].test
        except SyntaxError:
            continue
        src = lambda n: ast.get_source_segment(line, n)  # noqa: E731
        if (isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Lt)
                and isinstance(node.left, ast.Call) and getattr(node.left.func, "id", "") == "abs"
                and isinstance(node.left.args[0], ast.BinOp) and isinstance(node.left.args[0].op, ast.Sub)):
            out.append((line, src(node.left.args[0].left), src(node.left.args[0].right), "approx"))
        elif isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
            out.append((line, src(node.left), src(node.comparators[0]), "eq"))
        elif isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Is):
            out.append((line, src(node.left), src(node.comparators[0]), "is"))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            out.append((line, src(node.operand), "False", "truth"))
        else:
            out.append((line, src(node), "True", "truth"))
    return out


def diagnose(code: str, tests: str, timeout: float = 10.0) -> dict:
    """Run ``code`` against each assertion of ``tests`` separately; never raises."""
    spec = json.dumps({"code": code, "asserts": parse_tests(tests)})
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run([sys.executable, "-I", "-c", _HARNESS], input=spec, capture_output=True, text=True,
                                  timeout=timeout, cwd=tmp)
        except subprocess.TimeoutExpired:
            return {"defined": [], "error": ["Timeout", f"no result within {timeout:.0f} s"], "asserts": []}
    marker = proc.stdout.rfind("@@RESULT@@")
    if marker < 0:
        return {"defined": [], "error": ["Crash", (proc.stderr or proc.stdout)[-200:]], "asserts": []}
    return json.loads(proc.stdout[marker + len("@@RESULT@@"):].splitlines()[0])


def _body_key(code: str) -> str | None:
    """A function's body with its docstring, own name and whitespace normalised away."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) != 1:
        return None
    fn = funcs[0]
    body = fn.body[1:] if (fn.body and isinstance(fn.body[0], ast.Expr)
                           and isinstance(getattr(fn.body[0], "value", None), ast.Constant)) else fn.body
    if not body:
        return None
    text = "\n".join(ast.unparse(s) for s in body)
    return re.sub(rf"\b{re.escape(fn.name)}\(", "SELF(", text)


_TRAINING_BODIES: dict[str, str] | None = None


def training_bodies() -> dict[str, str]:
    """Normalised body -> name, for every function the fine-tuning data teaches."""
    global _TRAINING_BODIES
    if _TRAINING_BODIES is None:
        from ycode.lm.basics import training_concepts
        from ycode.lm.compose import all_exercises, skill_exercises

        bodies: dict[str, str] = {}
        for c in training_concepts():
            if key := _body_key(c.source()):
                bodies.setdefault(key, c.name)
        # Skill bodies count whether or not today's runs include them: the checkpoints being
        # analysed were trained on them up to round 56.
        for e in all_exercises(7) + skill_exercises():
            if key := _body_key(e.source()):
                bodies.setdefault(key, e.name)
        _TRAINING_BODIES = bodies
    return _TRAINING_BODIES


def classify(name: str, code: str, diag: dict, *, buggy: str | None = None) -> tuple[str, str]:
    """(category, one-line reason) for a failed answer."""
    if not code.strip():
        return "formatting", "no code in the answer"
    err = diag.get("error")
    if err and err[0] in ("SyntaxError", "IndentationError"):
        glued = re.search(r"\bdef[A-Za-z_]", code)
        return "formatting", ("def line glued to the name ('def' + name with no space)" if glued
                              else f"syntax error: {err[1][:80]}")
    if err and err[0] == "Timeout":
        return "coding", "did not finish (infinite loop?)"
    if err and err[0] in ("NameError", "AttributeError", "ImportError", "ModuleNotFoundError"):
        return "hallucination", f"{err[0]} while defining: {err[1][:80]}"
    if err:
        return "coding", f"{err[0]} while defining: {err[1][:80]}"
    defined = diag.get("defined", [])
    if name not in defined:
        others = [d for d in defined if d != name]
        return ("instruction_following", f"named the function {others[0]!r} instead of {name!r}") if others else (
            "formatting", "no function definition")
    if buggy is not None and code.strip() == buggy.strip():
        return "instruction_following", "handed the buggy code back unchanged"
    key = _body_key(code)
    copied = training_bodies().get(key) if key else None
    if copied and copied != name:
        return "instruction_following", f"wrote the body of training function {copied!r}"
    failed = [a for a in diag.get("asserts", []) if not a.get("ok")]
    errors = [a for a in failed if "error" in a]
    if errors:
        e = errors[0]["error"]
        if e[0] == "TypeError" and re.search(r"positional argument|required argument|unexpected keyword", e[1]):
            return "instruction_following", f"wrong parameters: {e[1][:80]}"
        if e[0] in ("NameError", "AttributeError", "ImportError", "ModuleNotFoundError"):
            return "hallucination", f"{e[0]}: {e[1][:80]}"
        return "coding", f"{e[0]} on {errors[0]['call']}: {e[1][:60]}"
    if name in FACT_ITEMS:
        return "factual", f"wrong result; needs {FACT_ITEMS[name]}"
    if failed:
        f = failed[0]
        return "reasoning", f"{f['call']} gave {f.get('actual')} (expected {f.get('expected')})"
    return "reasoning", "fails the tests"


def _test_sets(which: str) -> dict[tuple[str, str], tuple[str, str | None, str | None]]:
    """(suite, name) -> (tests, reference, buggy code) for the benchmark or the dev set."""
    from ycode.lm.evaluate import BUGGY, PROBLEMS, fresh_problems
    from ycode.lm.references import FRESH_REFERENCE, REFERENCE

    out: dict = {}
    if which == "dev":
        from ycode.lm.devset import dev_bugs, dev_problems

        for p in dev_problems():
            out[("write", p.name)] = (p.tests, None, None)
        for sig, code, tests in dev_bugs():
            out[("bugfix", sig.split("(")[0])] = (tests, None, code)
        return out
    refs = {**REFERENCE, **FRESH_REFERENCE}
    for p in tuple(PROBLEMS) + tuple(fresh_problems()):
        out[("write", p.name)] = (p.tests, refs.get(p.name), None)
    tests = {p.name: p.tests for p in PROBLEMS}
    for sig, code in BUGGY:
        name = sig.split("(")[0]
        out[("bugfix", name)] = (tests[name], REFERENCE.get(name), code)
    return out


def _group(which: str, suite: str, name: str) -> str:
    if which == "dev":
        from ycode.lm.devset import DEV_CONCEPT_NAMES

        return "dev_fix" if suite == "bugfix" else ("dev_concept" if name in DEV_CONCEPT_NAMES else "dev_slot")
    from ycode.lm.evaluate import PROBLEMS

    return "fixed" if suite == "bugfix" else ("standard" if name in {p.name for p in PROBLEMS} else "fresh")


def analyze(answers: list[dict], which: str = "bench") -> dict:
    """Diagnose and classify every failed answer; summarise points lost by category and group."""
    tests = _test_sets(which)
    items, lost, by_group = [], Counter(), {}
    for row in answers:
        key = (row["suite"], row["name"])
        if key not in tests:
            continue
        test_block, reference, buggy = tests[key]
        group = _group(which, *key)
        g = by_group.setdefault(group, {"passed": 0, "total": 0, "lost": Counter()})
        g["total"] += 1
        item = {"suite": row["suite"], "group": group, "name": row["name"], "pass": row["pass"],
                "prompt": row["prompt"], "answer": row["answer"], "code": row["code"], "expected": reference}
        if row["pass"]:
            g["passed"] += 1
        else:
            diag = diagnose(row["code"], test_block)
            item["category"], item["reason"] = classify(row["name"], row["code"], diag, buggy=buggy)
            item["failed_asserts"] = [a for a in diag.get("asserts", []) if not a.get("ok")][:3]
            item["error"] = diag.get("error")
            lost[item["category"]] += 1
            g["lost"][item["category"]] += 1
        items.append(item)
    total = sum(g["total"] for g in by_group.values())
    return {"set": which, "points": sum(g["passed"] for g in by_group.values()), "of": total,
            "lost_by_category": {c: lost[c] for c in CATEGORIES},
            "groups": {k: {"passed": v["passed"], "total": v["total"], "lost": dict(v["lost"])}
                       for k, v in sorted(by_group.items())},
            "items": items}


def compare(old: dict, new: dict) -> dict:
    """Item-by-item difference between two analyses of the same prompts."""
    before = {(i["suite"], i["name"]): i for i in old["items"]}
    gained, lost, changed = [], [], []
    for i in new["items"]:
        b = before.get((i["suite"], i["name"]))
        if b is None:
            continue
        if i["pass"] and not b["pass"]:
            gained.append(i["name"])
        elif b["pass"] and not i["pass"]:
            lost.append(i["name"])
        elif not i["pass"] and i.get("category") != b.get("category"):
            changed.append((i["name"], b.get("category"), i.get("category")))
        same_answer = b.get("answer") == i.get("answer")
        i.setdefault("same_answer_as_before", same_answer)
    return {"points": [old["points"], new["points"]], "gained": gained, "lost": lost,
            "category_changes": changed,
            "lost_by_category": {c: [old["lost_by_category"][c], new["lost_by_category"][c]] for c in CATEGORIES},
            "identical_answers": sum(1 for i in new["items"] if i.get("same_answer_as_before")),
            "items": len(new["items"])}


def model_fingerprint(model_dir: Path) -> str | None:
    """SHA-256 of a checkpoint's model.pt, cached next to it (keyed by size and mtime): hashing 5 GB is slow."""
    path = Path(model_dir) / "model.pt"
    if not path.is_file():
        return None
    stat = path.stat()
    stamp = f"{stat.st_size}:{stat.st_mtime_ns}"
    cache = Path(model_dir) / "model.sha256"
    if cache.is_file():
        cached_stamp, _, digest = cache.read_text().strip().partition(" ")
        if cached_stamp == stamp:
            return digest
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    try:
        cache.write_text(f"{stamp} {h.hexdigest()}")
    except OSError:
        pass
    return h.hexdigest()


def failure_rows(result: dict, meta: dict) -> list[dict]:
    """One flat record per failed item: prompt, expected, actual, category, model version, settings."""
    rows = []
    for i in result["items"]:
        if i["pass"]:
            continue
        rows.append({**meta, "set": result["set"], "group": i["group"], "suite": i["suite"], "name": i["name"],
                     "category": i["category"], "reason": i["reason"], "prompt": i["prompt"],
                     "expected_reference": i["expected"], "failed_asserts": i.get("failed_asserts", []),
                     "error": i.get("error"), "actual_answer": i["answer"], "actual_code": i["code"]})
    return rows


def lost_points_table(failures: list[dict]) -> dict:
    """Aggregate failure records: points lost per category, per group, and the most-failed items."""
    by_cat = Counter(f["category"] for f in failures)
    by_group: dict[str, Counter] = {}
    by_item: dict[tuple, Counter] = {}
    for f in failures:
        by_group.setdefault(f["group"], Counter())[f["category"]] += 1
        by_item.setdefault((f["set"], f["group"], f["name"]), Counter())[f["category"]] += 1
    items = sorted(by_item.items(), key=lambda kv: -sum(kv[1].values()))
    return {"total_lost": len(failures), "by_category": dict(by_cat.most_common()),
            "by_group": {g: dict(c.most_common()) for g, c in sorted(by_group.items())},
            "items": [{"set": k[0], "group": k[1], "name": k[2], "failures": sum(c.values()),
                       "categories": dict(c.most_common())} for k, c in items]}


def load_answers(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def format_report(result: dict) -> str:
    lines = [f"{result['set']}: {result['points']}/{result['of']} passed"]
    for group, g in result["groups"].items():
        lost = ", ".join(f"{k} {v}" for k, v in sorted(g["lost"].items(), key=lambda kv: -kv[1]))
        lines.append(f"  {group:12} {g['passed']:>3}/{g['total']:<3} lost: {lost or '-'}")
    lines.append("points lost by category: " + ", ".join(
        f"{c} {n}" for c, n in sorted(result["lost_by_category"].items(), key=lambda kv: -kv[1]) if n))
    for c in CATEGORIES:
        fails = [i for i in result["items"] if i.get("category") == c]
        if fails:
            lines.append(f"\n[{c}] {len(fails)}")
            lines += [f"  {i['group']:8} {i['name']:22} {i['reason']}" for i in fails]
    return "\n".join(lines)
