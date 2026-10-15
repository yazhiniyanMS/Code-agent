"""Classify held-out failures into the six v5.1 error classes, with the evidence for each.

Rules, applied in order to every failed answer:

1. ``formatting_or_parsing``: no code block or ``def``, a syntax error, or a function under the
   wrong name (the harness cannot call it).
2. ``incorrect_code_generation``: the code runs into undefined names, missing attributes or wrong
   types on the tests' inputs (an invented helper or API, e.g. ``a.dict``, ``toupper``).
3. ``incorrect_task_interpretation``: the code is (the body of) a trained function or computes
   exactly what another known function computes (a training function or another held-out item):
   it solves a different, coherent task, e.g. ``fifth_item`` for ``fourth_item``, ``a + b`` for
   ``smaller``; or a bug fix handed the buggy code back unchanged.
4. ``evaluation_failure``: the code agrees with the item's reference solution on every probe input
   yet fails the official tests (a timeout, a harness or test problem).
5. ``edge_case_failure``: the code agrees with the reference on at least 75% of probe inputs.
6. ``algorithmic_or_logic_error``: everything else: the right task, computed wrongly.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

CLASSES = ("incorrect_task_interpretation", "algorithmic_or_logic_error", "incorrect_code_generation",
           "edge_case_failure", "formatting_or_parsing", "evaluation_failure")
EDGE_AGREEMENT = 0.75


def _behaviour_twin(code: str, item, probes: list[tuple], candidates: dict[str, str]) -> str | None:
    """Name of a candidate function (training function or other held-out reference) whose outputs
    equal the answer's on every probe where both run, or None. One subprocess for all candidates."""
    import json
    import subprocess
    import sys
    import tempfile

    fn = re.match(r"\s*def (\w+)\(", code)
    if not fn or not probes:
        return None
    program = [code, "import copy, json", "_cands = {}"]
    for name, src in candidates.items():
        defined = re.match(r"\s*def (\w+)", src).group(1)
        program.append(f"try:\n    exec({src!r}, _g := {{}})\n    _cands[{name!r}] = _g[{defined!r}]\n"
                       "except Exception:\n    pass")
    program.append(f"""
def _run(f, args):
    try:
        return ('ok', repr(f(*copy.deepcopy(args))))
    except Exception as e:
        return ('err', type(e).__name__)
_probes = {probes!r}
_mine = [_run({fn.group(1)}, a) for a in _probes]
_identity = [('ok', repr(a[0])) for a in _probes]
_hits = []
if sum(x == y for x, y in zip(_mine, _identity)) == len(_probes):
    _cands = {{}}  # returning the input unchanged is a logic error, not another task
for _n, _f in _cands.items():
    _theirs = [_run(_f, a) for a in _probes]
    _both = [(x, y) for x, y in zip(_mine, _theirs) if x[0] == y[0] == 'ok']
    if len(_both) >= 4 and all(x == y for x, y in _both):
        _hits.append(_n)
print(json.dumps(_hits))""")
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "twin.py"
        script.write_text("\n".join(program))
        try:
            proc = subprocess.run([sys.executable, "-I", str(script)], cwd=tmp, capture_output=True,
                                  text=True, timeout=60, stdin=subprocess.DEVNULL)
            hits = json.loads(proc.stdout.strip().splitlines()[-1])
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            return None
    return hits[0] if hits else None


def _candidates_by_arity() -> dict[int, dict[str, str]]:
    """Training functions (one per distinct behaviour-relevant source) and held-out references."""
    import ycode.lm.compose as compose
    from ycode.lm.basics import CONCEPTS
    from ycode.lm.leakage import held_out_items

    out: dict[int, dict[str, str]] = {}

    def add(name: str, params: str, src: str) -> None:
        arity = len([p for p in params.split(",") if p.strip()])
        out.setdefault(arity, {}).setdefault(name, src)

    for c in CONCEPTS:
        add(c.name, c.params, c.source())
    for e in compose._raw_exercises(7) + compose.skill_exercises():
        add(e.name, e.params, e.source())
    for i in held_out_items():
        params = re.match(r"\s*def \w+\(([^)]*)\)", i.reference).group(1)
        add(f"{i.split}:{i.name}", params, i.reference)
    return out


def classify_failure(row: dict, item, diag_category: str, diag_reason: str, candidates: dict[int, dict[str, str]]) -> tuple[str, str]:
    """(class, evidence) for one failed answer. ``diag_category``/``reason`` come from
    ``erroranalysis.analyze`` (which runs each assertion)."""
    from ycode.lm.leakage import agreement, probe_inputs

    code = row.get("code") or ""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return "formatting_or_parsing", f"syntax error: {exc.msg}"
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if fn is None:
        return "formatting_or_parsing", "no function in the answer"
    if fn.name != item.name:
        return "formatting_or_parsing", f"function named {fn.name}, not {item.name}"
    if diag_category == "hallucination" or (diag_category == "coding" and re.search(r"NameError|AttributeError|TypeError", diag_reason)):
        return "incorrect_code_generation", diag_reason
    if "handed the buggy code back" in diag_reason:
        return "incorrect_task_interpretation", diag_reason
    if "wrote the body of training function" in diag_reason:
        return "incorrect_task_interpretation", diag_reason
    probes = probe_inputs(item)
    score = agreement(code, fn.name, item, probes, isolate=True) if probes else 0.0
    if score == 1.0:
        return "evaluation_failure", f"agrees with the reference on all {len(probes)} probes but fails the tests ({diag_reason})"
    twin = _behaviour_twin(code, item, probes, {n: s for n, s in candidates.get(item.arity, {}).items()
                                                if n != f"{item.split}:{item.name}"})
    if twin is not None and score < EDGE_AGREEMENT:
        return "incorrect_task_interpretation", f"computes {twin} instead (agreement with the reference {score:.2f})"
    if score >= EDGE_AGREEMENT:
        return "edge_case_failure", f"agrees with the reference on {score:.0%} of probes ({diag_reason})"
    return "algorithmic_or_logic_error", f"agreement with the reference {score:.0%} ({diag_reason})"


def classify_answers(answers: list[dict], split: str) -> dict:
    """Every failure of one answers file with its class and evidence, and the classes ranked."""
    from ycode.lm.erroranalysis import analyze
    from ycode.lm.leakage import CONTAMINATED_ITEMS, held_out_items

    items = {i.name: i for i in held_out_items() if i.split == split}
    result = analyze(answers, split)
    candidates = _candidates_by_arity()
    failures = []
    for it in result["items"]:
        if it["pass"]:
            continue
        item = items.get(it["name"])
        if item is None:
            continue
        cls, evidence = classify_failure(it, item, it.get("category", ""), it.get("reason", "") or "", candidates)
        failures.append({"split": split, "suite": it["suite"], "name": it["name"], "group": it["group"],
                         "contaminated": (split, it["name"]) in CONTAMINATED_ITEMS, "prompt": it["prompt"],
                         "expected": item.reference, "answer": it["answer"], "code": it["code"],
                         "test_outcome": it.get("reason"), "failed_asserts": it.get("failed_asserts"),
                         "class": cls, "evidence": evidence, "v5_category": it.get("category")})
    ranked = Counter(f["class"] for f in failures)
    clean = Counter(f["class"] for f in failures if not f["contaminated"])
    return {"split": split, "answered": len(result["items"]), "failures": len(failures),
            "ranked": ranked.most_common(), "ranked_clean": clean.most_common(),
            "items": failures}
