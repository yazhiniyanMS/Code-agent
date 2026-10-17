"""V7 step 1: why does the model write a familiar function instead of the requested one?

usage: mechanism.py FREQUENCY.json OUT.json NAME=DEV.jsonl,UNSEEN.jsonl [NAME=...]

Development tasks only (dev and unseen-v1, clean items); the fixed holdout (unseen-v2) is never read.
The first model named is the one the correlation analysis is about (round 54).

1. Every failure gets one of five V7 classes, derived from the v5.1 classifier
   (ycode.lm.failures.classify_failure) and its evidence, in this order:

   * ``syntax_or_format``: no parsable function, an unclosed answer, or a function under a name
     that is not a training function;
   * ``wrong_function_retrieved``: the answer is (the body of) a training function, behaves exactly
     like one on every probe input, or is a function named after a training function;
   * ``prompt_misunderstood``: a coherent different task that is not a training function: it
     computes another held-out item, hands a bug fix back unchanged, or returns another type than
     the reference on most probe inputs (other than returning its input unchanged);
   * ``edge_case_errors``: agrees with the reference on >= 75% of probe inputs (or on all of them);
   * ``correct_concept_wrong_implementation``: everything else (the right kind of answer, computed
     wrongly, including calls to helpers or methods that do not exist).

2. For the write items that retrieved a training function R, how does R relate to the requested
   task T? Each feature ranks R among all training functions of T's arity (percentile 1.0: R is
   the most similar; 0.5: what a random training function scores):
   ``name`` (shared words of the function names), ``wording`` (shared content words of T's prompt
   and R's training task text), ``code`` (similarity of T's reference code and R's code, Python
   tokens), ``frequency`` (examples of R's body per training round).

3. Across all clean write items: does a close training neighbour predict retrieval? For each
   feature, the AUC of the nearest training neighbour's similarity for retrieved-function failures
   against every other item, against passed items and against the other failures (0.5: no relation).
"""
from __future__ import annotations

import difflib
import io
import json
import re
import subprocess
import sys
import tempfile
import tokenize
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

V7_CLASSES = ("wrong_function_retrieved", "correct_concept_wrong_implementation", "prompt_misunderstood",
              "edge_case_errors", "syntax_or_format")
FEATURES = ("name", "wording", "code", "frequency")
STOP = set("a an the of that is are in to and or if it its returns return given with as be by for on from "
           "number numbers value values list string python function write implement which whether true "
           "false each every all".split())


def catalogue() -> dict[str, dict]:
    """Every function round 54 trained on (basics concepts with their aliases, the compose exercises
    the mix draws from, skill families of earlier rounds): name -> params, arity, task text, source."""
    import ycode.lm.compose as compose
    from ycode.lm.basics import CONCEPTS

    out: dict[str, dict] = {}

    def add(name, params, task, source):
        arity = len([p for p in params.split(",") if p.strip()])
        out.setdefault(name, {"name": name, "params": params, "arity": arity, "task": task, "source": source})

    for c in CONCEPTS:
        for name in (c.name, *c.aliases):
            add(name, c.params, c.task, c.source(name))
    for e in compose.all_exercises(7) + compose.skill_exercises():
        add(e.name, e.params, e.task, e.source())
    return out


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+|\d+", text.lower()) if w not in STOP}


def name_words(name: str) -> set[str]:
    return {w for w in name.lower().split("_") if w}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def code_tokens(source: str) -> list[str]:
    from ycode.lm.leakage import normalized_body

    body = normalized_body(source)
    try:
        return [t.string for t in tokenize.generate_tokens(io.StringIO(body).readline)
                if t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                                  tokenize.COMMENT, tokenize.ENDMARKER)]
    except (tokenize.TokenError, IndentationError):
        return body.split()


def task_text(prompt: str) -> str:
    m = re.search(r"`[^`]*`,? (?:that|which) (.*)", prompt, re.S)
    return m.group(1) if m else prompt


def features(item_name: str, prompt: str, reference: str, cand: dict, freq: dict[str, float]) -> dict[str, float]:
    from ycode.lm.leakage import normalized_body

    return {"name": jaccard(name_words(item_name), name_words(cand["name"])),
            "wording": jaccard(words(task_text(prompt)), words(cand["task"])),
            "code": difflib.SequenceMatcher(None, code_tokens(reference), cand["tokens"], autojunk=False).ratio(),
            "frequency": freq.get(normalized_body(cand["source"]), 0.0)}


def percentile(value: float, others: list[float]) -> float:
    if not others:
        return float("nan")
    return (sum(o < value for o in others) + 0.5 * sum(o == value for o in others)) / len(others)


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def _type_agreement(code: str, name: str, reference: str, probes: list[tuple]) -> tuple[float, bool]:
    """(share of probes where the answer returns the reference's type, answer returns its first
    argument unchanged on every probe), run in a subprocess."""
    prog = f"""
import copy, json
_a, _r = {{}}, {{}}
exec({code!r}, _a)
exec({reference!r}, _r)
fa, fr = _a[{name!r}], _r[{name!r}]
same = n = ident = 0
for args in {probes!r}:
    try:
        want = fr(*copy.deepcopy(args))
    except Exception:
        continue
    n += 1
    try:
        got = fa(*copy.deepcopy(args))
    except Exception:
        continue
    same += type(got) is type(want)
    ident += repr(got) == repr(args[0])
print(json.dumps([same / n if n else 0.0, n > 0 and ident == n]))
"""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "types.py"
        script.write_text(prog)
        try:
            proc = subprocess.run([sys.executable, "-I", str(script)], cwd=tmp, capture_output=True, text=True,
                                  timeout=30, stdin=subprocess.DEVNULL)
            share, ident = json.loads(proc.stdout.strip().splitlines()[-1])
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            return 0.0, False
    return share, ident


def v7_class(f: dict, item, train: dict[str, dict]) -> tuple[str, str | None]:
    """(V7 class, retrieved training function or None) of one classified failure."""
    from ycode.lm.leakage import probe_inputs

    cls, ev = f["class"], f["evidence"]
    if cls == "formatting_or_parsing":
        m = re.match(r"function named (\w+), not", ev)
        if m and m.group(1) in train:
            return "wrong_function_retrieved", m.group(1)
        return "syntax_or_format", None
    if cls == "incorrect_task_interpretation":
        m = re.search(r"training function '(\w+)'", ev) or re.match(r"computes ([\w]+) instead", ev)
        if m and m.group(1) in train:
            return "wrong_function_retrieved", m.group(1)
        return "prompt_misunderstood", None  # another held-out item's function, or a fix handed back
    if cls in ("edge_case_failure", "evaluation_failure"):
        return "edge_case_errors", None
    if cls == "algorithmic_or_logic_error":
        share, ident = _type_agreement(f["code"], item.name, item.reference, probe_inputs(item))
        if share < 0.5 and not ident:
            return "prompt_misunderstood", None
    return "correct_concept_wrong_implementation", None


def load(path: Path, split: str) -> tuple[list[dict], dict]:
    from ycode.lm.erroranalysis import analyze, load_answers
    from ycode.lm.failures import classify_answers

    answers = load_answers(path)
    return analyze(answers, split)["items"], classify_answers(answers, split)


def main() -> None:
    from ycode.lm.leakage import CONTAMINATED_ITEMS, held_out_items

    freq = json.loads(Path(sys.argv[1]).read_text())["per_round"]
    out_path = Path(sys.argv[2])
    models = [a.split("=", 1) for a in sys.argv[3:]]
    train = catalogue()
    for c in train.values():
        c["tokens"] = code_tokens(c["source"])
    by_arity: dict[int, list[dict]] = {}
    for c in train.values():
        by_arity.setdefault(c["arity"], []).append(c)
    items = {(i.split, i.name): i for i in held_out_items() if i.split in ("dev", "unseen")}

    result = {"classes": V7_CLASSES, "features": FEATURES, "training_functions": len(train), "models": {}}
    if out_path.exists():  # resume: models already analysed are kept
        result["models"] = json.loads(out_path.read_text()).get("models", {})
    reference_rows = None
    for model, paths in models:
        if model in result["models"]:
            reference_rows = reference_rows or result["models"][model]["rows"]
            print(model, "(already analysed)", result["models"][model]["classes"], flush=True)
            continue
        rows = []
        for split, path in zip(("dev", "unseen"), paths.split(",")):
            answered, classified = load(Path(path), split)
            failed = {(f["suite"], f["name"]): f for f in classified["items"]}
            for it in answered:
                if (split, it["name"]) in CONTAMINATED_ITEMS:
                    continue
                item = items[(split, it["name"])]
                row = {"split": split, "suite": it["suite"], "name": it["name"], "pass": it["pass"],
                       "prompt": it["prompt"], "code": it.get("code") or ""}
                if not it["pass"]:
                    f = failed[(it["suite"], it["name"])]
                    row["v51_class"], row["evidence"] = f["class"], f["evidence"]
                    row["class"], row["retrieved"] = v7_class(f, item, train)
                rows.append(row)
        fails = [r for r in rows if not r["pass"]]
        counts = Counter(r["class"] for r in fails)
        result["models"][model] = {
            "clean_items": len(rows), "passed": len(rows) - len(fails), "failed": len(fails),
            "classes": {c: counts.get(c, 0) for c in V7_CLASSES},
            "by_split": {s: {c: sum(1 for r in fails if r["split"] == s and r["class"] == c) for c in V7_CLASSES}
                         for s in ("dev", "unseen")},
            "rows": rows}
        print(model, f"passed {len(rows) - len(fails)}/{len(rows)}", {c: counts.get(c, 0) for c in V7_CLASSES}, flush=True)
        out_path.write_text(json.dumps(result, indent=1))
        if reference_rows is None:
            reference_rows = rows

    # 2-3: correlations, for the first model's write items
    ret_ranks = []
    nearest = []
    for r in reference_rows:
        if r["suite"] != "write":
            continue
        item = items[(r["split"], r["name"])]
        cands = [c for c in by_arity.get(item.arity, []) if c["name"] != item.name]
        feats = {c["name"]: features(item.name, r["prompt"], item.reference, c, freq) for c in cands}
        retrieved = r.get("retrieved") if not r["pass"] else None
        nearest.append({"item": r["name"], "split": r["split"], "retrieved": retrieved is not None,
                        "status": "pass" if r["pass"] else r["class"],
                        **{f"{k}_nn": max((v[k] for v in feats.values()), default=0.0) for k in FEATURES}})
        if retrieved and retrieved in feats:
            mine = feats[retrieved]
            ret_ranks.append({"name": r["name"], "split": r["split"], "retrieved": retrieved,
                              **{f"{k}_value": round(mine[k], 3) for k in FEATURES},
                              **{f"{k}_percentile": round(percentile(mine[k], [v[k] for n, v in feats.items()
                                                                                if n != retrieved]), 3)
                                 for k in FEATURES},
                              **{f"{k}_top1": all(mine[k] >= v[k] for v in feats.values()) for k in FEATURES},
                              "candidates": len(feats)})
    summary = {}
    for k in FEATURES:
        ps = [x[f"{k}_percentile"] for x in ret_ranks]
        summary[k] = {"mean_percentile": round(sum(ps) / len(ps), 3) if ps else None,
                      "top1": sum(x[f"{k}_top1"] for x in ret_ranks)}
        pos = [n[f"{k}_nn"] for n in nearest if n["retrieved"]]
        for label, keep in (("auc_nearest_neighbour", lambda n: not n["retrieved"]),
                            ("auc_vs_passed", lambda n: n["status"] == "pass"),
                            ("auc_vs_other_failures", lambda n: not n["retrieved"] and n["status"] != "pass")):
            a = auc(pos, [n[f"{k}_nn"] for n in nearest if keep(n)])
            summary[k][label] = None if a is None else round(a, 3)
    result["retrievals"] = {"model": models[0][0], "write_items": len(nearest), "retrieved": len(ret_ranks),
                            "features": summary, "rows": ret_ranks, "nearest": nearest}
    print("retrievals of", models[0][0], len(ret_ranks), "of", len(nearest), "write items")
    for k in FEATURES:
        print(f"  {k:9s}", summary[k])
    out_path.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
