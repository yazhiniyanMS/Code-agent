"""Versioned evaluation sets and their three scoring views, with explicit denominators.

Sets:

* ``bench``: the benchmark: 30 standard + 20 fresh write problems and 15 bug fixes (65). Its tests
  and the official score are unchanged; it is reported, never used to choose.
* ``dev``: 47 write problems + 15 bug fixes one slot word from trained concepts (62).
* ``unseen``: write problems whose concepts no generator produces (``ycode.lm.unseen``).

Views of a set:

* **official**: the item's own tests, every item;
* **clean**: official, without the items v5's checkpoints saw in equivalent form
  (``leakage.CONTAMINATED_ITEMS``);
* **robust**: official tests and the independent tests of ``robust_tests_v1`` both pass (every item).

The primary metric for unseen-concept generalisation is **unseen_clean** = clean dev + unseen.
``manifest()`` fingerprints every set (names, prompts, tests, robust tests), so results can be
traced to the exact items they were computed on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

EVAL_VERSION = "eval-v5.1-1"


@dataclass(frozen=True)
class EvalItem:
    split: str        # bench | dev | unseen | unseen2
    suite: str        # write | bugfix
    name: str
    prompt: str
    tests: str        # the official tests
    robust_tests: str  # independent assertions (robust_tests_v1); empty if none
    contaminated: bool


def eval_items(split: str) -> list[EvalItem]:
    from ycode.lm.evaluate import BUGFIX_PROMPT, BUGGY, PROBLEMS, fresh_problems
    from ycode.lm.leakage import CONTAMINATED_ITEMS
    from ycode.lm.robust_tests_v1 import ROBUST_TESTS

    def item(suite, name, prompt, tests):
        return EvalItem(split, suite, name, prompt, tests, "".join(ROBUST_TESTS.get((split, name), ())),
                        (split, name) in CONTAMINATED_ITEMS)

    if split == "bench":
        tests = {p.name: p.tests for p in PROBLEMS}
        return ([item("write", p.name, p.prompt, p.tests) for p in tuple(PROBLEMS) + tuple(fresh_problems())]
                + [item("bugfix", sig.split("(")[0], BUGFIX_PROMPT.format(code=code), tests[sig.split("(")[0]])
                   for sig, code in BUGGY])
    if split == "dev":
        from ycode.lm.devset import dev_bugs, dev_problems

        return ([item("write", p.name, p.prompt, p.tests) for p in dev_problems()]
                + [item("bugfix", sig.split("(")[0], BUGFIX_PROMPT.format(code=code), tests)
                   for sig, code, tests in dev_bugs()])
    if split == "unseen":
        from ycode.lm.unseen import unseen_problems

        return [item("write", p.name, p.prompt, p.tests) for p in unseen_problems()]
    if split == "unseen2":  # independently written (V6); its own tests have >= 4 asserts, no robust extra
        from ycode.lm.unseen2 import unseen2_problems

        return [item("write", p.name, p.prompt, p.tests) for p in unseen2_problems()]
    raise ValueError(f"unknown evaluation set {split!r}")


def fingerprint(items: list[EvalItem]) -> str:
    rows = [[i.split, i.suite, i.name, i.prompt, i.tests, i.robust_tests] for i in items]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def manifest() -> dict:
    from ycode.lm.robust_tests_v1 import ROBUST_VERSION
    from ycode.lm.unseen import UNSEEN_VERSION
    from ycode.lm.unseen2 import UNSEEN2_VERSION

    out = {"eval_version": EVAL_VERSION, "robust_version": ROBUST_VERSION, "unseen_version": UNSEEN_VERSION,
           "unseen2_version": UNSEEN2_VERSION}
    for split in ("bench", "dev", "unseen", "unseen2"):
        items = eval_items(split)
        out[split] = {"items": len(items), "clean_items": sum(not i.contaminated for i in items),
                      "fingerprint": fingerprint(items),
                      "contaminated": sorted(f"{i.suite}:{i.name}" for i in items if i.contaminated)}
    return out


def robust_pass(item: EvalItem, code: str) -> bool:
    """Official tests and the robust tests both pass (one isolated subprocess)."""
    from ycode.lm.evaluate import run_tests

    return run_tests(code, item.tests + "\n" + item.robust_tests)


def score(answers: list[dict], split: str) -> dict:
    """Official, clean and robust scores of one answers file (rows from ``eval --answers-out``),
    each with its denominator, plus the per-item verdicts."""
    items = {(i.suite, i.name): i for i in eval_items(split)}
    rows = {(a["suite"], a["name"]): a for a in answers if (a["suite"], a["name"]) in items}
    per_item = []
    for key, it in items.items():
        a = rows.get(key)
        official = bool(a and a["pass"])
        robust = official and robust_pass(it, a.get("code", ""))
        per_item.append({"suite": it.suite, "name": it.name, "official": official, "robust": robust,
                         "contaminated": it.contaminated, "answered": a is not None})
    answered = [p for p in per_item if p["answered"]]
    clean = [p for p in answered if not p["contaminated"]]
    return {
        "set": split, "eval_version": EVAL_VERSION, "fingerprint": fingerprint(list(items.values())),
        "official": sum(p["official"] for p in answered), "official_of": len(answered),
        "clean": sum(p["official"] for p in clean), "clean_of": len(clean),
        "robust": sum(p["robust"] for p in answered), "robust_of": len(answered),
        "robust_clean": sum(p["robust"] for p in clean),
        "weak_passes": sorted(f"{p['suite']}:{p['name']}" for p in answered if p["official"] and not p["robust"]),
        "items": per_item,
    }
