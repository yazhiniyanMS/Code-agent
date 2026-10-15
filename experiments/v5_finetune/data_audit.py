"""Audit one round's fine-tuning data: duplicates, contradictions, wrong answers, repetition, noise.

usage: data_audit.py [SEED]   (builds the round's examples exactly like build_round.py, without writing them)
"""
import json
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/home/user/Code-agent")
import ycode.lm.compose as compose  # noqa: E402
from ycode.lm.basics import (  # noqa: E402
    BUGFIX_PROMPT, CONCEPTS, _passes, _rename_tests, basics_examples, training_concepts,
)
from ycode.lm.evaluate import extract_code  # noqa: E402

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1060
BUG_HEAD = BUGFIX_PROMPT.split("```")[0]


def task_of(prompt: str) -> str:
    """The prompt with the function name replaced, so renamed copies of one task compare equal."""
    return re.sub(r"`[A-Za-z_]\w*(?=[(`])", "`F", prompt)


def body_of(code: str) -> str:
    lines = code.strip().splitlines()
    body = [ln for ln in lines[1:] if not ln.strip().startswith('"""')]
    return "\n".join(body).strip()


def main() -> None:
    parts = {
        "basics": basics_examples(SEED, write_per_concept=6, bugfix_per_concept=4),
        "compose_write": compose.compose_examples(SEED, per_exercise=3),
        "compose_bugfix": compose.compose_bugfix_examples(SEED, per_exercise=1),
        "contrast": compose.contrast_examples(SEED),
        "novel": compose.novel_examples(SEED),
    }
    examples = [(src, ex) for src, exs in parts.items() for ex in exs]
    report = {"seed": SEED, "examples": {k: len(v) for k, v in parts.items()}, "total": len(examples)}

    # 1. exact duplicates
    pairs = Counter((ex.prompt, ex.response) for _, ex in examples)
    report["exact_duplicates"] = sum(n - 1 for n in pairs.values() if n > 1)

    # 2. contradictions: same task text (name masked), different answer bodies
    by_task = defaultdict(set)
    for _, ex in examples:
        if not ex.prompt.startswith(BUG_HEAD):
            by_task[task_of(ex.prompt)].add(body_of(extract_code(ex.response)))
    contradictions = {t: sorted(b) for t, b in by_task.items() if len(b) > 1}
    report["write_tasks"] = len(by_task)
    report["contradictory_tasks"] = len(contradictions)
    report["contradiction_examples"] = [{"task": t[:140], "bodies": b[:3]} for t, b in list(contradictions.items())[:8]]

    # 3. answers that fail their own tests (basics writes: concept tests; compose: exercise tests)
    concept_tests = {}
    for c in CONCEPTS:
        for n in (c.name, *c.aliases):
            concept_tests[n] = (c.tests, n)
    exercise_tests = {}
    for e in compose.all_exercises(SEED) + compose.novel_exercises(SEED):
        if (t := e.tests()) is not None:
            exercise_tests.setdefault(e.name, t)
    checked = wrong = 0
    wrong_examples = []
    for src, ex in examples:
        code = extract_code(ex.response)
        m = re.match(r"def (\w+)\(", code)
        if not m:
            continue
        name = m[1]
        if name in concept_tests:
            tests = _rename_tests(concept_tests[name][0], name)
        elif name in exercise_tests:
            tests = exercise_tests[name]
        else:
            continue  # renamed to a neutral name: its source concept is checked under its own name
        checked += 1
        if not _passes(code, tests):
            wrong += 1
            wrong_examples.append({"source": src, "prompt": ex.prompt[:120], "code": code[:200]})
    report["answers_checked"], report["answers_failing_own_tests"] = checked, wrong
    report["wrong_examples"] = wrong_examples[:5]

    # 4. repetition: examples per distinct answer body (bodies the model sees most often)
    bodies = Counter(body_of(extract_code(ex.response)) for _, ex in examples)
    report["distinct_answer_bodies"] = len(bodies)
    report["most_repeated_bodies"] = [{"body": b[:80], "examples": n} for b, n in bodies.most_common(12)]
    per_concept = Counter()
    names = {n: c.name for c in training_concepts() for n in (c.name, *c.aliases)}
    for ex in parts["basics"]:
        m = re.match(r"def (\w+)\(", extract_code(ex.response))
        if m and m[1] in names:
            per_concept[names[m[1]]] += 1
    report["basics_examples_per_concept"] = {"mean": round(sum(per_concept.values()) / max(1, len(per_concept)), 1),
                                             "max": max(per_concept.values(), default=0)}
    one_liners = sum(1 for _, ex in examples if len(body_of(extract_code(ex.response)).splitlines()) <= 1)
    report["one_line_answers_share"] = round(one_liners / len(examples), 3)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
