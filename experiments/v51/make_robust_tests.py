"""Generate ycode/lm/robust_tests_v1.py: independent tests for every benchmark, dev and unseen item.

The official tests of some items are too short (fibonacci is checked at 0, 1 and 10, so
n * (n + 1) // 2 passes; first_word passes with text.split()[-2]). The official tests and scores
stay as they are; this suite adds, per item, assertions computed by running the item's reference
solution on extra inputs, plus hand-written edge cases for the items known to be weak. A "robust"
pass is a pass of the official tests and of these.

Extra inputs stay inside the domain the official tests use (non-negative numbers stay
non-negative, numbers stay within twice the largest official value, non-empty lists and strings stay
non-empty), so a correct solution is never failed for an input the task does not promise to handle
or for being too slow on an input far larger than the task's own. Inputs on which the reference raises are skipped.

usage: python experiments/v51/make_robust_tests.py   (rewrites the module; deterministic)
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ycode.lm.leakage import _kind, held_out_items  # noqa: E402

PER_ITEM = 10  # extra assertions per item (besides the hand-written ones)

# Edge cases for items whose official tests accept known wrong code.
HAND = {
    "first_word": ["assert first_word('The quick brown fox') == 'The'", "assert first_word('a b') == 'a'",
                   "assert first_word('single') == 'single'", "assert first_word('  padded words ') == 'padded'"],
    "fibonacci": ["assert fibonacci(2) == 1", "assert fibonacci(3) == 2", "assert fibonacci(5) == 5",
                  "assert fibonacci(7) == 13", "assert fibonacci(12) == 144"],
    "second_largest": ["assert second_largest([1, 2, 3]) == 2", "assert second_largest([10, 5, 10, 8]) == 8",
                       "assert second_largest([-1, -5, -3]) == -3"],
    "maximum": ["assert maximum([9, 1, 2]) == 9", "assert maximum([1, 2, 9]) == 9", "assert maximum([-3, -1, -2]) == -1"],
    "minimum": ["assert minimum([1, 9, 2]) == 1", "assert minimum([9, 2, 1]) == 1", "assert minimum([-3, -1, -2]) == -3"],
    "is_between": ["assert is_between(3, 1, 3)", "assert not is_between(0, 1, 3)", "assert is_between(5, 5, 5)",
                   "assert not is_between(6, 1, 5)"],
}


def _domain_ok(args: tuple, calls: list[tuple]) -> bool:
    """True if every argument looks like the official tests' arguments in the same position."""
    for pos, value in enumerate(args):
        kind = _kind(value)
        seen = [c[pos] for c in calls if _kind(c[pos]) == kind]  # same-kind official values only
        if not seen:
            return False
        if kind in (int, float):
            if min(seen) >= 0 and value < 0:
                return False
            if abs(value) > max(12, 2 * max(abs(v) for v in seen)):
                return False  # keep the scale of the official tests: fibonacci(194) times out a correct naive solution
            if all(isinstance(v, int) for v in seen) and isinstance(value, float):
                return False
        elif hasattr(value, "__len__") and all(len(v) > 0 for v in seen) and len(value) == 0:
            return False
    return True


def _official_calls(item) -> list[tuple]:
    calls = []
    for node in ast.walk(ast.parse(item.tests)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == item.name and not node.keywords:
            try:
                args = tuple(ast.literal_eval(a) for a in node.args)
            except (ValueError, SyntaxError):
                continue
            if len(args) == item.arity:
                calls.append(args)
    return calls


def _expected(reference: str, name: str, args: tuple):
    import copy
    import signal

    ns: dict = {}
    exec(reference, ns)

    def alarm(*_):
        raise TimeoutError

    old = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, 1.0)
    try:
        return ("ok", ns[name](*copy.deepcopy(args)))
    except BaseException:  # noqa: BLE001 - outside the reference's domain, or too slow
        return ("err", None)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _assertion(name: str, args: tuple, value) -> str:
    call = f"{name}({', '.join(repr(a) for a in args)})"
    if isinstance(value, float):
        return f"assert abs({call} - {value!r}) < 1e-9"
    if value is None:
        return f"assert {call} is None"
    return f"assert {call} == {value!r}"


def _candidates(item, calls: list[tuple]) -> list[tuple]:
    """Inputs inside the official tests' domain: per position, the official values and the typed pool
    values that fit the domain, combined; then seeded random values that fit it."""
    import itertools
    import random

    from ycode.lm.leakage import _POOLS, _random_value

    pools = []
    for pos in range(item.arity):
        values = []
        kinds_here = sorted({_kind(c[pos]) for c in calls}, key=str)  # sorted: set order of types varies by process
        for v in [c[pos] for c in calls] + [x for kind in kinds_here for x in _POOLS.get(kind, [])]:
            if v not in values and _domain_ok(tuple(v if i == pos else calls[0][i] for i in range(item.arity)), calls):
                values.append(v)
        pools.append(values)
    per_arg = max(2, round(60 ** (1 / max(1, item.arity))))
    out = [p for p in itertools.product(*(pool[:per_arg] for pool in pools)) if p not in calls]
    rng = random.Random(item.name)
    kinds = [sorted({_kind(c[pos]) for c in calls}, key=str) for pos in range(item.arity)]
    for _ in range(2000):
        if len(out) >= 140:
            break
        args = tuple(_random_value(rng.choice(k), rng) for k in kinds)
        if args not in out and args not in calls and _domain_ok(args, calls):
            out.append(args)
    return out


def extra_tests(item) -> list[str]:
    calls = _official_calls(item)
    picked, outputs = [], []
    candidates = _candidates(item, calls)
    results = [(p, _expected(item.reference, item.name, p)) for p in candidates]
    results = [(p, v) for p, (status, v) in results if status == "ok" and len(repr(v)) < 200 and len(repr(p)) < 200]
    # first, inputs whose expected outputs are not yet covered (most discriminating), then the rest
    for p, v in results:
        if len(picked) >= PER_ITEM:
            break
        if repr(v) not in outputs:
            picked.append((p, v))
            outputs.append(repr(v))
    for p, v in results:
        if len(picked) >= PER_ITEM:
            break
        if (p, v) not in picked:
            picked.append((p, v))
    return list(dict.fromkeys(HAND.get(item.name, []) + [_assertion(item.name, p, v) for p, v in picked]))


def main() -> None:
    from ycode.lm.unseen import UNSEEN_VERSION

    items = held_out_items(include_unseen=True)
    lines = ['"""Robust tests v1 (generated by experiments/v51/make_robust_tests.py: do not edit by hand).',
             "",
             "Independent assertions per held-out item, computed from its reference solution on inputs inside",
             "the official tests' domain, plus hand-written edge cases for items with weak official tests.",
             "A robust pass = official tests + these. The official tests and scores are unchanged.",
             '"""', "", 'ROBUST_VERSION = "robust-v1"', f'UNSEEN_VERSION = "{UNSEEN_VERSION}"', "",
             "ROBUST_TESTS: dict[tuple[str, str], str] = {"]
    for item in sorted(items, key=lambda i: (i.split, i.name)):
        tests = extra_tests(item)
        lines.append(f"    ({item.split!r}, {item.name!r}): (")
        for t in tests:
            lines.append(f"        {t + chr(10)!r}")
        lines.append("    ),")
    lines.append("}")
    out = ROOT / "ycode" / "lm" / "robust_tests_v1.py"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}: {len(items)} items")


if __name__ == "__main__":
    main()
