import ast
import copy
import signal
import sys
import time

sys.path.insert(0, "/home/user/Code-agent-next")
import ycode.lm.compose as compose  # noqa: E402
from ycode.lm.basics import CONCEPTS, _passes  # noqa: E402
from ycode.lm.leakage import (  # noqa: E402
    _SIG, HeldOutItem, _arity, agreement, held_out_items, normalized_body, probe_inputs,
)
from ycode.lm.unseen import unseen_problems, unseen_references  # noqa: E402

refs = unseen_references()
items = []
for p in unseen_problems():
    m = _SIG.search(p.prompt)
    items.append(HeldOutItem("unseen", p.name, _arity(m.group(2)), p.tests, refs[p.name]))
compose.INCLUDE_SKILLS = True
names = {n for c in CONCEPTS for n in (c.name, *c.aliases)}
fns = {}
for c in CONCEPTS:
    for n in (c.name, *c.aliases):
        fns[c.source(n)] = (n, len([q for q in c.params.split(",") if q.strip()]))
for seed in (7, 1055, 1101, 1201, 1202, 1203):
    for e in compose._raw_exercises(seed) + compose.novel_exercises(seed) + compose.skill_exercises():
        names.add(e.name)
        for x in (e, compose.loop_rendering(e)):
            if x is not None:
                fns[x.source()] = (x.name, len([q for q in x.params.split(",") if q.strip()]))
by_body = {}
for src, v in fns.items():
    by_body.setdefault((normalized_body(src), v[1]), (src, v))
fns = [(src, v) for src, v in by_body.values()]
for i in held_out_items(include_unseen=False):  # unseen must not duplicate a benchmark or dev item either
    fns.append((i.reference, (f"{i.split}:{i.name}", i.arity)))
print(len(items), "unseen items,", len(fns), "candidate functions", flush=True)
print("name clashes with training:", sorted({i.name for i in items} & names), flush=True)


def first_call(item):
    for node in ast.walk(ast.parse(item.tests)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == item.name:
            return tuple(ast.literal_eval(a) for a in node.args)


def alarm(*_):
    raise TimeoutError


signal.signal(signal.SIGALRM, alarm)


def call(f, args, t=0.05):
    signal.setitimer(signal.ITIMER_REAL, t)
    try:
        return ("ok", repr(f(*copy.deepcopy(args))))
    except BaseException as e:  # noqa: BLE001
        return ("err", type(e).__name__)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


first = {}
for it in items:
    ns = {}
    exec(it.reference, ns)
    args = first_call(it)
    first[it.name] = (args, call(ns[it.name], args, 1.0))
t0 = time.time()
flagged = []
for src, (fname, ar) in fns:
    ns = {}
    defined = src.split("def ", 1)[1].split("(", 1)[0].strip()
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.5)
        exec(src, ns)
        signal.setitimer(signal.ITIMER_REAL, 0)
        f = ns[defined]
    except BaseException:  # noqa: BLE001
        signal.setitimer(signal.ITIMER_REAL, 0)
        continue
    for it in items:
        if it.arity != ar:
            continue
        args, expected = first[it.name]
        if call(f, args) != expected:
            continue
        if not _passes(f"{src}\n\n{it.name} = {defined}", it.tests, timeout=0.5):
            continue
        a = agreement(src, defined, it, probe_inputs(it), isolate=True)
        if a >= 0.9:
            flagged.append((it.name, fname, round(a, 3)))
            print(f"FLAG {a:.3f}: unseen {it.name} <- {fname}", flush=True)
print(f"checked in {time.time() - t0:.0f}s; flagged items:", sorted({f[0] for f in flagged}))
print("done")
