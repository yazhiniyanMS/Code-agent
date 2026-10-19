"""V8 stage 2: a balanced curriculum split for testing transfer rather than memorization.

usage: transfer_split.py        (writes ycode/lm/v8_transfer_split.json and experiments/v8/sets/t_*.json)

From the held-out-clean programs of the 215 curriculum families (idioms v3), without any program
whose skeleton matches V7's neighbour set:

* held-out families (``t_families``): about 30% of the families, stratified by source (v1 / v2 / v3)
  and level (1: one idiom, 2: several steps, 3: algorithms), never trained. Their programs are the
  cross-family test; a program whose AST skeleton equals a training program's is dropped (counted).
* training families: the rest (the 12 constant-parameterised v1 families of L2's curriculum are always
  among them, so experiments/v8/sets/family_new.json stays a within-family test). Every family with
  several behaviours keeps one variant out (``t_within``: same family, another constant, condition
  or direction). The remaining behaviours, at most 2 per family, are the
  training programs (``t_trained``), with all their names (aliases) and phrasings.

Balance: siblings (similar names, different behaviour), aliases and renamed copies (same behaviour,
other names), boundary cases (every program's inputs include empty and single-item cases) and
multi-step programs (levels 2-3) are all present; the split file lists the counts.
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "v8"))

from build_sets import distinct, program_tasks  # noqa: E402

from ycode.lm import idioms  # noqa: E402
from ycode.lm.leakage import skeleton  # noqa: E402

HELD_OUT_SHARE = 0.3
PER_FAMILY = 2
SEED = 8


def main() -> None:
    programs, _ = idioms.held_out_check(idioms.candidate_programs(family_list=idioms.families("v3")))
    avoid = idioms.neighbour_skeletons()
    programs = [p for p in programs if not any(skeleton(p.source(i)) in avoid for i in range(len(p.bodies)))]
    fams = {f.key: f for f in idioms.families("v3")}
    k_keys = {p.family for p in idioms.k_family_programs("v1")}
    v1 = set(idioms.FAMILIES_V1)
    v2 = {f.key for f in idioms.FAMILIES[len(idioms.FAMILIES_V1):len(idioms.families("v2"))]}
    source = lambda k: "v1" if k in v1 else "v2" if k in v2 else "v3"  # noqa: E731
    by_family: dict[str, list] = defaultdict(list)
    for p in programs:
        by_family[p.family].append(p)
    behaviours = {k: distinct(ps) for k, ps in by_family.items()}

    rng = random.Random(SEED)
    strata: dict[tuple, list[str]] = defaultdict(list)
    for k in sorted(behaviours):
        if k not in k_keys:
            strata[(source(k), fams[k].level)].append(k)
    held = set()
    for key in sorted(strata):
        ks = strata[key]
        rng.shuffle(ks)
        held |= set(ks[:round(HELD_OUT_SHARE * len(ks))])
    train_fams = sorted(set(behaviours) - held)

    trained, within = [], []
    for k in train_fams:
        bs = sorted(behaviours[k], key=lambda p: p.name)
        if len(bs) > 1:
            out = rng.choice(bs)
            within.append(out)
            bs = [b for b in bs if b is not out]
        rng.shuffle(bs)
        trained += bs[:PER_FAMILY]
    train_skeletons = {skeleton(q.source(i)) for p in trained for q in by_family[p.family]
                       if q.source(0, "f") == p.source(0, "f") for i in range(len(q.bodies))}
    cross, dropped = [], 0
    for k in sorted(held):
        for p in behaviours[k]:
            if any(skeleton(p.source(i)) in train_skeletons for i in range(len(p.bodies))):
                dropped += 1
                continue
            cross.append(p)
    # family, a name and the behaviour (the program's code renamed to f): a name alone can cover two variants
    # that only the task text tells apart
    names = lambda ps: [[p.family, p.name, p.source(0, "f")] for p in ps]  # noqa: E731
    level = lambda ps: {lv: sum(fams[p.family].level == lv for p in ps) for lv in (1, 2, 3)}  # noqa: E731
    split = {"version": "v8-transfer-1", "seed": SEED, "held_out_share": HELD_OUT_SHARE, "per_family": PER_FAMILY,
             "train_families": train_fams, "held_out_families": sorted(held),
             "trained": names(trained), "within": names(within), "cross": names(cross),
             "counts": {"train_families": len(train_fams), "held_out_families": len(held),
                        "trained_behaviours": len(trained), "within_tests": len(within), "cross_tests": len(cross),
                        "cross_dropped_skeleton_twin_of_a_trained_program": dropped,
                        "trained_by_level": level(trained), "cross_by_level": level(cross),
                        "trained_with_aliases": sum(len(fams[p.family].names) > 1 for p in trained),
                        "trained_with_a_trained_sibling": sum(
                            sum(q.family == p.family for q in trained) > 1 for p in trained)}}
    (ROOT / "ycode" / "lm" / "v8_transfer_split.json").write_text(json.dumps(split, indent=1))
    out = ROOT / "experiments" / "v8" / "sets"
    siblings = [q for ps in by_family.values() for q in distinct(ps)]
    for name, ps in (("t_trained", trained), ("t_within", within), ("t_families", cross)):
        tasks = program_tasks(ps, name, siblings)
        (out / f"{name}.json").write_text(json.dumps({"version": f"{name}-v1", "variants": ["original"], "probes": 40,
                                                      "tasks": tasks}, indent=1))
    print(json.dumps(split["counts"], indent=1))


if __name__ == "__main__":
    main()
