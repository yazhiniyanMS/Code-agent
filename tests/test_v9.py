"""V9-1: grammar v2 takes its parts from the training families only, trains nothing a test task
computes, and holds out combinations of trained parts."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _split():
    return json.loads((ROOT / "ycode" / "lm" / "v9_grammar_split.json").read_text())


def test_every_part_comes_from_a_training_family():
    from ycode.lm import compositions2 as G

    transfer = json.loads((ROOT / "ycode" / "lm" / "v8_transfer_split.json").read_text())
    sources = {s[-1] for s in G.SELECT} | {t[-1] for t in G.TRANSFORM} | {a[-1] for a in G.AGGREGATE}
    assert sources <= set(transfer["train_families"])
    assert not sources & set(transfer["held_out_families"])


def test_split_holds_out_new_combinations_of_trained_parts():
    from ycode.lm import compositions2 as G
    from ycode.lm import rounddata

    split = _split()
    train, test = set(split["train"]), set(split["test"])
    assert len(train) == 350 and len(test) == 60 and not train & test and set(split["trained_sample"]) <= train
    by_name = {p.name: p for p in G.all_combinations()}
    for domain in G.DOMAINS:
        parts = set().union(*(G._parts(by_name[n]) for n in train if G.PARTS[n][0] == domain))
        assert all(G._parts(by_name[n]) <= parts for n in test if G.PARTS[n][0] == domain)
    programs = rounddata._curriculum_programs({**rounddata.DEFAULTS, "idioms_select": "v9_grammar_train"})
    assert {p.name for p in programs} == train
    for name in ("c2_test", "c2_trained"):
        tasks = json.loads((ROOT / "experiments" / "v9" / "sets" / f"{name}.json").read_text())["tasks"]
        assert all(len(t["probe_args"]) == 40 for t in tasks)
    assert {t["name"] for t in json.loads((ROOT / "experiments" / "v9" / "sets" / "c2_test.json").read_text())["tasks"]} == test


def test_no_trained_or_held_out_combination_leaks():
    """Training and test combinations: no held-out item (name, skeleton, behaviour), and no training
    program computes a cross-family, held-out-family, within-family or neighbour task."""
    from ycode.lm import compositions2 as G
    from ycode.lm import idioms
    from ycode.lm.leakage import skeleton

    split = _split()
    by_name = {p.name: p for p in G.all_combinations()}
    used = [by_name[n] for n in split["train"] + split["test"]]
    kept, dropped = idioms.held_out_check(used)
    assert not dropped, dropped
    avoid = idioms.neighbour_skeletons()
    assert not any(skeleton(p.source(i)) in avoid for p in used for i in range(2))
    tasks = G.test_tasks()
    clashes = []
    for p in (by_name[n] for n in split["train"]):
        ns: dict = {}
        exec(p.source(0), ns)
        clashes += [(s, n, p.name) for s, n, args, want in tasks if G._computes(ns[p.name], args, want)]
    assert not clashes, clashes
    assert split["report"]["expressible_t_families"]  # recorded before training (an upper bound)
