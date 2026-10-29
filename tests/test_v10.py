"""V10: fine-tuning from a compact checkpoint (C2) keeps compact checkpoints valid, and kept models
can live outside the run directory."""

import json

import pytest
import torch

from tests.test_v51 import _cfg
from ycode.lm.data import InstructionExample


@pytest.fixture
def base_model(data_dir, tmp_path):
    from ycode.lm.train import train

    train(data_dir, tmp_path / "base", _cfg(max_steps=2), log=lambda *_: None)
    return tmp_path / "base"


@pytest.fixture
def sft(data_dir, tmp_path):
    from ycode.lm.basics import write_basics_dataset

    examples = [InstructionExample(f"Write `f{i}(x)`.", f"```python\ndef f{i}(x):\n    return x + {i}\n```") for i in range(8)]
    write_basics_dataset(tmp_path / "sft", data_dir / "tokenizer.json", examples=examples * 3, seed=0, log=lambda *_: None)
    return tmp_path / "sft"


def _finetune(sft, init, out, layers):
    from ycode.lm.train import TrainConfig, train

    train(sft, out, TrainConfig(stage="sft", init_from=str(init), max_steps=2, batch_size=2, warmup_steps=1,
                                eval_iters=0, eval_interval=1000, device="cpu", precision="fp32", lr=1e-3,
                                train_layers=layers, compact_checkpoints=True, final_bf16=True),
          log=lambda *_: None)
    return out


def test_the_final_weights_only_save_frees_the_optimizer_state_first(base_model, sft, tmp_path, monkeypatch):
    from ycode.lm import train as T

    optimizers, saves = [], []
    build, save = T.build_optimizer, T.save_checkpoint
    monkeypatch.setattr(T, "build_optimizer", lambda *a, **k: optimizers.append(build(*a, **k)) or optimizers[-1])
    monkeypatch.setattr(T, "save_checkpoint", lambda *a, **k: saves.append(
        (k["step"], k["optimizer"] is None, sum(len(o.state) for o in optimizers[-1].optimizers))) or save(*a, **k))

    def run(out, final_bf16):
        T.train(sft, out, T.TrainConfig(stage="sft", init_from=str(base_model), max_steps=2, batch_size=2,
                                        warmup_steps=1, eval_iters=0, eval_interval=1000, save_interval=1,
                                        device="cpu", precision="fp32", lr=1e-3, train_layers=1,
                                        compact_checkpoints=True, final_bf16=final_bf16), log=lambda *_: None)
        return torch.load(out / "model.pt", map_location="cpu", weights_only=True)["model"]

    weights = run(tmp_path / "bf16", True)
    assert saves[0][0] == 1 and not saves[0][1] and saves[0][2] > 0  # an intermediate save keeps the optimizer
    assert saves[-1] == (2, True, 0)  # the final weights-only save: momentum freed before it
    reference = run(tmp_path / "fp32", False)  # the same run, saved with its optimizer
    assert saves[-1][2] > 0
    assert set(weights) == set(reference)
    assert all(torch.equal(weights[k], reference[k].to(torch.bfloat16)) for k in weights)


def test_a_run_from_a_compact_checkpoint_references_the_plain_base(base_model, sft, tmp_path):
    from ycode.lm.train import load_checkpoint

    first = _finetune(sft, base_model, tmp_path / "first", 1)
    second = _finetune(sft, first, tmp_path / "second", 1)  # trains what `first` trained: base stands in
    payload = torch.load(second / "model.pt", map_location="cpu", weights_only=True)
    assert payload["base"]["path"] == str(base_model.resolve())
    base = load_checkpoint(base_model)[0].state_dict()
    one = load_checkpoint(first)[0].state_dict()
    two = load_checkpoint(second)[0].state_dict()
    stored = set(payload["model"])
    assert stored == set(torch.load(first / "model.pt", map_location="cpu", weights_only=True)["model"])
    for k, v in two.items():
        if k not in stored:  # every tensor neither run trained is the base's, in both runs
            assert torch.equal(v, base[k]) and torch.equal(one[k], base[k])
    assert any(not torch.equal(two[k], one[k]) for k in stored)  # the second run did train


def test_a_run_that_trains_less_than_its_compact_start_cannot_reference_the_base(base_model, sft, tmp_path):
    first = _finetune(sft, base_model, tmp_path / "first", 2)
    with pytest.raises(ValueError, match="does not train"):
        _finetune(sft, first, tmp_path / "second", 1)


def test_kept_models_go_to_keep_dir(data_dir, base_model, tmp_path, monkeypatch):
    import shutil

    from ycode.lm import experiment, rounddata

    def fake_round(out_dir, seed, tokenizer_path, **kwargs):
        import numpy as np

        shutil.copytree(data_dir, out_dir, dirs_exist_ok=True)
        n = len(np.load(out_dir / "sft_index.npy")) if (out_dir / "sft_index.npy").exists() else 0
        (out_dir / "sft_tags.json").write_text(json.dumps(["legacy:write"] * n))
        manifest = {"seed": seed, "options": kwargs.get("options") or {}}
        (out_dir / "manifest.json").write_text(json.dumps(manifest))
        (out_dir / "hygiene.json").write_text("[]")
        return manifest

    monkeypatch.setattr(rounddata, "write_round_dataset", fake_round)
    spec = {"name": "tiny10", "anchor": str(base_model), "anchor_sha256": "x", "tokenizer": str(data_dir / "tokenizer.json"),
            "out": str(tmp_path / "exp"), "work_dir": str(tmp_path / "ram"), "keep_models": "compact",
            "keep_dir": str(tmp_path / "keep"), "seeds": [1], "metric": "unseen_clean",
            "train": {"steps": 1, "precision": "fp32", "train_layers": 1, "init_optim": False, "compact": True,
                      "final_bf16": True, "save_every": 1},
            "eval": {"limit": 1, "diagnose": 0, "bench": False, "dev": False, "unseen": False, "sets": []},
            "arms": [{"name": "C"}], "comparisons": []}
    record = experiment.run_one(spec, spec["arms"][0], 1, tmp_path / "exp", log=lambda *_: None)
    kept = tmp_path / "keep" / "C" / "1" / "model_bf16" / "model.pt"
    assert kept.is_file() and record["kept_model"]["path"] == str(kept)
    assert not (tmp_path / "exp" / "C" / "1" / "model_bf16").exists()


def test_runner_starts_from_another_checkpoint_with_other_examples(data_dir, base_model, sft, tmp_path, monkeypatch):
    import shutil

    from ycode.lm import experiment, rounddata

    start = _finetune(sft, base_model, tmp_path / "start", 1)  # a compact checkpoint, like C2
    seen = {}

    def fake_round(out_dir, seed, tokenizer_path, **kwargs):
        shutil.copytree(sft, out_dir, dirs_exist_ok=True)
        seen["seed"] = seed
        manifest = {"seed": seed, "options": kwargs.get("options") or {}}
        (out_dir / "manifest.json").write_text(json.dumps(manifest))
        (out_dir / "hygiene.json").write_text("[]")
        return manifest

    monkeypatch.setattr(rounddata, "write_round_dataset", fake_round)
    spec = {"name": "tiny10b", "anchor": str(base_model), "anchor_sha256": "x", "tokenizer": str(data_dir / "tokenizer.json"),
            "out": str(tmp_path / "exp"), "work_dir": str(tmp_path / "ram"), "keep_models": "compact",
            "keep_dir": str(tmp_path / "keep"), "seeds": [1], "metric": "unseen_clean",
            "train": {"steps": 1, "precision": "fp32", "train_layers": 1, "init_optim": False, "compact": True,
                      "final_bf16": True, "save_every": 1, "schedule": "cosine"},
            "eval": {"limit": 1, "diagnose": 0, "bench": False, "dev": False, "unseen": False, "sets": []},
            "arms": [{"name": "R", "init_from": str(start), "data_seed_offset": 5}], "comparisons": []}
    record = experiment.run_one(spec, spec["arms"][0], 1, tmp_path / "exp", log=lambda *_: None)
    assert seen["seed"] == 6 and record["init_from"] == str(start)
    kept = torch.load(record["kept_model"]["path"], map_location="cpu", weights_only=True)
    assert kept["base"]["path"] == str(base_model.resolve())  # the plain base, not the compact start
    log = (tmp_path / "exp" / "R" / "1" / "train.log").read_text()
    assert str(start) in log or "cosine" in json.dumps(record["train"])


def test_packing_a_checkpoint_restores_every_tensor_bit_for_bit(base_model, sft, tmp_path):
    from ycode.lm import packckpt
    from ycode.lm.train import load_checkpoint

    ft = _finetune(sft, base_model, tmp_path / "ft", 1)
    packed = tmp_path / "ft.ypk"
    packckpt.main(["pack", str(ft), str(packed)])
    assert packckpt.verify(ft, packed)
    out = packckpt.unpack(packed, tmp_path / "restored")
    a = torch.load(ft / "model.pt", map_location="cpu", weights_only=True)
    b = torch.load(out, map_location="cpu", weights_only=True)
    assert packckpt.content_sha256(a["model"]) == packckpt.content_sha256(b["model"])
    assert all(torch.equal(a["model"][k].view(torch.int16) if a["model"][k].dtype == torch.bfloat16 else a["model"][k],
                           b["model"][k].view(torch.int16) if b["model"][k].dtype == torch.bfloat16 else b["model"][k])
               for k in a["model"])
    m1, m2 = load_checkpoint(ft)[0].state_dict(), load_checkpoint(tmp_path / "restored")[0].state_dict()
    assert all(torch.equal(m1[k], m2[k]) for k in m1)


def test_grammar3_parts_come_from_training_families_and_nothing_leaks():
    from pathlib import Path

    from ycode.lm import compositions2 as G2
    from ycode.lm import compositions3 as G
    from ycode.lm import idioms, rounddata
    from ycode.lm.leakage import skeleton

    root = Path(__file__).resolve().parents[1]
    transfer = json.loads((root / "ycode" / "lm" / "v8_transfer_split.json").read_text())
    sources = {s[-1] for s in G.SOURCE} | {v[-1] for v in G.VIEW} | {a[-1] for a in G.AGGREGATE}
    assert sources <= set(transfer["train_families"]) and not sources & set(transfer["held_out_families"])
    split = json.loads((root / "ycode" / "lm" / "v10_grammar3_split.json").read_text())
    train, test = set(split["train"]), set(split["test"])
    assert not train & test and set(split["trained_sample"]) <= train
    by_name = {p.name: p for p in G.all_combinations()}
    for src in G.SOURCE:
        parts = set().union(*(G._parts(by_name[n]) for n in train if G.PARTS[n][0] == src[0]))
        assert all(G._parts(by_name[n]) <= parts for n in test if G.PARTS[n][0] == src[0])
    programs = rounddata._curriculum_programs({**rounddata.DEFAULTS, "idioms_select": "v10_grammar3_train"})
    assert {p.name for p in programs} == train
    used = [by_name[n] for n in split["train"] + split["test"]]
    assert not idioms.held_out_check(used)[1]
    avoid = idioms.neighbour_skeletons()
    assert not any(skeleton(p.source(i)) in avoid for p in used for i in range(2))
    tasks = G2.test_tasks()
    clashes = []
    for p in (by_name[n] for n in split["train"]):
        ns: dict = {}
        exec(p.source(0), ns)
        clashes += [(s, n, p.name) for s, n, args, want in tasks if G2._computes(ns[p.name], args, want)]
    assert not clashes, clashes
    for name in ("c3_test", "c3_trained"):
        sets = json.loads((root / "experiments" / "v10" / "sets" / f"{name}.json").read_text())["tasks"]
        assert len(sets) in (60, 40) and all(len(t["probe_args"]) == 40 for t in sets)


def test_minimal_pairs_differ_in_exactly_one_part_and_in_behaviour():
    import re

    from ycode.lm import compositions2 as G
    from ycode.lm import idioms, rounddata

    programs = rounddata._curriculum_programs({**rounddata.DEFAULTS, "idioms_select": "v9_grammar_train"})[:120]
    by_code = {}
    for p in programs:
        for i in range(2):
            by_code[p.source(i)] = p
    examples = G.minimal_pair_examples(3, programs, per_program=1, weight=20)
    assert len(examples) > 100 and all(e.contrast_weight == 20 for e in examples)
    for e in examples:
        code = re.search(r"```python\n(.*?)\n```", e.response, re.S).group(1)
        name = re.match(r"def (\w+)\(", code).group(1)
        p = next(q for src, q in by_code.items() if src.replace(f"def {q.name}(", f"def {name}(") == code)
        sibling = next(q for src, q in by_code.items() if src.replace(f"def {q.name}(", f"def {name}(") == e.contrast)
        assert sum(a != b for a, b in zip(G.PARTS[p.name], G.PARTS[sibling.name])) == 1
        assert idioms._run(code, name, list(p.inputs)) != idioms._run(e.contrast, name, list(p.inputs))


def test_retention_aware_replay_draws_by_weight_at_the_same_token_share(data_dir, monkeypatch, tmp_path):
    from ycode.lm import rounddata

    production = [InstructionExample(f"Write `p{i}(x)`.", f"```python\ndef p{i}(x):\n    return x - {i}\n```") for i in range(200)]
    monkeypatch.setattr(rounddata, "_legacy_examples", lambda seed, o: production)
    weights = [100.0 if i < 20 else 0.01 for i in range(200)]  # the first 20 were forgotten
    (tmp_path / "w.json").write_text(json.dumps({"seed": 1, "n": 200, "weights": weights}))
    o = {**rounddata.DEFAULTS, "legacy_share": 0.5, "legacy_weights": str(tmp_path / "w.json")}
    from ycode.lm.data import encode_example
    from ycode.lm.tokenizer import BPETokenizer

    tok = BPETokenizer.load(data_dir / "tokenizer.json")
    each = len(encode_example(tok, production[0], weighted=True)[0])
    picked, tags, _ = rounddata._legacy_sample(1, o, 40 * each, data_dir / "tokenizer.json", None, lambda *_: None)
    uniform, _, _ = rounddata._legacy_sample(1, {**o, "legacy_weights": None}, 40 * each, data_dir / "tokenizer.json",
                                             None, lambda *_: None)
    assert abs(len(picked) - len(uniform)) <= 2  # the same share of the round
    forgotten = {f"p{i}(" for i in range(20)}
    assert sum(any(f in e.prompt for f in forgotten) for e in picked) >= 18  # weighted: nearly all 20 drawn
    again = rounddata._legacy_sample(1, o, 40 * each, data_dir / "tokenizer.json", None, lambda *_: None)[0]
    assert [e.prompt for e in again] == [e.prompt for e in picked]  # seeded
    (tmp_path / "w.json").write_text(json.dumps({"seed": 2, "n": 200, "weights": weights}))
    with pytest.raises(ValueError, match="not seed 1"):
        rounddata._legacy_sample(1, o, 40 * each, data_dir / "tokenizer.json", None, lambda *_: None)

def test_retention_aware_replay_within_tag_keeps_each_kinds_share(data_dir, monkeypatch, tmp_path):
    """V11-C: weights scaled within each tag keep the bug-fix share of a uniform draw (forgetting
    scores weight write examples far more, which would otherwise crowd the bug fixes out)."""
    from ycode.lm import rounddata

    write = [InstructionExample(f"Write `p{i}(x)`.", f"```python\ndef p{i}(x):\n    return x - {i}\n```") for i in range(150)]
    bugs = [InstructionExample(f"This function has a bug. Find and fix it:\n```python\ndef q{i}(x):\n    return x + {i}\n```",
                               f"```python\ndef q{i}(x):\n    return x - {i}\n```") for i in range(50)]
    production = write + bugs
    monkeypatch.setattr(rounddata, "_legacy_examples", lambda seed, o: production)
    weights = [10.0] * 150 + [0.1 if i >= 5 else 1.0 for i in range(50)]  # bug fixes scored low; q0-q4 the most
    (tmp_path / "w.json").write_text(json.dumps({"seed": 1, "n": 200, "weights": weights}))
    o = {**rounddata.DEFAULTS, "legacy_share": 0.5, "legacy_weights": str(tmp_path / "w.json")}
    tok_path = data_dir / "tokenizer.json"
    from ycode.lm.data import encode_example
    from ycode.lm.tokenizer import BPETokenizer

    each = len(encode_example(BPETokenizer.load(tok_path), production[0], weighted=True)[0])
    by_score = rounddata._legacy_sample(1, o, 60 * each, tok_path, None, lambda *_: None)[1]
    within = rounddata._legacy_sample(1, {**o, "legacy_weights_within_tag": True}, 60 * each, tok_path, None,
                                      lambda *_: None)
    assert by_score.count("legacy:bugfix") <= 2  # by raw score the bug fixes are crowded out
    share = within[1].count("legacy:bugfix") / len(within[1])
    assert 0.12 <= share <= 0.40  # about their 25% share of the round
    drawn = {e.prompt.split("def ")[1][:3] for e in within[0] if e.prompt.startswith("This function")}
    assert sum(f"q{i}(" in drawn for i in range(5)) >= 3  # inside the kind, the most forgotten are preferred
