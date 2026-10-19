"""V8: exact exposure logging, source tags, replay sampled from the production mix, RAM-disk runs
with task sets scored on probe inputs."""

import json
from pathlib import Path

import pytest

from tests.test_v51 import _cfg
from ycode.lm.data import InstructionExample

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def base_model(data_dir, tmp_path):
    from ycode.lm.train import train

    train(data_dir, tmp_path / "base", _cfg(max_steps=2), log=lambda *_: None)
    return tmp_path / "base"


def _examples():
    cur = [InstructionExample(f"Write `f{i}(x)`.", f"```python\ndef f{i}(x):\n    return x + {i}\n```") for i in range(6)]
    legacy = [InstructionExample(f"Write `g{i}(s)`.", f"```python\ndef g{i}(s):\n    return s * {i}\n```") for i in range(4)]
    return cur + legacy, [f"curriculum:fam:f{i % 3}" for i in range(6)] + ["legacy:write"] * 4


def test_trainer_logs_exactly_the_tokens_it_trains_on(data_dir, base_model, tmp_path):
    from ycode.lm.basics import write_basics_dataset
    from ycode.lm.train import TrainConfig, load_checkpoint, train

    examples, tags = _examples()
    sft = tmp_path / "sft"
    write_basics_dataset(sft, data_dir / "tokenizer.json", examples=examples * 3, tags=tags * 3, seed=0, log=lambda *_: None)
    assert len(json.loads((sft / "sft_tags.json").read_text())) == 30
    out = tmp_path / "run"
    train(sft, out, TrainConfig(stage="sft", init_from=str(base_model), max_steps=3, batch_size=2, warmup_steps=1,
                                eval_iters=0, eval_interval=1000, device="cpu", precision="fp32", lr=1e-3),
          log=lambda *_: None)
    exp = json.loads((out / "exposures.json").read_text())
    block = load_checkpoint(base_model)[0].cfg.block_size
    assert sum(t for _, (t, _a) in exp["rows"]) == 3 * 2 * (block + 1)  # every trained token, once
    assert all(row >= exp["n_val"] and 0 <= a <= t for row, (t, a) in exp["rows"])


def test_source_tags_token_shares_and_exposure_summary(data_dir, tmp_path):
    import numpy as np

    from ycode.lm.basics import write_basics_dataset
    from ycode.lm.rounddata import exposure_summary, tokens_by_source

    examples, tags = _examples()
    sft = tmp_path / "sft"
    write_basics_dataset(sft, data_dir / "tokenizer.json", examples=examples, tags=tags, seed=3, log=lambda *_: None)
    by = tokens_by_source(sft)
    assert by["curriculum"]["examples"] == 6 and by["legacy"]["examples"] == 4
    assert abs(sum(g["token_share"] for g in by.values()) - 1) < 1e-3
    index = np.load(sft / "sft_index.npy")
    row_tags = json.loads((sft / "sft_tags.json").read_text())
    rows = [[i, [int(index[i][1]), 10 ** 6]] for i, t in enumerate(row_tags) if t == "curriculum:fam:f0"]  # seen whole
    (tmp_path / "exp.json").write_text(json.dumps({"n_val": 0, "rows": rows}))
    s = exposure_summary(sft, tmp_path / "exp.json")
    assert s["curriculum_behaviours"] == 3 and s["exposures_per_behaviour"]["never_seen"] == 2
    assert s["exposures_per_behaviour"]["max"] >= 2  # f0 has two examples (i = 0, 3)


def test_legacy_share_samples_the_production_round(data_dir, monkeypatch):
    from ycode.lm import rounddata

    production = [InstructionExample(f"Write `p{i}(x)`.", f"```python\ndef p{i}(x):\n    return x - {i}\n```") for i in range(200)]
    monkeypatch.setattr(rounddata, "_legacy_examples", lambda seed, o: production)
    o = {**rounddata.DEFAULTS, "legacy_share": 0.5}
    from ycode.lm.data import encode_example
    from ycode.lm.tokenizer import BPETokenizer

    tok = BPETokenizer.load(data_dir / "tokenizer.json")
    each = len(encode_example(tok, production[0], weighted=True)[0])
    picked, tags, replay_rows = rounddata._legacy_sample(1, o, own_tokens=40 * each, tokenizer_path=data_dir / "tokenizer.json",
                                                         replay_dir=None, log=lambda *_: None)
    assert 30 <= len(picked) <= 50 and set(tags) == {"legacy:write"} and replay_rows == round(len(picked) / 200 * 200)
    again = rounddata._legacy_sample(1, o, 40 * each, data_dir / "tokenizer.json", None, lambda *_: None)[0]
    assert [e.prompt for e in again] == [e.prompt for e in picked]  # seeded


def test_runner_trains_on_a_ram_disk_and_scores_task_sets(data_dir, base_model, tmp_path, monkeypatch):
    import shutil

    from ycode.lm import experiment, rounddata

    def fake_round(out_dir, seed, tokenizer_path, **kwargs):
        import numpy as np

        shutil.copytree(data_dir, out_dir, dirs_exist_ok=True)
        n = len(np.load(Path(out_dir) / "sft_index.npy"))
        (Path(out_dir) / "sft_tags.json").write_text(json.dumps(["legacy:write"] * n))
        manifest = {"seed": seed, "options": kwargs.get("options") or {}}
        (Path(out_dir) / "manifest.json").write_text(json.dumps(manifest))
        (Path(out_dir) / "hygiene.json").write_text("[]")
        return manifest

    monkeypatch.setattr(rounddata, "write_round_dataset", fake_round)
    ram = tmp_path / "ram"
    task = {"group": "g", "name": "add1", "params": "x", "prompts": {"original": "Write a Python function `add1(x)` that returns x plus 1."},
            "tests": "assert add1(1) == 2", "test_inputs": [[1]], "expected": ["2"], "reference": "def add1(x):\n    return x + 1",
            "wrong": None, "wrong_outputs": None, "probe_args": ["(1,)", "(5,)"], "probe_expected": ["2", "6"]}
    (tmp_path / "set.json").write_text(json.dumps({"version": "t", "variants": ["original"], "tasks": [task]}))
    spec = {"name": "tiny8", "anchor": str(base_model), "anchor_sha256": "x", "tokenizer": str(data_dir / "tokenizer.json"),
            "out": str(tmp_path / "exp"), "work_dir": str(ram), "seeds": [1], "metric": "unseen_clean",
            "train": {"steps": 1, "precision": "fp32", "train_layers": 2, "init_optim": False, "compact": True,
                      "final_bf16": True, "save_every": 1},
            "eval": {"limit": 1, "diagnose": 0, "bench": False,
                     "sets": [{"name": "tiny", "set": str(tmp_path / "set.json")}]},  # absolute: taken as is
            "arms": [{"name": "C"}], "comparisons": []}
    record = experiment.run_one(spec, spec["arms"][0], 1, tmp_path / "exp", log=lambda *_: None)
    s = record["sets"]["tiny"]["scores"]["original"]
    assert s["tasks"] == 1 and s["strict"] in (0, 1) and (tmp_path / "exp" / "C" / "1" / "set_tiny.jsonl").is_file()
    assert record["exposures"]["trained_tokens"] > 0
    assert not (ram / "C" / "1" / "model").exists() and not (tmp_path / "exp" / "C" / "1" / "model").exists()


def test_transfer_split_trains_its_programs_only():
    from ycode.lm import idioms, rounddata
    from ycode.lm.leakage import skeleton

    split = json.loads((ROOT / "ycode" / "lm" / "v8_transfer_split.json").read_text())
    assert 100 <= split["counts"]["trained_behaviours"] <= 150
    assert not set(split["train_families"]) & set(split["held_out_families"])
    programs = rounddata._curriculum_programs({**rounddata.DEFAULTS, "idioms_select": "v8_transfer"})
    behaviours = {(p.family, p.source(0, "f")) for p in programs}
    assert len(behaviours) == split["counts"]["trained_behaviours"]
    assert {p.family for p in programs} <= set(split["train_families"])
    trained = {skeleton(p.source(i)) for p in programs for i in range(len(p.bodies))}
    for name in ("t_within", "t_families"):  # no test program is a trained behaviour; cross tests share no skeleton
        tasks = json.loads((ROOT / "experiments" / "v8" / "sets" / f"{name}.json").read_text())["tasks"]
        import re

        assert not {(t["family"], re.sub(rf"\b{t['name']}\b", "f", t["reference"])) for t in tasks} & behaviours
        if name == "t_families":
            assert not any(skeleton(t["reference"]) in trained for t in tasks)
    assert not trained & idioms.neighbour_skeletons()
