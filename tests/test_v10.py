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
