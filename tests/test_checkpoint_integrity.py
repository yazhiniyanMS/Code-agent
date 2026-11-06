"""Checkpoint saves are all-or-nothing, a resume never pairs weights with another step's optimizer state, and a
new run never replaces an existing checkpoint without being told to. Tiny synthetic models only."""
import json

import pytest
import torch

from tests.test_v51 import _cfg
from ycode.lm import train as T


def _fail_on(monkeypatch, *, optimizer_at=None, model_at=None):
    """Make torch.save fail for the optimizer state or the weights saved at one step, after writing part of
    the file, as a full disk or a killed process would."""
    real = torch.save

    def save(obj, f, *a, **k):
        failing = (optimizer_at is not None and isinstance(obj, dict) and obj.get("step") == optimizer_at
                   and "optimizer" in obj) or (model_at is not None and isinstance(obj, dict)
                                                and obj.get("step") == model_at and "model" in obj)
        if failing:
            (f.write if hasattr(f, "write") else open(f, "wb").write)(b"partial")
            raise OSError(28, "No space left on device")
        return real(obj, f, *a, **k)

    monkeypatch.setattr(T.torch, "save", save)


def test_a_new_run_does_not_replace_an_existing_checkpoint(data_dir, tmp_path):
    out = tmp_path / "run"
    T.train(data_dir, out, _cfg(max_steps=2), log=lambda *_: None)
    saved = T.file_sha256(out / "model.pt")
    with pytest.raises(FileExistsError, match=r"already holds a checkpoint \(step 2\).*--resume.*--overwrite"):
        T.train(data_dir, out, _cfg(max_steps=2, seed=7), log=lambda *_: None)
    assert T.file_sha256(out / "model.pt") == saved
    T.train(data_dir, out, _cfg(max_steps=2, seed=7, overwrite=True), log=lambda *_: None)
    assert T.file_sha256(out / "model.pt") != saved


def test_a_resume_never_pairs_weights_with_another_steps_optimizer_state(data_dir, tmp_path, monkeypatch):
    out = tmp_path / "run"
    _fail_on(monkeypatch, optimizer_at=4)  # the weights of step 4 are saved, then the optimizer save fails
    with pytest.raises(OSError):
        T.train(data_dir, out, _cfg(max_steps=4, save_interval=2), log=lambda *_: None)
    monkeypatch.undo()
    assert torch.load(out / "model.pt", weights_only=True)["step"] == 4
    assert torch.load(out / "optim.pt", weights_only=True)["step"] == 2
    lines = []
    T.train(data_dir, out, _cfg(max_steps=6, save_interval=2, resume=True), log=lines.append)
    assert any("optimizer state" in line and "step 2" in line and "step 4" in line and "fresh" in line
               for line in lines), lines


def test_an_interrupted_weights_save_leaves_the_previous_checkpoint(data_dir, tmp_path, monkeypatch):
    out = tmp_path / "run"
    synced = []
    monkeypatch.setattr(T.os, "fsync", lambda fd: synced.append(fd))
    _fail_on(monkeypatch, model_at=4)
    with pytest.raises(OSError):
        T.train(data_dir, out, _cfg(max_steps=4, save_interval=2), log=lambda *_: None)
    assert len(synced) >= 3  # step 2's weights, optimizer state and info.json reached the disk before renaming
    monkeypatch.undo()
    assert sorted(p.name for p in out.iterdir() if p.suffix != ".jsonl") == ["info.json", "model.pt", "optim.pt"]
    assert torch.load(out / "model.pt", weights_only=True)["step"] == 2  # the previous checkpoint, intact
    info = json.loads((out / "info.json").read_text())
    assert info["step"] == 2 and T.verify_checkpoint(out) == info["sha256"]
