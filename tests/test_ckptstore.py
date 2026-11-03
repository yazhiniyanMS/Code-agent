"""Checkpoints found by identity (ycode.lm.ckptstore), with tiny synthetic checkpoints and no training.

A chain like V13's (round 54 -> C2 -> R1 -> S1: a base, a pack against it and two delta packs) is made under
one directory, as on the machine that trained it; then only the packs and the base are moved to another
directory with another layout, as in a fresh container, and the recorded paths no longer exist."""
import shutil

import pytest
import torch

from ycode.lm import ckptstore, packckpt
from ycode.lm.ckptstore import CheckpointMismatch, CheckpointNotFound


def _save(path, model, base=None):
    path.mkdir(parents=True, exist_ok=True)
    payload = {"format": "ycode-lm", "model": model}
    if base is not None:
        payload["base"] = {"path": str(base), "size": (base / "model.pt").stat().st_size}
    torch.save(payload, path / "model.pt")
    return path


def _step(state, seed):
    g = torch.Generator().manual_seed(seed)
    return {k: (v.float() + 0.05 * torch.randn(v.shape, generator=g)).to(v.dtype) for k, v in state.items()}


@pytest.fixture
def chain(tmp_path, monkeypatch):
    monkeypatch.delenv(ckptstore.ENV, raising=False)
    monkeypatch.setenv(ckptstore.ENV_BASES, "")  # restored after the test (the runner sets it)
    old, new = tmp_path / "old", tmp_path / "new"
    g = torch.Generator().manual_seed(0)
    frozen = {"blocks.0.w": torch.randn(8, 8, generator=g).to(torch.bfloat16)}
    trained = {"blocks.1.w": torch.randn(8, 8, generator=g).to(torch.bfloat16), "norm.w": torch.ones(8)}
    base = _save(old / "loop" / "anchor", {**frozen, **trained})
    states, packs = {}, {}
    previous = None
    for i, name in enumerate(("C2", "R1", "S1")):
        trained = _step(trained, i + 1)
        states[name] = trained
        packs[name] = old / "ckpt" / f"{name}.ypk"
        packs[name].parent.mkdir(exist_ok=True)
        packckpt.pack(_save(old / name, trained, base=base), packs[name], parent=previous)
        previous = packs[name]
    hashes = {name: packckpt.content_sha256(states[name]) for name in states}
    (new / "packs").mkdir(parents=True)
    for name, path in packs.items():  # another layout: every pack in one folder, the base elsewhere
        shutil.copy(path, new / "packs" / path.name)
    shutil.copytree(base, new / "bases" / "round54")
    base_sha = ckptstore.file_sha256(new / "bases" / "round54" / "model.pt")
    shutil.rmtree(old)
    ref = {"name": "S1", "content_sha256": hashes["S1"], "file": "packs/S1.ypk", "parents": [hashes["R1"], hashes["C2"]],
           "base": {"sha256": base_sha, "size": (new / "bases" / "round54" / "model.pt").stat().st_size}}
    return {"root": new, "states": states, "hashes": hashes, "ref": ref, "base_sha": base_sha, "monkeypatch": monkeypatch}


def test_a_delta_chain_loads_by_identity_after_moving_to_another_machine(chain):
    root = chain["root"]
    s1 = root / "packs" / "S1.ypk"
    with pytest.raises(FileNotFoundError):  # the recorded paths are gone: without a checkpoint directory it fails
        packckpt.unpacked_state(s1)
    chain["monkeypatch"].setenv(ckptstore.ENV, str(root))
    with pytest.raises(CheckpointMismatch, match="cannot check bases/round54"):  # a base with no hash to check
        packckpt.unpacked_state(s1)
    chain["monkeypatch"].setenv(ckptstore.ENV_BASES, chain["base_sha"])
    state, _, header = packckpt.unpacked_state(s1)  # parents and base found by identity, every tensor checked
    assert header["content_sha256"] == chain["hashes"]["S1"]
    assert all(torch.equal(state[k], chain["states"]["S1"][k]) for k in state)
    links = ckptstore.resolve_pack(chain["ref"], root)
    assert [(link.role, link.path.relative_to(root).as_posix()) for link in links] == [
        ("checkpoint", "packs/S1.ypk"), ("parent 1", "packs/R1.ypk"), ("parent 2", "packs/C2.ypk"),
        ("base", "bases/round54")]
    assert links[-1].identity == chain["base_sha"]


def test_a_missing_checkpoint_says_what_was_looked_for(chain):
    with pytest.raises(CheckpointNotFound, match=r"S1 \(content 0{12}\.\.\.\) not found under .*packs/C2\.ypk"):
        ckptstore.resolve_pack({**chain["ref"], "content_sha256": "0" * 64, "file": None}, chain["root"])
    with pytest.raises(CheckpointNotFound, match=ckptstore.ENV):  # no checkpoint directory configured
        ckptstore.resolve_pack(chain["ref"], None)


def test_another_checkpoint_under_the_expected_name_is_never_used(chain):
    root = chain["root"]
    (root / "packs" / "S1.ypk").rename(root / "S1.real")  # the real S1 is not a .ypk any more: not found
    shutil.copy(root / "packs" / "R1.ypk", root / "packs" / "S1.ypk")  # and another pack has its name
    with pytest.raises(CheckpointMismatch, match=r"packs/S1\.ypk is another checkpoint"):
        ckptstore.resolve_pack(chain["ref"], root)


def test_a_chain_other_than_the_spec_expects_is_refused(chain):
    ref = {**chain["ref"], "parents": [chain["hashes"]["C2"]]}  # S1 was packed against R1, not C2
    with pytest.raises(CheckpointMismatch, match="the spec expects"):
        ckptstore.resolve_pack(ref, chain["root"])
    with pytest.raises(CheckpointMismatch, match="the spec expects"):  # the chain goes on past the expected end
        ckptstore.resolve_pack({**chain["ref"], "parents": [chain["hashes"]["R1"]]}, chain["root"])


def test_a_missing_parent_fails_loading_and_resolving(chain):
    root = chain["root"]
    (root / "packs" / "R1.ypk").unlink()
    chain["monkeypatch"].setenv(ckptstore.ENV, str(root))
    chain["monkeypatch"].setenv(ckptstore.ENV_BASES, chain["base_sha"])
    with pytest.raises(CheckpointNotFound, match=r"parent pack \(recorded at .*R1\.ypk\)"):
        packckpt.unpacked_state(root / "packs" / "S1.ypk")
    with pytest.raises(CheckpointNotFound, match=r"parent 1 of S1"):
        ckptstore.resolve_pack(chain["ref"], root)


def test_a_missing_or_different_base_is_refused(chain):
    root = chain["root"]
    model = root / "bases" / "round54" / "model.pt"
    payload = torch.load(model, weights_only=True)
    payload["model"]["blocks.0.w"] = -payload["model"]["blocks.0.w"]  # same size; only a frozen layer differs,
    torch.save(payload, model)  # which no pack's content hash covers
    with pytest.raises(CheckpointMismatch, match=r"only other checkpoints .* have that size"):
        ckptstore.resolve_pack(chain["ref"], root)
    chain["monkeypatch"].setenv(ckptstore.ENV, str(root))
    chain["monkeypatch"].setenv(ckptstore.ENV_BASES, chain["base_sha"])
    with pytest.raises(CheckpointMismatch, match="only other checkpoints"):  # loading refuses it too
        packckpt.unpacked_state(root / "packs" / "C2.ypk")
    shutil.rmtree(root / "bases")
    with pytest.raises(CheckpointNotFound, match=r"base of S1 \(recorded as .*loop/anchor"):
        ckptstore.resolve_pack(chain["ref"], root)


def test_a_compact_checkpoint_finds_its_base_by_identity(tmp_path, monkeypatch):
    import json

    from ycode.lm.train import base_state

    monkeypatch.setenv(ckptstore.ENV_BASES, "")
    old = tmp_path / "old" / "anchor"
    _save(old, {"w": torch.arange(4.0)})
    ref = {"path": str(old), "size": (old / "model.pt").stat().st_size}
    shutil.move(str(old), str(tmp_path / "store" / "round54"))
    _save(tmp_path / "store" / "twin", {"w": torch.arange(4.0) + 1})  # same size, other weights
    monkeypatch.setenv(ckptstore.ENV, str(tmp_path / "store"))
    with pytest.raises(CheckpointMismatch, match="cannot check round54, twin"):  # nothing to tell them apart
        base_state({"base": ref})
    model = tmp_path / "store" / "round54" / "model.pt"  # the hash recorded when it was saved (train.py's info.json)
    (model.parent / "info.json").write_text(json.dumps({"sha256": ckptstore.file_sha256(model)}))
    assert torch.equal(base_state({"base": ref})["w"], torch.arange(4.0))


def test_a_spec_names_its_checkpoints_by_identity(chain):
    from ycode.lm.experiment import resolve_checkpoints

    root = chain["root"]
    anchor = {"name": "round 54", "sha256": chain["base_sha"], "size": chain["ref"]["base"]["size"], "file": "bases/round54"}
    spec = {"anchor_checkpoint": anchor, "arms": [{"name": "T1", "init_checkpoint": chain["ref"]}]}
    with pytest.raises(CheckpointNotFound, match=ckptstore.ENV):
        resolve_checkpoints(dict(spec, arms=[dict(spec["arms"][0])]))
    chain["monkeypatch"].setenv(ckptstore.ENV, str(root))
    resolved = resolve_checkpoints(spec)
    import os

    assert os.environ[ckptstore.ENV_BASES] == chain["base_sha"]  # passed on to the processes the runner starts
    assert resolved["anchor"] == str(root / "bases" / "round54") and resolved["anchor_sha256"] == chain["base_sha"]
    arm = resolved["arms"][0]
    assert arm["init_from"] == str(root / "packs" / "S1.ypk")
    assert [link["role"] for link in arm["init_chain"]] == ["checkpoint", "parent 1", "parent 2", "base"]
    with pytest.raises(ValueError, match="both"):
        resolve_checkpoints({"anchor": "/somewhere", "anchor_checkpoint": anchor})


def test_the_t1_spec_in_a_fresh_container_names_what_is_missing(tmp_path, monkeypatch):
    """V13's T1 starts from S1 1301 by identity: without it, the run stops before anything is built."""
    import json
    from pathlib import Path

    from ycode.lm.experiment import resolve_checkpoints

    spec = json.loads((Path(__file__).parents[1] / "experiments/v13/specs/v13_t1.json").read_text())
    assert spec["init_checkpoint"]["content_sha256"].startswith("5075d097") and "init_from" not in spec
    monkeypatch.setenv(ckptstore.ENV, str(tmp_path))  # an empty checkpoint directory
    monkeypatch.setenv(ckptstore.ENV_BASES, "")
    with pytest.raises(CheckpointNotFound, match=r"round 54 \(recorded as loop/anchor, model\.pt of 5206846811 bytes"):
        resolve_checkpoints(dict(spec))
    with pytest.raises(CheckpointNotFound, match=r"S1 1301 \(content 5075d097a657\.\.\.\) not found"):
        resolve_checkpoints({k: v for k, v in spec.items() if k != "anchor_checkpoint"})
