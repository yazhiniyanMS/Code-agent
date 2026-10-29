"""The runner refuses a run the machine's memory cannot hold, before training instead of at the first save."""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ycode.lm import experiment  # noqa: E402

ROUND54 = Path("/home/user/lmwork/loop/anchor")


def _anchor(tmp_path: Path) -> Path:
    """A 4-block model: 10 weights per block, 3 in the final norm, 20 in the embedding (no other tensors)."""
    state = {f"blocks.{i}.w": torch.zeros(10) for i in range(4)}
    state.update({"norm.weight": torch.zeros(3), "tok_emb.weight": torch.zeros(20)})
    torch.save({"config": {"n_layer": 4}, "model": state}, tmp_path / "model.pt")
    return tmp_path


def test_training_memory_counts_every_weight_and_the_momentum_of_the_trained_ones(tmp_path):
    anchor = _anchor(tmp_path)
    total, top2 = 4 * 10 + 3 + 20, 2 * 10 + 3  # the top 2 blocks and the final norm are trained
    assert experiment._training_ram_gb(anchor, {"train_layers": 2, "precision": "fp32"}) == \
        pytest.approx((total + top2) * 4 / 1e9 + 0.6)
    assert experiment._training_ram_gb(anchor, {"train_layers": 2, "precision": "bf16"}) == \
        pytest.approx((total + top2) * 2 / 1e9 + 0.6)
    assert experiment._training_ram_gb(anchor, {"precision": "fp32"}) == pytest.approx(2 * total * 4 / 1e9 + 0.6)


def test_the_momentum_freed_before_a_final_save_is_two_bytes_per_trained_weight(tmp_path):
    assert experiment._momentum_freed_gb(_anchor(tmp_path), {"train_layers": 2}) == pytest.approx((2 * 10 + 3) * 2 / 1e9)


def test_only_a_final_bf16_save_without_earlier_saves_is_final_only():
    assert experiment._final_only({"final_bf16": True, "steps": 8, "save_every": 8})
    assert not experiment._final_only({"final_bf16": True, "steps": 8, "save_every": 4})  # earlier saves keep it
    assert not experiment._final_only({"steps": 8, "save_every": 8})


@pytest.mark.skipif(not (ROUND54 / "model.pt").is_file(), reason="round 54 is not on this machine")
def test_the_estimate_matches_the_measured_peak_of_real_runs():
    # V10-V12 runs from C2 / R1 (24 of 34 layers, fp32) peaked at 10.8-11.3 GB resident
    assert 10.5 < experiment._training_ram_gb(ROUND54, {"train_layers": 24, "precision": "fp32"}) < 11.5
    # V12-S1's trained tensors are 2.127 GB in bf16: Lion's bf16 momentum is the same size
    assert 2.0 < experiment._momentum_freed_gb(ROUND54, {"train_layers": 24}) < 2.2


def test_available_memory_is_read_from_meminfo():
    assert experiment._mem_available_gb("MemTotal: 16477352 kB\nMemAvailable:   11376884 kB\n") == \
        pytest.approx(11376884 * 1024 / 1e9)
    assert experiment._mem_available_gb("MemTotal: 1 kB\n") is None


def test_a_ram_disk_is_recognised_by_its_longest_mount():
    mounts = "/dev/vda / ext4 rw 0 0\ntmpfs /dev/shm tmpfs rw 0 0\n/dev/vdb /dev/shm/disk ext4 rw 0 0\n"
    assert experiment._on_tmpfs(Path("/dev/shm/ycode-v12/L/1301"), mounts)
    assert not experiment._on_tmpfs(Path("/home/user/lmwork/v12"), mounts)
    assert not experiment._on_tmpfs(Path("/dev/shm/disk/run"), mounts)  # a disk mounted inside the RAM disk
    assert not experiment._on_tmpfs(Path("/dev/shmother"), mounts)  # a prefix of the name is not the mount


def test_a_run_that_cannot_fit_is_refused_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "_training_ram_gb", lambda anchor, train: 11.0)
    monkeypatch.setattr(experiment, "_checkpoint_gb", lambda anchor, train: 2.1)
    monkeypatch.setattr(experiment, "_on_tmpfs", lambda path, mounts=None: True)
    # V12-S1's second seed: 1.3 GB pack + 2.1 GB starting model on the RAM disk left about 12 GB available
    monkeypatch.setattr(experiment, "_mem_available_gb", lambda meminfo=None: 12.0)
    with pytest.raises(RuntimeError, match="memory available; training needs about 13.6 GB"):  # 11.0 + 2.1 + 0.5
        experiment._check_memory(tmp_path, {}, tmp_path)
    monkeypatch.setattr(experiment, "_mem_available_gb", lambda meminfo=None: 14.2)  # S1's first seed: fits
    assert experiment._check_memory(tmp_path, {}, tmp_path) == (pytest.approx(13.6), 14.2)
    monkeypatch.setattr(experiment, "_on_tmpfs", lambda path, mounts=None: False)  # checkpoint on disk: model only
    monkeypatch.setattr(experiment, "_mem_available_gb", lambda meminfo=None: 12.0)
    assert experiment._check_memory(tmp_path, {}, tmp_path)[0] == pytest.approx(11.5)
    monkeypatch.setattr(experiment, "_mem_available_gb", lambda meminfo=None: None)  # unknown: no refusal
    assert experiment._check_memory(tmp_path, {}, tmp_path)[1] is None


def test_a_final_weights_only_save_needs_the_checkpoint_minus_the_freed_momentum(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "_training_ram_gb", lambda anchor, train: 11.0)
    monkeypatch.setattr(experiment, "_momentum_freed_gb", lambda anchor, train: 2.1)
    monkeypatch.setattr(experiment, "_on_tmpfs", lambda path, mounts=None: True)
    monkeypatch.setattr(experiment, "_mem_available_gb", lambda meminfo=None: 12.0)
    final = {"final_bf16": True, "steps": 8, "save_every": 8}
    # V12-S1's second seed with the trainer freeing its momentum first: the save fits where the training does
    monkeypatch.setattr(experiment, "_checkpoint_gb", lambda anchor, train: 2.1)
    assert experiment._check_memory(tmp_path, final, tmp_path) == (pytest.approx(11.5), 12.0)
    monkeypatch.setattr(experiment, "_checkpoint_gb", lambda anchor, train: 4.0)  # a checkpoint larger than the momentum
    with pytest.raises(RuntimeError, match="needs about 13.4 GB"):  # 11.0 - 2.1 + 4.0 + 0.5
        experiment._check_memory(tmp_path, final, tmp_path)
    with pytest.raises(RuntimeError, match="needs about 15.5 GB"):  # earlier saves keep the optimizer: 11.0 + 4.0 + 0.5
        experiment._check_memory(tmp_path, {**final, "save_every": 4}, tmp_path)
