"""Tests for YCode-LM v3: the 40M preset and the Muon optimizer."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")

from ycode.lm.model import GPT, PRESETS, GPTConfig  # noqa: E402
from ycode.lm.optim import Muon, build_optimizer, orthogonalize  # noqa: E402
from ycode.lm.train import TrainConfig, load_checkpoint, train  # noqa: E402


def test_v3_preset_is_about_40m_params():
    cfg = GPTConfig(vocab_size=8192, **PRESETS["v3-40m"])
    params = GPT(cfg).num_params()
    assert 39e6 <= params <= 42e6
    assert cfg.arch_version == 3 and cfg.block_size == 1024 and cfg.kv_heads == 2


@pytest.mark.parametrize("shape", [(64, 32), (32, 64), (48, 48)])
def test_orthogonalize_gives_near_unit_singular_values(shape):
    torch.manual_seed(0)
    s = torch.linalg.svdvals(orthogonalize(torch.randn(*shape)).float())
    # Muon's 5-step quintic iteration maps singular values into roughly [0.7, 1.2] by design.
    assert s.max() < 1.3 and s.min() > 0.5


def test_orthogonalize_improves_conditioning():
    torch.manual_seed(0)
    u, _, v = torch.linalg.svd(torch.randn(48, 48))
    g = u @ torch.diag(torch.logspace(-1, 1, 48)) @ v  # condition number 100
    s = torch.linalg.svdvals(orthogonalize(g).float())
    assert s.max() / s.min() < 5


def test_muon_rejects_non_matrix_params():
    with pytest.raises(ValueError):
        Muon([torch.nn.Parameter(torch.ones(4))])


def test_build_optimizer_splits_parameters():
    model = GPT(GPTConfig(vocab_size=100, block_size=16, n_layer=2, n_head=2, n_embd=16))
    opt = build_optimizer(model, kind="muon", lr=1e-3, muon_lr=0.02, weight_decay=0.1)
    muon, adam = opt.optimizers
    muon_ids = {id(p) for g in muon.param_groups for p in g["params"]}
    assert id(model.embed.weight) not in muon_ids  # embeddings stay on AdamW
    assert id(model.blocks[0].attn.qkv.weight) in muon_ids
    total = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert total == model.num_params()
    opt.set_lr_scale(0.5)
    assert muon.param_groups[0]["lr"] == pytest.approx(0.01) and adam.param_groups[0]["lr"] == pytest.approx(5e-4)
    with pytest.raises(ValueError):
        build_optimizer(model, kind="sgd", lr=1e-3, muon_lr=0.02, weight_decay=0.0)


def test_muon_step_reduces_loss():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=32))
    opt = build_optimizer(model, kind="muon", lr=3e-3, muon_lr=0.02, weight_decay=0.0)
    x = torch.randint(0, 50, (8, 16))
    first = None
    for _ in range(30):
        _, loss = model(x, x)
        first = first if first is not None else loss.item()
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert loss.item() < first * 0.7


def _cfg(**kw):
    base = dict(preset="v3-tiny", model_overrides={"block_size": 32}, batch_size=4, max_steps=20, warmup_steps=2,
                eval_interval=20, eval_iters=2, log_interval=1000, device="cpu", precision="fp32",
                optimizer="muon", lr=3e-3)
    base.update(kw)
    return TrainConfig(**base)


def test_v3_training_with_muon_and_resume(data_dir, tmp_path):
    logs = []
    train(data_dir, tmp_path / "m", _cfg(max_steps=10), log=logs.append)
    assert any("YCode-LM v3" in line and "optimizer muon" in line for line in logs)
    summary = train(data_dir, tmp_path / "m", _cfg(max_steps=14, resume=True), log=logs.append)
    assert summary["steps"] == 14
    assert not any("starting it fresh" in line for line in logs)  # Muon state resumed
    model, _, payload = load_checkpoint(tmp_path / "m")
    assert payload["version"] == 3


def test_resume_with_different_optimizer_is_handled(data_dir, tmp_path):
    logs = []
    train(data_dir, tmp_path / "m", _cfg(max_steps=4, optimizer="adamw"), log=logs.append)
    train(data_dir, tmp_path / "m", _cfg(max_steps=6, resume=True), log=logs.append)
    assert any("starting it fresh" in line for line in logs)


def test_cli_defaults_to_muon_for_v3(data_dir, tmp_path, capsys):
    from ycode.lm import cli

    assert cli.main(["train", "--data", str(data_dir), "--out", str(tmp_path / "m"), "--preset", "v3-tiny",
                     "--context", "32", "--steps", "2", "--batch-size", "2", "--device", "cpu"]) == 0
    assert "optimizer muon" in capsys.readouterr().out


def test_segmented_training_matches_schedule(data_dir, tmp_path):
    from ycode.lm.train import lr_at

    logs = []
    first = train(data_dir, tmp_path / "m", _cfg(max_steps=12, until_step=5), log=logs.append)
    assert first["steps"] == 5
    second = train(data_dir, tmp_path / "m", _cfg(max_steps=12, resume=True), log=logs.append)
    assert second["steps"] == 12
    cfg = _cfg(max_steps=12)
    assert lr_at(5, cfg) == pytest.approx(cfg.lr)  # segment 1 ended mid-run at full LR, not decayed
