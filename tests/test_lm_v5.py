"""v5 training at 1.5B scale: low-memory Lion (updates inside backward), sharded (FSDP) training
with Muon, fp16 Newton-Schulz on GPUs without bf16, and autotrain's out-of-memory fallback."""

import copy
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")

from ycode.lm.model import GPT, PRESETS, GPTConfig  # noqa: E402
from ycode.lm.optim import Lion, build_optimizer  # noqa: E402
from ycode.lm.train import TrainConfig, load_checkpoint, train  # noqa: E402


def _tiny(seed=0):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=100, **{**PRESETS["v5-tiny"], "block_size": 32}))


def test_v5_presets_sizes():
    from ycode.lm.autotrain import preset_params

    assert 1.45e9 < preset_params("v5-1.5b") < 1.6e9
    assert 3.5e8 < preset_params("v5-385m") < 4.0e8


def test_lion_in_backward_matches_regular_step():
    m1 = _tiny()
    m2 = copy.deepcopy(m1)
    x, y = torch.randint(0, 100, (2, 32)), torch.randint(0, 100, (2, 32))
    o1 = build_optimizer(m1, kind="lion", lr=1e-3, muon_lr=0, weight_decay=0.1)
    o2 = build_optimizer(m2, kind="lion", lr=1e-3, muon_lr=0, weight_decay=0.1)
    o2.optimizers[0].attach()
    for _ in range(3):
        m1(x, y)[1].backward()
        o1.step()
        o1.zero_grad()
        m2(x, y)[1].backward()  # updates happen here
        assert all(p.grad is None for p in m2.parameters())  # gradients freed as they are used
    for a, b in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(a, b)


def test_lion_moves_each_weight_by_lr_and_keeps_bf16_state():
    p = torch.nn.Parameter(torch.zeros(4))
    opt = Lion([p], lr=0.1)
    p.grad = torch.tensor([1.0, -2.0, 0.5, -0.1])
    opt.step()
    assert torch.allclose(p.detach(), torch.tensor([-0.1, 0.1, -0.1, 0.1]))
    assert opt.state[p]["exp_avg"].dtype == torch.bfloat16
    fresh = Lion([p], lr=0.5)
    fresh.load_state_dict(opt.state_dict())  # torch's default would upcast the state to fp32
    assert fresh.state[p]["exp_avg"].dtype == torch.bfloat16
    assert fresh.param_groups[0]["lr"] == 0.1


def test_training_with_lion_low_memory_and_resume(data_dir, tmp_path):
    cfg = dict(preset="v3-tiny", model_overrides={"block_size": 32}, batch_size=4, warmup_steps=2,
               eval_interval=6, eval_iters=2, log_interval=1000, device="cpu", precision="fp32",
               optimizer="lion", lr=3e-3, grad_checkpoint=True)
    logs = []
    s1 = train(data_dir, tmp_path / "m", TrainConfig(max_steps=12, until_step=6, **cfg), log=logs.append)
    assert s1["steps"] == 6
    assert not any("overflow" in line for line in logs)
    state = torch.load(tmp_path / "m" / "optim.pt", weights_only=True)["optimizer"]["optimizers"][0]
    assert next(iter(state["state"].values()))["exp_avg"].dtype == torch.bfloat16
    s2 = train(data_dir, tmp_path / "m", TrainConfig(max_steps=12, resume=True, **cfg), log=lambda *_: None)
    assert s2["steps"] == 12 and s2["final"]["train"] < 5.5  # untrained ~ ln(400) = 6.0
    with pytest.raises(ValueError, match="grad-accum 1"):
        train(data_dir, tmp_path / "x", TrainConfig(max_steps=2, grad_accum=2, **cfg), log=lambda *_: None)


def test_newton_schulz_dtype():
    from ycode.lm.optim import _ns_dtype

    assert _ns_dtype(torch.device("cpu")) == torch.bfloat16


def test_sharded_training_two_processes_with_muon_and_resume(data_dir, tmp_path):
    """FSDP across 2 processes (gloo on CPU), as used for 1.5B on 2 GPUs: Muon gathers each sharded
    matrix to orthogonalize it, checkpoints gather the full model, optimizer state is per rank."""
    out = tmp_path / "fsdp"

    def run(*extra):
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "--master_port=29519",
               "-m", "ycode.lm.cli", "train", "--data", str(data_dir), "--out", str(out), "--preset", "v3-tiny",
               "--context", "32", "--steps", "8", "--batch-size", "2", "--device", "cpu", "--precision", "fp32",
               "--eval-interval", "4", "--optimizer", "muon", "--shard", *extra]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-3000:]
        return proc.stdout

    first = run("--until-step", "4")
    assert "sharded (FSDP)" in first
    assert (out / "optim-rank0-of-2.pt").is_file() and (out / "optim-rank1-of-2.pt").is_file()
    second = run("--resume")
    assert "starting it fresh" not in second
    model, _, payload = load_checkpoint(out)
    assert payload["step"] == 8
    # The gathered checkpoint is a normal, complete model.
    assert model.num_params() == GPT(model.cfg).num_params()
    assert payload["val_loss"] < 6.0


def test_autotrain_halves_batch_after_out_of_memory(tmp_path, monkeypatch):
    from ycode.lm import autotrain
    from ycode.lm.autotrain import AutoTrainer, Plan, State
    from ycode.lm.hub import LocalHub

    plan = Plan(base_model="unused", batch_size=8, grad_accum=1, first_segment_steps=2, device="cpu")
    trainer = AutoTrainer(plan, tmp_path / "w", LocalHub(tmp_path / "hub", "you/ckpt"), None,
                          log=lambda *_: None)
    out = tmp_path / "w" / "base"
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        if len(calls) == 1:
            return 1, "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"
        out.mkdir(parents=True, exist_ok=True)
        (out / "info.json").write_text('{"step": 2}')
        return 0, ""

    monkeypatch.setattr(autotrain, "_run_streaming", fake_run)
    state = State()
    assert trainer.run_phase(state, "pretrain", tmp_path, tmp_path, out, total=2)
    assert (state.batch_size, state.grad_accum) == (4, 2)
    retry = calls[1]
    assert retry[retry.index("--batch-size") + 1] == "4" and retry[retry.index("--grad-accum") + 1] == "2"


def test_autotrain_shards_large_models_on_multiple_gpus(tmp_path):
    from ycode.lm.autotrain import AutoTrainer, Plan
    from ycode.lm.hub import LocalHub

    plan = Plan(base_model="unused", preset="v5-1.5b", nproc=2, shard=True)
    trainer = AutoTrainer(plan, tmp_path / "w", LocalHub(tmp_path / "hub", "x/y"), None, log=lambda *_: None)
    pre = trainer._train_cmd("pretrain", tmp_path, tmp_path, tmp_path, 10, 5)
    sft = trainer._train_cmd("sft", tmp_path, tmp_path, tmp_path, 10, 5)
    assert "--shard" in pre and "torch.distributed.run" in pre
    assert sft[sft.index("--optimizer") + 1] == "muon"  # AdamW's two states would not fit


def test_half_precision_loading_for_inference(data_dir, tmp_path):
    """Big models load straight into bf16 (half the RAM, no fp32 copy) and still generate."""
    from ycode.lm.generate import LocalLM
    from ycode.lm.tokenizer import BPETokenizer
    from ycode.lm.train import export_checkpoint, save_checkpoint

    tok = BPETokenizer.load(data_dir / "tokenizer.json")
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, **{**PRESETS["v5-tiny"], "block_size": 64}))
    save_checkpoint(tmp_path / "full", model, tok, step=1, stage="sft", val_loss=None)
    export_checkpoint(tmp_path / "full", tmp_path / "slim", dtype="bf16", max_shard_mb=0.05)

    small = LocalLM(tmp_path / "slim", device="cpu")  # tiny model: auto keeps full precision
    assert small.dtype is None and next(small.model.parameters()).dtype == torch.float32
    half = LocalLM(tmp_path / "slim", device="cpu", dtype="bf16")
    assert half.dtype == torch.bfloat16
    assert all(p.dtype == torch.bfloat16 for p in half.model.parameters())
    assert half.model.head.weight is half.model.embed.weight  # still tied
    assert half.model.rope_cos.dtype == torch.float32  # positions keep full precision
    torch.manual_seed(0)
    assert isinstance(half.complete("def f(", max_new_tokens=5), str)
    with half.precision():
        x = torch.randint(0, tok.vocab_size, (1, 16))
        _, loss_half = half.model(x, x)
    _, loss_full = small.model(x, x)
    assert abs(loss_half.item() - loss_full.item()) < 0.1
