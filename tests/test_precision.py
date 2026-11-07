"""Device-aware precision (ycode.lm.precision): what runs is decided once, stated, and never silently swapped.

CPU paths run for real. MPS and CUDA paths are dispatch logic tested with the device's availability and
PyTorch's autocast support mocked: they show which context and dtype would be used, not that an Apple or
NVIDIA GPU computes them (none is available here)."""
import json

import pytest
import torch

from tests.test_v10 import base_model, sft  # noqa: F401 - fixtures
from ycode.lm import precision as P
from ycode.lm.precision import UnsupportedPrecision, resolve


@pytest.fixture
def fake_mps(monkeypatch):
    """An MPS device that exists, with bf16 and fp16 autocast support set per test."""
    support = {torch.bfloat16: True, torch.float16: True}
    real = P.device_available
    monkeypatch.setattr(P, "device_available", lambda d: d == "mps" or real(d))
    real_runs = P.autocast_runs
    monkeypatch.setattr(P, "autocast_runs", lambda t, dt: ((support[dt], "" if support[dt] else
                                                             "In MPS autocast, but the target dtype is not supported")
                                                            if t == "mps" else real_runs(t, dt)))
    return support


def test_cpu_plans_run_what_they_say(monkeypatch):
    plan = resolve("bf16", "cpu")
    assert (plan.actual, plan.autocast, plan.device_type, plan.fallback) == ("bf16", True, "cpu", False)
    with plan.context():
        assert torch.is_autocast_enabled("cpu") and (torch.ones(4, 4) @ torch.ones(4, 4)).dtype == torch.bfloat16
    fp32 = resolve("fp32", "cpu")
    with fp32.context():
        assert not torch.is_autocast_enabled("cpu") and (torch.ones(4, 4) @ torch.ones(4, 4)).dtype == torch.float32
    assert resolve("fp16", "cpu").actual == "fp16"  # asked for fp16: fp16, not bf16
    monkeypatch.setattr(P, "cpu_supports_bf16", lambda: True)
    assert resolve("auto", "cpu").actual == "bf16"
    monkeypatch.setattr(P, "cpu_supports_bf16", lambda: False)
    plan = resolve("auto", "cpu")
    assert plan.actual == "fp32" and "without bf16 units" in plan.reason
    assert resolve("auto", "cpu", policy="memory").actual == "bf16"  # a large model's weights: half the memory
    assert resolve("auto", "cpu", policy="fp32").actual == "fp32"


def test_requests_that_cannot_run_are_refused():
    with pytest.raises(UnsupportedPrecision, match="unknown precision 'float16'"):
        resolve("float16", "cpu")
    if not torch.cuda.is_available():
        with pytest.raises(UnsupportedPrecision, match="cuda is not available"):
            resolve("bf16", "cuda")
    if not P.device_available("mps"):
        with pytest.raises(UnsupportedPrecision, match="mps is not available"):
            resolve("fp32", "mps")


def test_mps_autocasts_on_mps_not_on_the_cpu(fake_mps):
    plan = resolve("bf16", "mps")
    assert (plan.actual, plan.autocast, plan.device_type) == ("bf16", True, "mps")
    context = plan.context()
    assert isinstance(context, torch.autocast) and context.device == "mps" and context.fast_dtype == torch.bfloat16
    assert "bf16 autocast on mps" in plan.describe()
    assert resolve("auto", "mps").actual == "fp32"  # training: 16-bit on MPS only when asked for


def test_mps_without_bf16_autocast_is_refused_not_run_in_fp32(fake_mps):
    fake_mps[torch.bfloat16] = False
    with pytest.raises(UnsupportedPrecision, match="bf16 cannot run on mps.*target dtype is not supported"):
        resolve("bf16", "mps")
    plan = resolve("auto", "mps", policy="memory")  # the local provider's large model: fp16 instead, stated
    assert plan.actual == "fp16" and "bf16 autocast does not run" in plan.reason and not plan.fallback
    fake_mps[torch.float16] = False
    plan = resolve("auto", "mps", policy="memory")
    assert (plan.actual, plan.autocast, plan.fallback) == ("fp32", False, True)
    assert "FALLBACK to fp32" in plan.describe() and plan.record()["fallback"] is True


def test_cuda_where_autocast_would_be_disabled_with_only_a_warning(monkeypatch):
    # PyTorch disables autocast with a warning when it cannot run it; a created context is no proof
    monkeypatch.setattr(P, "device_available", lambda d: True)
    monkeypatch.setattr(P, "cuda_has_bf16", lambda device=None: True)
    if not torch.cuda.is_available():
        with pytest.raises(UnsupportedPrecision, match="CUDA is not available.*Disabling autocast"):
            resolve("bf16", "cuda")
        plan = resolve("auto", "cuda")
        assert plan.fallback and plan.actual == "fp32"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_bf16_on_a_real_gpu():
    plan = resolve("auto", "cuda")
    with plan.context():
        y = torch.ones(8, 8, device="cuda") @ torch.ones(8, 8, device="cuda")
    assert y.dtype == plan.dtype


def test_the_trainer_states_and_records_its_precision(base_model, sft, tmp_path):  # noqa: F811
    from ycode.lm.train import TrainConfig, train

    lines = []
    train(sft, tmp_path / "bf16", TrainConfig(stage="sft", init_from=str(base_model), max_steps=1, batch_size=2,
                                              eval_iters=0, eval_interval=1000, device="cpu", precision="bf16"),
          log=lines.append)
    plan = json.loads((tmp_path / "bf16" / "train_log.json").read_text())["summary"]["precision_plan"]
    assert plan == {"device": "cpu", "requested": "bf16", "actual": "bf16", "autocast": True, "reason": "",
                    "fallback": False}
    assert any(line.startswith("precision: requested bf16, running bf16 autocast on cpu") for line in lines)
    if not torch.cuda.is_available():  # refused before any model is loaded
        with pytest.raises(UnsupportedPrecision, match="no GPU is visible"):
            train(sft, tmp_path / "cuda", TrainConfig(stage="sft", init_from=str(base_model), max_steps=1,
                                                      device="cuda", precision="bf16"), log=lambda *_: None)
        assert not (tmp_path / "cuda").exists()


def test_the_local_provider_runs_the_precision_it_was_asked_for(base_model, monkeypatch):  # noqa: F811
    from ycode.lm import generate
    from ycode.lm.generate import LocalLM

    for asked, weights in (("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)):
        lm = LocalLM(base_model, device="cpu", dtype=asked)
        assert lm.plan.actual == asked and next(lm.model.parameters()).dtype == weights
        assert isinstance(lm.chat([("user", "hi")], max_new_tokens=2, temperature=0), str)
    assert LocalLM(base_model, device="cpu").plan.actual == "fp32"  # a small model: auto keeps fp32
    monkeypatch.setattr(generate, "stored_params", lambda _: 1_500_000_000)  # a large one: 16-bit weights
    big = LocalLM(base_model, device="cpu")
    assert big.plan.actual == "bf16" and "half the weights' memory" in big.plan.reason
    with pytest.raises(UnsupportedPrecision):
        LocalLM(base_model, device="cpu", dtype="half")


def test_the_hardware_benchmark_records_the_plan_and_refuses_what_cannot_run(fake_mps):
    from ycode.lm import hardware

    row = hardware.benchmark("cpu", "bf16", preset="tiny", train=False, gen_tokens=2)
    assert row["plan"]["actual"] == "bf16" and row["tokens_per_s"] > 0
    fake_mps[torch.bfloat16] = False
    row = hardware.benchmark("mps", "bf16", preset="tiny", steps=1)
    assert "tokens_per_s" not in row and "bf16 cannot run on mps" in row["error"]
