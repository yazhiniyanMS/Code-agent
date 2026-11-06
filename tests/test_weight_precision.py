"""16-bit weights meet the model's fp32 residual stream only under autocast on the tensors' own device.

Regression for ``RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float !=
c10::BFloat16`` from ``ycode`` on an Apple Silicon Mac: LocalLM loads a large model's weights in bf16 and
entered ``torch.autocast("cpu")`` for every device but CUDA, so on MPS nothing was cast and the fp32 hidden
states reached the bf16 weights of the first block's attention projection. No Mac runs here: the tests put
the tensors on the CPU and the autocast on another device type, the same mismatch the other way round, and
check LocalLM's context for MPS without MPS tensors (mocked)."""
import pytest
import torch

from tests.test_v10 import base_model  # noqa: F401 - fixture
from ycode.lm import model as model_mod
from ycode.lm.model import GPT, PRESETS, GPTConfig, WeightPrecisionError, autocast_on

PROMPT = "give me code for calculator in python"


def _tiny(dtype=torch.bfloat16) -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=97, **PRESETS["tiny"])).to(dtype).eval()


def _another_device_autocast():
    """An autocast that is on, for a device type the CPU tensors here are not on (on a Mac with main's
    LocalLM it was the other way round: a CPU autocast around MPS tensors)."""
    try:
        context = torch.autocast(device_type="mps", dtype=torch.bfloat16)
    except RuntimeError as exc:  # PyTorch without MPS autocast
        pytest.skip(f"this PyTorch has no MPS autocast: {exc}")
    return context


def test_16bit_weights_run_without_an_autocast_from_the_caller():
    model, idx = _tiny(), torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        expected, _ = model(idx)
    with torch.no_grad():
        got, _ = model(idx)  # was: RuntimeError: expected m1 and m2 to have the same dtype ... (CPU wording)
    assert torch.equal(got, expected)  # the same computation: bf16 matmuls, fp32 residual stream
    assert torch.equal(model.generate(idx, 5, temperature=0), _greedy_under_cpu_autocast(model, idx, 5))


def _greedy_under_cpu_autocast(model, idx, n):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        return model.generate(idx, n, temperature=0)


def test_an_autocast_for_another_device_type_does_not_count():
    model, idx = _tiny(), torch.tensor([[5, 6, 7]])
    with _another_device_autocast():
        assert autocast_on("mps") and not autocast_on("cpu")
        out = model.generate(idx, 4, temperature=0)
    assert torch.equal(out, _greedy_under_cpu_autocast(model, idx, 4))


def test_the_calculator_prompt_through_local_lm_with_the_macs_mismatch(base_model, monkeypatch):  # noqa: F811
    """The reported path, end to end: LocalLM.chat -> generate -> GPT.forward, with bf16 weights and the
    autocast on for another device type than the tensors', as on the Mac."""
    from ycode.lm.generate import LocalLM

    lm = LocalLM(base_model, device="cpu", dtype="bf16")
    assert {p.dtype for p in lm.model.parameters()} == {torch.bfloat16}
    monkeypatch.setattr(lm, "precision", _another_device_autocast)
    answer = lm.chat([("user", PROMPT)], max_new_tokens=8, temperature=0)
    assert isinstance(answer, str)


def test_local_lm_autocasts_on_the_models_own_device_type():
    """Mocked: no MPS tensors here, only the context LocalLM enters for a model on each device."""
    from ycode.lm.generate import LocalLM

    lm = object.__new__(LocalLM)
    lm.dtype = torch.bfloat16
    lm.device = "mps"
    with lm.precision():
        assert autocast_on("mps") and not autocast_on("cpu")  # main entered the CPU's autocast here
    lm.device = "cpu"
    with lm.precision():
        assert autocast_on("cpu") and not autocast_on("mps")
    if not torch.cuda.is_available():  # PyTorch turns CUDA autocast off with a warning: said, not ignored
        lm.device = "cuda:0"
        with pytest.raises(WeightPrecisionError, match="bfloat16 on cuda"), lm.precision():
            pass
    lm.dtype = None
    with lm.precision():
        assert not autocast_on("cpu") and not autocast_on("mps")


def test_weights_a_device_cannot_autocast_fail_with_a_clear_error(monkeypatch):
    model = _tiny()
    monkeypatch.setattr(model_mod, "autocast_on", lambda device_type: False)  # as if PyTorch disabled it
    with pytest.raises(WeightPrecisionError, match="weights are bfloat16 on cpu.*Load the model in fp32"):
        model(torch.tensor([[1, 2]]))


def test_fp32_weights_and_a_callers_autocast_are_left_as_they_were():
    model, idx = _tiny(torch.float32), torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        logits, _ = model(idx)
    assert logits.dtype == torch.float32 and not autocast_on("cpu")  # fp32 weights: no autocast added
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        logits, _ = model(idx)
    assert logits.dtype == torch.bfloat16  # the caller's autocast still applies to fp32 weights (training)
    half = _tiny(torch.float16)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        logits, _ = half(idx)  # a caller's autocast for this device is kept, not replaced by the weights' type
    assert logits.dtype == torch.bfloat16


def test_training_with_fp32_weights_is_unchanged():
    model, idx = _tiny(torch.float32).train(), torch.randint(0, 97, (2, 16))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, loss = model(idx, idx)
    loss.backward()
    assert loss.dtype == torch.float32 and model.embed.weight.grad.dtype == torch.float32
