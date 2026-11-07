"""Which precision a model actually runs in, on which device, decided once and stated, never assumed.

Training and inference ask for a precision (``auto``, ``fp32``, ``bf16``, ``fp16``) on a device (``cpu``,
``cuda[:N]``, ``mps``). ``resolve`` turns that into a ``PrecisionPlan``: the precision that will really run,
whether autocast is active and on which device type, and, for ``auto``, why. An explicit request that cannot
run as asked raises ``UnsupportedPrecision``; it is never replaced by another precision. Two traps this
avoids:

- autocast is per device type: a ``torch.autocast("cpu")`` context does nothing for tensors on MPS or CUDA;
- PyTorch disables autocast with only a warning when it cannot run it (e.g. CUDA requested with no GPU), so a
  context that was created is not proof that 16-bit compute happens. ``resolve`` enters the context once and
  checks ``torch.is_autocast_enabled``.

Under autocast, matrix multiplications run in the 16-bit type and PyTorch keeps reductions, norms and
softmax in fp32; ``fp32`` means no autocast at all.
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ycode.lm.optim import cuda_has_bf16

PRECISIONS = ("auto", "fp32", "bf16", "fp16")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


class UnsupportedPrecision(ValueError):
    pass


def cpu_supports_bf16() -> bool:
    """True on CPUs with native bfloat16 units (AMX / AVX512-BF16)."""
    try:
        flags = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    return "amx_bf16" in flags or "avx512_bf16" in flags


def device_available(device: str) -> bool:
    kind = device.split(":")[0]
    if kind == "cpu":
        return True
    if kind == "cuda":
        return torch.cuda.is_available()
    if kind == "mps":
        mps = getattr(torch.backends, "mps", None)
        return bool(mps and mps.is_available())
    return False


def autocast_runs(device_type: str, dtype: torch.dtype) -> tuple[bool, str]:
    """Whether autocast really turns on for ``dtype`` on ``device_type`` (entered once and checked, since
    PyTorch may disable it with only a warning), and the warning if it did not."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with torch.autocast(device_type=device_type, dtype=dtype):
                on = torch.is_autocast_enabled(device_type)
        except (RuntimeError, ValueError) as exc:
            return False, str(exc)
    return on, "; ".join(str(w.message) for w in caught)


@dataclass(frozen=True)
class PrecisionPlan:
    device: str
    requested: str
    actual: str  # fp32 | bf16 | fp16
    autocast: bool  # 16-bit compute under torch.autocast on device_type
    reason: str = ""  # why ``actual`` (for an auto request)
    fallback: bool = False  # auto chose 16-bit, but autocast does not run here: fp32 instead

    @property
    def device_type(self) -> str:
        return self.device.split(":")[0]

    @property
    def dtype(self) -> torch.dtype:
        return DTYPES[self.actual]

    def context(self):
        """The autocast context for the device the tensors are on (a no-op for fp32)."""
        if not self.autocast:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=self.dtype)

    def describe(self) -> str:
        how = (f"{self.actual} autocast on {self.device_type}" if self.autocast else f"{self.actual}, no autocast")
        why = f" ({self.reason})" if self.reason else ""
        return (f"precision: requested {self.requested}, running {how}, device {self.device}{why}"
                + (", FALLBACK to fp32" if self.fallback else ""))

    def record(self) -> dict:
        return asdict(self)


def _auto(device_type: str, device: str, policy: str) -> tuple[str, str]:
    """The ``auto`` choice and why. ``speed`` (training): 16-bit only where the hardware computes it natively.
    ``memory`` (a large model's inference): 16-bit wherever it runs, since it halves the weights' memory.
    ``fp32``: a model small enough to keep in fp32."""
    if policy == "fp32":
        return "fp32", "small enough to run in fp32"
    if device_type == "cuda":
        bf16 = cuda_has_bf16(device)
        return ("bf16", "a GPU with bf16 units") if bf16 else ("fp16", "a GPU without bf16 units")
    if policy == "memory":
        if device_type == "mps":
            if autocast_runs("mps", torch.bfloat16)[0]:
                return "bf16", "half the weights' memory"
            return "fp16", "half the weights' memory; bf16 autocast does not run on this MPS"
        return "bf16", "half the weights' memory" + ("" if device_type != "cpu" or cpu_supports_bf16()
                                                     else ", emulated on this CPU")
    if device_type == "cpu" and cpu_supports_bf16():
        return "bf16", "a CPU with bf16 units"
    if device_type == "cpu":
        return "fp32", "a CPU without bf16 units, where bf16 is emulated and slower"
    return "fp32", f"16-bit training on {device_type} is used only when asked for explicitly"


def resolve(requested: str, device: str, *, policy: str = "speed") -> PrecisionPlan:
    """The precision ``requested`` (auto | fp32 | bf16 | fp16) will run in on ``device``. An explicit
    16-bit request runs as asked, under autocast on the device's own type, or raises UnsupportedPrecision.
    ``policy`` is how ``auto`` chooses (see ``_auto``)."""
    if requested not in PRECISIONS:
        raise UnsupportedPrecision(f"unknown precision {requested!r}: choose from {', '.join(PRECISIONS)}")
    device_type = device.split(":")[0]
    if not device_available(device):
        hint = (" (this PyTorch build includes CUDA, but no GPU is visible)"
                if device_type == "cuda" and torch.backends.cuda.is_built() else "")
        raise UnsupportedPrecision(f"device {device} is not available on this machine{hint}")
    if requested == "auto":
        actual, reason = _auto(device_type, device, policy)
    else:
        actual, reason = requested, ""
    if actual == "fp32":
        return PrecisionPlan(device, requested, "fp32", False, reason)
    on, why = autocast_runs(device_type, DTYPES[actual])
    if not on:
        if requested == "auto":  # auto picks only what the hardware supports; say so if it still cannot run
            return PrecisionPlan(device, requested, "fp32", False, f"{actual} autocast does not run here: {why}",
                                 fallback=True)
        raise UnsupportedPrecision(f"{actual} cannot run on {device}: PyTorch {torch.__version__} does not "
                                   f"autocast {actual} there ({why or 'autocast stayed off'}); use fp32")
    return PrecisionPlan(device, requested, actual, True, reason)
