"""What this machine can run YCode-LM on, measured rather than assumed: ``python -m ycode.lm.cli hardware``.

Before training on a machine (an Apple Silicon MacBook in particular), this reports:

- the machine: OS, chip, cores (performance and efficiency cores on Apple Silicon), memory and how much is
  available, free disk, and the Python, PyTorch and NumPy versions;
- PyTorch's devices: the CPU, Apple's Metal backend (MPS) and CUDA, whether each is built and available;
- a correctness check of each available device against the CPU: one forward and backward pass of a small
  model with the same weights and batch, compared value by value;
- with ``--bench``, inference (greedy decoding with the KV cache) and training (forward, backward and an
  optimizer step) throughput for each device and precision the trainer actually uses there. Each benchmark
  first estimates its memory and is skipped, with the reason, if the machine does not have it to spare.

The recommendation is the fastest configuration whose device passed its check, for training and for
inference separately; nothing is chosen from a device's name. The JSON report can be compared with a later
run or with another machine's.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# The trainer uses mixed precision (autocast) on CPU and CUDA only: on MPS it trains in fp32 whatever
# --precision asks for (ycode/lm/train.py), so that is what is benchmarked for training there.
TRAIN_PRECISIONS = {"cpu": ("fp32", "bf16"), "cuda": ("fp32", "bf16"), "mps": ("fp32",)}
INFER_PRECISIONS = {"cpu": ("fp32", "bf16"), "cuda": ("fp32", "bf16"), "mps": ("fp32", "bf16", "fp16")}
MEMORY_SHARE = 0.6  # a benchmark may use at most this share of the memory available when it starts


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def parse_meminfo(text: str) -> dict:
    """Total and available memory (GB) from Linux's /proc/meminfo."""
    kb = {m.group(1): int(m.group(2)) for m in re.finditer(r"^(\w+):\s+(\d+) kB", text, re.M)}
    return {"total_gb": kb.get("MemTotal", 0) / 1e6, "available_gb": kb.get("MemAvailable", 0) / 1e6}


def parse_vm_stat(text: str) -> float | None:
    """Available memory (GB) from macOS's vm_stat: free, inactive, speculative and purgeable pages, which the
    system can hand out without swapping."""
    size = re.search(r"page size of (\d+) bytes", text)
    if not size:
        return None
    pages = {m.group(1): int(m.group(2)) for m in re.finditer(r"^Pages (\w[\w ]*?):\s+(\d+)\.", text, re.M)}
    free = sum(pages.get(k, 0) for k in ("free", "inactive", "speculative", "purgeable"))
    return free * int(size.group(1)) / 1e9


def parse_memory_pressure(text: str) -> int | None:
    """The free-memory percentage macOS's memory_pressure reports (low means the system is under pressure)."""
    m = re.search(r"System-wide memory free percentage:\s*(\d+)%", text)
    return int(m.group(1)) if m else None


def machine(path: str | Path = ".") -> dict:
    """The machine as it is now: OS, chip, cores, memory, free disk under ``path``, software versions."""
    info: dict = {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
                  "apple_silicon": sys.platform == "darwin" and platform.machine() == "arm64",
                  "cores": os.cpu_count(), "python": platform.python_version()}
    if sys.platform == "darwin":
        info["cpu"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        for key, name in (("performance_cores", "hw.perflevel0.physicalcpu"),
                          ("efficiency_cores", "hw.perflevel1.physicalcpu")):
            value = _run(["sysctl", "-n", name])
            info[key] = int(value) if value and value.isdigit() else None
        total = _run(["sysctl", "-n", "hw.memsize"])
        info["memory_total_gb"] = int(total) / 1e9 if total and total.isdigit() else None
        vm = _run(["vm_stat"])
        info["memory_available_gb"] = parse_vm_stat(vm) if vm else None
        pressure = _run(["memory_pressure", "-Q"])
        info["memory_free_percent"] = parse_memory_pressure(pressure) if pressure else None
        info["cpu_bf16"] = _run(["sysctl", "-n", "hw.optional.arm.FEAT_BF16"]) == "1"
    else:
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text()
            meminfo = parse_meminfo(Path("/proc/meminfo").read_text())
        except OSError:
            cpuinfo, meminfo = "", {"total_gb": None, "available_gb": None}
        model = re.search(r"^model name\s*:\s*(.*)$", cpuinfo, re.M)
        info["cpu"] = model.group(1) if model else platform.processor() or None
        info["memory_total_gb"], info["memory_available_gb"] = meminfo["total_gb"], meminfo["available_gb"]
        info["cpu_bf16"] = "amx_bf16" in cpuinfo or "avx512_bf16" in cpuinfo
    info["disk_free_gb"] = shutil.disk_usage(path).free / 1e9
    for module in ("torch", "numpy"):
        try:
            info[module] = __import__(module).__version__
        except ImportError:
            info[module] = None
    return info


def devices() -> dict:
    """PyTorch's devices on this machine: built into this PyTorch, and usable now."""
    import torch

    mps = getattr(torch.backends, "mps", None)
    found = {"cpu": {"available": True},
             "mps": {"built": bool(mps and mps.is_built()), "available": bool(mps and mps.is_available())},
             "cuda": {"built": torch.backends.cuda.is_built(), "available": torch.cuda.is_available()}}
    if found["cuda"]["available"]:
        found["cuda"]["name"] = torch.cuda.get_device_name(0)
    return found


def _small_model(preset: str = "tiny", seed: int = 0):
    import torch

    from ycode.lm.model import GPT, PRESETS, GPTConfig

    torch.manual_seed(seed)
    return GPT(GPTConfig(**PRESETS[preset], vocab_size=512))


def check_device(device: str, *, preset: str = "tiny", rtol: float = 1e-3) -> dict:
    """Run the same forward and backward pass on ``device`` and on the CPU and compare: the loss, and every
    gradient, relative to its size. A device that fails or computes something else is not recommended."""
    import torch

    try:
        reference = _small_model(preset)
        model = _small_model(preset).to(device)
        g = torch.Generator().manual_seed(1)
        idx = torch.randint(0, 512, (2, 32), generator=g)
        targets = torch.randint(0, 512, (2, 32), generator=g)
        _, loss_ref = reference(idx, targets)
        loss_ref.backward()
        _, loss = model(idx.to(device), targets.to(device))
        loss.backward()
        worst = 0.0
        for (name, p_ref), (_, p) in zip(reference.named_parameters(), model.named_parameters()):
            if p_ref.grad is None:
                continue
            scale = float(p_ref.grad.abs().max()) or 1.0
            worst = max(worst, float((p.grad.detach().cpu() - p_ref.grad).abs().max()) / scale)
        loss_diff = abs(loss.item() - loss_ref.item()) / max(abs(loss_ref.item()), 1e-12)
    except Exception as exc:  # an op the device does not support, an out-of-memory, ...
        return {"device": device, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    ok = loss_diff <= rtol and worst <= 10 * rtol
    return {"device": device, "ok": ok, "loss_rel_diff": loss_diff, "grad_rel_diff": worst,
            **({} if ok else {"error": f"differs from the CPU beyond rtol {rtol}"})}


def estimate_gb(params: int, trained: int, *, train: bool, precision: str, batch: int, seq: int, d_model: int,
                n_layer: int) -> float:
    """Rough memory a benchmark needs: the weights, plus for training the gradients and optimizer state of
    the trained weights and the activations of one batch."""
    weight_bytes = 2 if precision in ("bf16", "fp16") and not train else 4
    total = params * weight_bytes
    if train:
        total += trained * 4 * 3  # gradient and two optimizer moments (AdamW), fp32
        total += batch * seq * d_model * n_layer * 4 * 16  # activations kept for the backward pass, roughly
    return total / 1e9


def _sync(device: str) -> None:
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def benchmark(device: str, precision: str, *, preset: str = "small", model_dir: str | Path | None = None,
              train: bool = True, train_layers: int | None = None, batch: int = 2, seq: int = 256,
              steps: int = 3, gen_tokens: int = 32, available_gb: float | None = None) -> dict:
    """Throughput of one configuration: training tokens/s (forward, backward, AdamW step; the top
    ``train_layers`` blocks only, if given) or inference tokens/s (greedy decoding with the KV cache).
    The first step is a warm-up and is not timed."""
    import contextlib

    import torch

    row = {"device": device, "precision": precision, "task": "train" if train else "inference",
           "model": str(model_dir) if model_dir else preset}
    if model_dir:
        from ycode.lm.train import load_checkpoint

        model = load_checkpoint(Path(model_dir), "cpu")[0]
    else:
        model = _small_model(preset)
    cfg = model.cfg
    seq = min(seq, cfg.block_size)
    trainable = list(model.parameters())
    if train and train_layers:
        for p in model.parameters():
            p.requires_grad_(False)
        trainable = [p for b in model.blocks[-train_layers:] for p in b.parameters()]
        for p in trainable:
            p.requires_grad_(True)
    need = estimate_gb(model.num_params(), sum(p.numel() for p in trainable), train=train, precision=precision,
                       batch=batch, seq=seq, d_model=cfg.n_embd, n_layer=cfg.n_layer)
    row.update(params=model.num_params(), estimated_gb=round(need, 4))
    if available_gb is not None and need > MEMORY_SHARE * available_gb:
        return {**row, "skipped": f"needs ~{need:.1f} GB, more than {MEMORY_SHARE:.0%} of the "
                                  f"{available_gb:.1f} GB available"}
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    try:
        if train:
            model.to(device).train()
            optimizer = torch.optim.AdamW(trainable, lr=1e-4)
            autocast = (torch.autocast(device_type=device.split(":")[0], dtype=dtype)
                        if dtype is not None else contextlib.nullcontext())
            g = torch.Generator().manual_seed(0)
            batches = [(torch.randint(0, cfg.vocab_size, (batch, seq), generator=g).to(device),
                        torch.randint(0, cfg.vocab_size, (batch, seq), generator=g).to(device))
                       for _ in range(steps + 1)]
            times = []
            for x, y in batches:
                t = time.perf_counter()
                with autocast:
                    _, loss = model(x, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                _sync(device)
                times.append(time.perf_counter() - t)
            timed = sum(times[1:])
            row.update(tokens_per_s=round(batch * seq * steps / timed, 1), seconds_per_step=round(timed / steps, 3))
        else:
            # 16-bit weights compute under autocast on their own device; activations between layers stay fp32
            model.to(device=device, dtype=dtype or torch.float32).eval()
            autocast = (torch.autocast(device_type=device.split(":")[0], dtype=dtype)
                        if dtype is not None else contextlib.nullcontext())
            prompt = torch.randint(0, cfg.vocab_size, (1, 16), generator=torch.Generator().manual_seed(0)).to(device)
            with torch.no_grad(), autocast:
                model.generate(prompt, 2, temperature=0)  # warm-up
                _sync(device)
                t = time.perf_counter()
                model.generate(prompt, gen_tokens, temperature=0)
                _sync(device)
            row.update(tokens_per_s=round(gen_tokens / (time.perf_counter() - t), 1))
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return row


def recommend(checks: dict, rows: list[dict]) -> dict:
    """The fastest measured configuration, for training and for inference, among devices that passed their
    check. Nothing is recommended for a task with no successful measurement."""
    passed = {d for d, c in checks.items() if c.get("ok")}
    best = {}
    for task in ("train", "inference"):
        fits = [r for r in rows if r["task"] == task and r["device"] in passed and r.get("tokens_per_s")]
        if fits:
            top = max(fits, key=lambda r: r["tokens_per_s"])
            best[task] = {k: top[k] for k in ("device", "precision", "tokens_per_s", "model")}
    return best


def report(*, bench: bool = False, preset: str = "small", model_dir: str | None = None,
           train_layers: int | None = None, steps: int = 3, path: str | Path = ".", log=print) -> dict:
    """Machine, devices, checks and (with ``bench``) benchmarks and the recommendation, as one dict."""
    result = {"machine": machine(path), "devices": devices()}
    usable = [d for d, info in result["devices"].items() if info.get("available")]
    result["checks"] = {d: check_device(d) for d in usable}
    for d, c in result["checks"].items():
        log(f"check {d}: {'ok' if c['ok'] else 'FAILED: ' + c.get('error', '')}")
    if bench:
        rows = []
        for d in usable:
            for task, precisions in (("train", TRAIN_PRECISIONS), ("inference", INFER_PRECISIONS)):
                for precision in precisions.get(d.split(":")[0], ("fp32",)):
                    available = machine(path)["memory_available_gb"]
                    row = benchmark(d, precision, preset=preset, model_dir=model_dir, train=task == "train",
                                    train_layers=train_layers, steps=steps, available_gb=available)
                    rows.append(row)
                    log(f"{task:<9} {d:<4} {precision:<4} " + (f"{row['tokens_per_s']} tokens/s" if "tokens_per_s" in row
                                                         else row.get("skipped") or row.get("error", "")))
        result["benchmarks"] = rows
        result["recommendation"] = recommend(result["checks"], rows)
    return result


def main(args) -> int:
    result = report(bench=args.bench, preset=args.preset, model_dir=args.model, train_layers=args.train_layers,
                    steps=args.steps, path=args.path)
    m = result["machine"]
    print(f"{m['system']} {m['machine']}{' (Apple Silicon)' if m['apple_silicon'] else ''}: {m.get('cpu')}, "
          f"{m['cores']} cores, {m['memory_total_gb'] or 0:.1f} GB memory "
          f"({m['memory_available_gb'] or 0:.1f} GB available), {m['disk_free_gb']:.1f} GB free disk; "
          f"PyTorch {m['torch']}; devices: "
          + ", ".join(d for d, info in result["devices"].items() if info.get("available")))
    if "recommendation" in result:
        for task, best in result["recommendation"].items():
            print(f"recommended for {task}: {best['device']} {best['precision']} ({best['tokens_per_s']} tokens/s "
                  f"on {best['model']})")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1) + "\n")
    return 0
