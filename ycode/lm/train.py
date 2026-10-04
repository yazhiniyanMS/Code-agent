"""Training loops for pretraining (next-token prediction on code) and
instruction tuning (SFT, loss only on answers)."""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ycode.lm.model import GPT, PRESETS, GPTConfig
from ycode.lm.tokenizer import BPETokenizer

Log = Callable[[str], None]
CHECKPOINT_NAME = "model.pt"


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cpu_supports_bf16() -> bool:
    """True on CPUs with native bfloat16 units (AMX / AVX512-BF16)."""
    try:
        flags = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    return "amx_bf16" in flags or "avx512_bf16" in flags


def pick_precision(requested: str, device: str) -> str:
    if requested != "auto":
        return requested
    if device == "cuda":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp32"
    if device == "cpu" and cpu_supports_bf16():
        return "bf16"
    return "fp32"


@dataclass
class TrainConfig:
    stage: str = "pretrain"  # pretrain | sft
    preset: str = "small"
    model_overrides: dict = field(default_factory=dict)
    batch_size: int = 16
    grad_accum: int = 1
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 100
    max_steps: int = 2000
    max_minutes: float | None = None
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_iters: int = 20
    log_interval: int = 10
    device: str = "auto"
    precision: str = "auto"  # auto | fp32 | bf16
    seed: int = 1337
    init_from: str | None = None  # checkpoint dir to start from (required for sft)
    resume: bool = False


# ------------------------------------------------------------- checkpoints


def save_checkpoint(out_dir: Path, model: GPT, tok: BPETokenizer, *, step: int, stage: str,
                    val_loss: float | None, optimizer: torch.optim.Optimizer | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "ycode-lm",
        "version": 1,
        "config": model.cfg.to_dict(),
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "tokenizer": tok.to_dict(),
        "step": step,
        "stage": stage,
        "val_loss": val_loss,
    }
    path = out_dir / CHECKPOINT_NAME
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    if optimizer is not None:
        torch.save({"optimizer": optimizer.state_dict(), "step": step}, out_dir / "optim.pt")
    (out_dir / "info.json").write_text(json.dumps({
        "step": step, "stage": stage, "val_loss": val_loss, "params": model.num_params(),
        "config": model.cfg.to_dict(),
    }, indent=2))
    return path


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[GPT, BPETokenizer, dict]:
    path = Path(path)
    if path.is_dir():
        path = path / CHECKPOINT_NAME
    if not path.is_file():
        raise FileNotFoundError(f"No model checkpoint at {path}")
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "ycode-lm":
        raise ValueError(f"{path} is not a YCode model checkpoint")
    model = GPT(GPTConfig(**payload["config"]))
    model.load_state_dict(payload["model"])
    model.to(device)
    tok = BPETokenizer.from_dict(payload["tokenizer"])
    return model, tok, payload


# ------------------------------------------------------------------ data


class PretrainData:
    def __init__(self, data_dir: Path, block_size: int, batch_size: int, device: str) -> None:
        self.splits = {s: np.memmap(data_dir / f"{s}.bin", dtype=np.uint16, mode="r") for s in ("train", "val")}
        for split, arr in self.splits.items():
            if len(arr) <= block_size + 1:
                raise ValueError(f"{split}.bin has only {len(arr)} tokens; need more than block_size={block_size}")
        self.block_size, self.batch_size, self.device = block_size, batch_size, device

    def batch(self, split: str, gen: torch.Generator):
        data = self.splits[split]
        ix = torch.randint(len(data) - self.block_size - 1, (self.batch_size,), generator=gen)
        x = torch.stack([torch.from_numpy(data[i: i + self.block_size].astype(np.int64)) for i in ix.tolist()])
        y = torch.stack([torch.from_numpy(data[i + 1: i + 1 + self.block_size].astype(np.int64)) for i in ix.tolist()])
        return x.to(self.device), y.to(self.device), None


class SFTData:
    def __init__(self, data_dir: Path, block_size: int, batch_size: int, device: str, pad_id: int,
                 val_fraction: float = 0.05) -> None:
        self.tokens = np.memmap(data_dir / "sft_tokens.bin", dtype=np.uint16, mode="r")
        self.mask = np.memmap(data_dir / "sft_mask.bin", dtype=np.uint8, mode="r")
        index = np.load(data_dir / "sft_index.npy")
        if len(index) < 2:
            raise ValueError("Not enough instruction examples for SFT (need at least 2).")
        n_val = max(1, int(len(index) * val_fraction))
        self.index = {"val": index[:n_val], "train": index[n_val:]}
        self.block_size, self.batch_size, self.device, self.pad_id = block_size, batch_size, device, pad_id

    def batch(self, split: str, gen: torch.Generator):
        index = self.index[split]
        picks = torch.randint(len(index), (self.batch_size,), generator=gen).tolist()
        seqs = []
        for p in picks:
            start, length = index[p]
            length = min(int(length), self.block_size + 1)
            seqs.append((self.tokens[start: start + length].astype(np.int64),
                         self.mask[start: start + length].astype(np.int64)))
        width = max(len(t) for t, _ in seqs) - 1
        x = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        y = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        m = torch.zeros((len(seqs), width), dtype=torch.long)
        for row, (t, mk) in enumerate(seqs):
            n = len(t) - 1
            x[row, :n] = torch.from_numpy(t[:-1])
            y[row, :n] = torch.from_numpy(t[1:])
            m[row, :n] = torch.from_numpy(mk[1:])
        return x.to(self.device), y.to(self.device), m.to(self.device)


# --------------------------------------------------------------- training


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model: GPT, data, cfg: TrainConfig, gen: torch.Generator, autocast=None) -> dict[str, float]:
    model.eval()
    out = {}
    autocast = autocast if autocast is not None else contextlib.nullcontext()
    for split in ("train", "val"):
        with autocast:
            losses = [model(*data.batch(split, gen))[1].item() for _ in range(cfg.eval_iters)]
        out[split] = float(sum(losses) / len(losses))
    model.train()
    return out


def train(data_dir: Path, out_dir: Path, cfg: TrainConfig, *, log: Log = print) -> dict:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    if device == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))

    start_step = 0
    if cfg.init_from or cfg.resume:
        source = Path(cfg.init_from) if cfg.init_from else out_dir
        model, tok, payload = load_checkpoint(source, device)
        if cfg.resume:
            start_step = int(payload.get("step", 0))
        log(f"Loaded {source} (step {payload.get('step')}, stage {payload.get('stage')})")
    else:
        if cfg.stage == "sft":
            raise ValueError("SFT needs a pretrained model: pass --init-from <model dir>")
        tok = BPETokenizer.load(data_dir / "tokenizer.json")
        if cfg.preset not in PRESETS:
            raise ValueError(f"unknown preset {cfg.preset!r}; choose from {', '.join(PRESETS)}")
        model_cfg = GPTConfig(vocab_size=tok.vocab_size, **{**PRESETS[cfg.preset], **cfg.model_overrides})
        model = GPT(model_cfg).to(device)

    block = model.cfg.block_size
    if cfg.stage == "sft":
        data = SFTData(data_dir, block, cfg.batch_size, device, pad_id=tok.eot_id)
    else:
        data = PretrainData(data_dir, block, cfg.batch_size, device)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95), fused=(device == "cuda"),
    )
    if cfg.resume and (out_dir / "optim.pt").is_file():
        state = torch.load(out_dir / "optim.pt", map_location=device, weights_only=True)
        optimizer.load_state_dict(state["optimizer"])

    precision = pick_precision(cfg.precision, device)
    autocast = torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16,
                              enabled=(precision == "bf16" and device in ("cuda", "cpu")))
    gen = torch.Generator().manual_seed(cfg.seed + start_step)
    eval_gen = torch.Generator().manual_seed(cfg.seed + 999)

    log(f"Model: {model.num_params() / 1e6:.2f}M parameters, context {block}, device {device}, "
        f"precision {precision}, stage {cfg.stage}")
    t0 = time.time()
    tokens_seen = 0
    best_val = None
    history = []
    step = start_step
    model.train()
    while step < cfg.max_steps:
        if cfg.max_minutes is not None and (time.time() - t0) / 60 >= cfg.max_minutes:
            log(f"Time limit of {cfg.max_minutes} minutes reached.")
            break
        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss_total = 0.0
        for _ in range(cfg.grad_accum):
            x, y, m = data.batch("train", gen)
            with autocast:
                _, loss = model(x, y, loss_mask=m)
            (loss / cfg.grad_accum).backward()
            loss_total += loss.item() / cfg.grad_accum
            tokens_seen += x.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % cfg.log_interval == 0:
            elapsed = time.time() - t0
            log(f"step {step:>6} | loss {loss_total:.3f} | lr {lr:.2e} | "
                f"{tokens_seen / max(elapsed, 1e-9):,.0f} tok/s | {elapsed / 60:.1f} min")
        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            losses = estimate_loss(model, data, cfg, eval_gen, autocast)
            history.append({"step": step, **losses})
            log(f"eval step {step}: train {losses['train']:.3f} | val {losses['val']:.3f}")
            if best_val is None or losses["val"] < best_val:
                best_val = losses["val"]
            save_checkpoint(out_dir, model, tok, step=step, stage=cfg.stage, val_loss=losses["val"],
                            optimizer=optimizer)

    if not history or history[-1]["step"] != step:
        losses = estimate_loss(model, data, cfg, eval_gen, autocast)
        history.append({"step": step, **losses})
        log(f"final eval step {step}: train {losses['train']:.3f} | val {losses['val']:.3f}")
        save_checkpoint(out_dir, model, tok, step=step, stage=cfg.stage, val_loss=losses["val"],
                        optimizer=optimizer)
    summary = {"steps": step, "minutes": (time.time() - t0) / 60, "tokens": tokens_seen,
               "final": history[-1], "params": model.num_params(), "out_dir": str(out_dir)}
    (out_dir / "train_log.json").write_text(json.dumps({"summary": summary, "history": history}, indent=2))
    return summary
