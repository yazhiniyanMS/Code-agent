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

from ycode.lm.model import DEFAULT_PRESET, GPT, PRESETS, GPTConfig, rope_tables
from ycode.lm.optim import build_optimizer, cuda_has_bf16
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
    if device.startswith("cuda"):
        # bf16 needs Ampere or newer (A100, RTX 30xx+); older GPUs such as the T4 use fp16.
        return "bf16" if cuda_has_bf16() else "fp16"
    if device == "cpu" and cpu_supports_bf16():
        return "bf16"
    return "fp32"


@dataclass
class TrainConfig:
    stage: str = "pretrain"  # pretrain | sft
    preset: str = DEFAULT_PRESET
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
    precision: str = "auto"  # auto | fp32 | bf16 | fp16 (fp16 uses dynamic loss scaling)
    seed: int = 1337
    init_from: str | None = None  # checkpoint dir to start from (required for sft)
    resume: bool = False
    # v2 efficiency / quality options
    schedule: str = "wsd"  # wsd (warmup-stable-decay) | cosine
    decay_frac: float = 0.2  # wsd: final fraction of training spent decaying the LR
    anneal_mix: float = 0.2  # pretrain: share of instruction data mixed in during the decay phase
    pack: bool = True  # sft: pack several examples per sequence instead of padding
    compile: bool = False  # torch.compile the model for training
    grad_checkpoint: bool = False  # recompute activations in backward: much less memory, ~30% slower
    save_interval: int | None = None  # checkpoint every N steps without evaluating (cheap crash safety)
    until_step: int | None = None  # stop early at this step (segmented runs); schedule still spans max_steps
    # adamw | muon (Muon for hidden matrices + AdamW for the rest) | lion (low memory: one bf16
    # state per weight, applied during backward so gradients are never all held at once)
    optimizer: str = "adamw"
    muon_lr: float = 0.02
    # Multi-GPU only: shard weights, gradients and optimizer state across GPUs (FSDP) instead of
    # replicating them (DDP). Needed when a model does not fit on one GPU, e.g. 1.5B on 16 GB T4s.
    shard: bool = False
    # Train only the top N transformer blocks (plus the final norm); the rest stay frozen. Frozen
    # layers need no gradients or stored activations, so each step costs far less: the way to
    # fine-tune a 1.5B model on a CPU.
    train_layers: int | None = None


# ------------------------------------------------------------- checkpoints


def save_checkpoint(out_dir: Path, model: GPT, tok: BPETokenizer, *, step: int, stage: str,
                    val_loss: float | None, optimizer=None, state: dict | None = None) -> Path:
    """Write model.pt (+ optim.pt). ``state`` overrides model.state_dict() (sharded training)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    state = state if state is not None else model.state_dict()
    payload = {
        "format": "ycode-lm",
        "version": model.cfg.arch_version,
        "config": model.cfg.to_dict(),
        "model": {k: v.detach().cpu() for k, v in state.items()},
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
        # Write-then-rename: a resumed optimizer's state may be memory-mapped from the old file,
        # and overwriting that file in place would pull its pages away (SIGBUS).
        optim_tmp = out_dir / "optim.pt.tmp"
        torch.save({"optimizer": optimizer.state_dict(), "step": step}, optim_tmp)
        os.replace(optim_tmp, out_dir / "optim.pt")
    (out_dir / "info.json").write_text(json.dumps({
        "step": step, "stage": stage, "val_loss": val_loss, "params": model.num_params(),
        "config": model.cfg.to_dict(),
    }, indent=2))
    return path


def load_checkpoint(path: Path, device: str = "cpu", *,
                    dtype: torch.dtype | None = None) -> tuple[GPT, BPETokenizer, dict]:
    """Load a model. ``dtype`` (e.g. torch.bfloat16) keeps the weights in that precision: a 1.5B
    model then needs 3 GB of RAM instead of 6 GB, with no full-precision copy made on the way."""
    path = Path(path)
    if path.is_dir():
        path = path / CHECKPOINT_NAME
    if not path.is_file():
        raise FileNotFoundError(f"No model checkpoint at {path}")
    # Memory-map on CPU: a 1.5B checkpoint is 6 GB, and mapping avoids holding a second copy in RAM.
    mmap = {"mmap": True} if device == "cpu" else {}
    payload = torch.load(path, map_location=device, weights_only=True, **mmap)
    if payload.get("format") != "ycode-lm":
        raise ValueError(f"{path} is not a YCode model checkpoint")
    if "shards" in payload:  # large exported models are split into several files
        state: dict[str, torch.Tensor] = {}
        for name in payload["shards"]:
            shard_path = path.parent / name
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing model shard {shard_path}")
            state.update(torch.load(shard_path, map_location=device, weights_only=True, **mmap))
        for key, source in payload.get("tied", {}).items():
            state[key] = state[source]
        payload["model"] = state
    if dtype is not None:
        # Build without allocating weights, then adopt the checkpoint's tensors directly.
        with torch.device("meta"):
            model = GPT(GPTConfig(**payload["config"]))
        state = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in payload["model"].items()}
        model.load_state_dict(state, assign=True)
        model.head.weight = model.embed.weight  # keep the embedding/head weights tied
        cos, sin = rope_tables(model.cfg.n_embd // model.cfg.n_head, model.cfg.block_size)
        model.rope_cos, model.rope_sin = cos, sin  # position tables stay fp32 for accuracy
    else:
        model = GPT(GPTConfig(**payload["config"]))
        model.load_state_dict(payload["model"])
    payload["model"] = {}  # the weights now live in the model; drop the duplicate
    model.to(device)
    tok = BPETokenizer.from_dict(payload["tokenizer"])
    return model, tok, payload


def export_checkpoint(src: Path, dst: Path, *, dtype: str = "bf16", max_shard_mb: float | None = None) -> Path:
    """Write a slim, inference-only copy of a checkpoint (no optimizer state).

    bf16 halves the file size; weights are converted back to float32 on load."""
    src = Path(src)
    path = src / CHECKPOINT_NAME if src.is_dir() else src
    if not path.is_file():
        raise FileNotFoundError(f"No model checkpoint at {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "ycode-lm":
        raise ValueError(f"{path} is not a YCode model checkpoint")
    if dtype not in ("bf16", "fp32"):
        raise ValueError("dtype must be bf16 or fp32")
    target = torch.bfloat16 if dtype == "bf16" else torch.float32
    converted: dict[int, torch.Tensor] = {}  # keep tied weights (embedding/head) stored once
    model_state = {}
    for key, tensor in payload["model"].items():
        ptr = tensor.data_ptr()
        if ptr not in converted:
            converted[ptr] = tensor.to(target) if tensor.is_floating_point() else tensor
        model_state[key] = converted[ptr]
    payload["model"] = model_state
    shard_files: list[str] = []
    if max_shard_mb is not None:
        # Split the weights into files under max_shard_mb (e.g. GitHub's 100 MB file limit).
        # Tied tensors are stored once and re-linked on load.
        seen: dict[int, str] = {}
        tied: dict[str, str] = {}
        shards: list[dict[str, torch.Tensor]] = [{}]
        sizes = [0]
        limit = max_shard_mb * 1e6
        for key, tensor in model_state.items():
            ptr = tensor.data_ptr()
            if ptr in seen:
                tied[key] = seen[ptr]
                continue
            seen[ptr] = key
            nbytes = tensor.numel() * tensor.element_size()
            if sizes[-1] and sizes[-1] + nbytes > limit:
                shards.append({})
                sizes.append(0)
            shards[-1][key] = tensor
            sizes[-1] += nbytes
        shard_files = [f"model-{i + 1:05d}-of-{len(shards):05d}.pt" for i in range(len(shards))]
        payload["shards"] = shard_files
        payload["tied"] = tied
        payload["model"] = {}
    payload["exported_dtype"] = dtype
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for old_shard in dst.glob("model-*-of-*.pt"):
        old_shard.unlink()  # never leave stale shards from a previous export
    torch.save(payload, dst / CHECKPOINT_NAME)
    if shard_files:
        for name, shard in zip(shard_files, shards):
            torch.save(shard, dst / name)
    params = sum(v.numel() for v in converted.values())
    (dst / "info.json").write_text(json.dumps({
        "step": payload.get("step"), "stage": payload.get("stage"), "val_loss": payload.get("val_loss"),
        "version": payload.get("version"), "params": params, "dtype": dtype, "config": payload["config"],
        "files": [CHECKPOINT_NAME, *shard_files],
    }, indent=2))
    return dst / CHECKPOINT_NAME


# ------------------------------------------------------------------ data


class PretrainData:
    """Random windows from the token stream. During the LR decay phase a share
    of each batch can come from instruction data ("annealing"), so the base
    model already knows the chat format before SFT."""

    def __init__(self, data_dir: Path, block_size: int, batch_size: int, device: str) -> None:
        self.splits = {s: np.memmap(data_dir / f"{s}.bin", dtype=np.uint16, mode="r") for s in ("train", "val")}
        for split, arr in self.splits.items():
            if len(arr) <= block_size + 1:
                raise ValueError(f"{split}.bin has only {len(arr)} tokens; need more than block_size={block_size}")
        sft_path = data_dir / "sft_tokens.bin"
        self.sft = np.memmap(sft_path, dtype=np.uint16, mode="r") if sft_path.is_file() else None
        if self.sft is not None and len(self.sft) <= block_size + 1:
            self.sft = None
        self.offsets = np.arange(block_size + 1)
        self.block_size, self.batch_size, self.device = block_size, batch_size, device

    def _windows(self, data, n: int, gen: torch.Generator) -> np.ndarray:
        starts = torch.randint(len(data) - self.block_size - 1, (n,), generator=gen).numpy()
        return np.asarray(data[starts[:, None] + self.offsets], dtype=np.int64)  # one vectorised gather

    def batch(self, split: str, gen: torch.Generator, mix: float = 0.0):
        n_mix = int(round(self.batch_size * mix)) if (self.sft is not None and split == "train") else 0
        rows = self._windows(self.splits[split], self.batch_size - n_mix, gen)
        if n_mix:
            rows = np.concatenate([rows, self._windows(self.sft, n_mix, gen)])
        t = torch.from_numpy(rows)
        return t[:, :-1].to(self.device), t[:, 1:].to(self.device), None


class SFTData:
    """Instruction examples with an answer-only loss mask.

    With packing (default), each row is filled with several whole examples
    back to back, so no compute is wasted on padding."""

    def __init__(self, data_dir: Path, block_size: int, batch_size: int, device: str, pad_id: int,
                 val_fraction: float = 0.05, pack: bool = True) -> None:
        self.tokens = np.memmap(data_dir / "sft_tokens.bin", dtype=np.uint16, mode="r")
        self.mask = np.memmap(data_dir / "sft_mask.bin", dtype=np.uint8, mode="r")
        index = np.load(data_dir / "sft_index.npy")
        if len(index) < 2:
            raise ValueError("Not enough instruction examples for SFT (need at least 2).")
        n_val = max(1, int(len(index) * val_fraction))
        self.index = {"val": index[:n_val], "train": index[n_val:]}
        self.block_size, self.batch_size, self.device, self.pad_id = block_size, batch_size, device, pad_id
        self.pack = pack

    def _example(self, index, gen: torch.Generator):
        start, length = index[int(torch.randint(len(index), (1,), generator=gen))]
        return (self.tokens[start: start + length].astype(np.int64),
                self.mask[start: start + length].astype(np.int64))

    def _row(self, index, gen: torch.Generator):
        need = self.block_size + 1
        if not self.pack:
            t, m = self._example(index, gen)
            return t[:need], m[:need]
        toks, masks, size = [], [], 0
        while size < need:
            t, m = self._example(index, gen)
            toks.append(t)
            masks.append(m)
            size += len(t)
        return np.concatenate(toks)[:need], np.concatenate(masks)[:need]

    def batch(self, split: str, gen: torch.Generator, mix: float = 0.0):
        seqs = [self._row(self.index[split], gen) for _ in range(self.batch_size)]
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


def lr_at(step: int, cfg: TrainConfig, progress: float | None = None) -> float:
    """Learning rate for this step.

    ``progress`` (0..1) is the larger of step- and time-based progress, so a
    run stopped by ``max_minutes`` still completes its LR decay instead of
    ending at a high learning rate.
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if progress is None:
        progress = step / max(1, cfg.max_steps)
    progress = min(1.0, max(0.0, progress))
    if cfg.schedule == "cosine":
        warm = cfg.warmup_steps / max(1, cfg.max_steps)
        p = min(1.0, max(0.0, (progress - warm) / max(1e-9, 1 - warm)))
        return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * p))
    decay_start = 1.0 - cfg.decay_frac
    if progress < decay_start:
        return cfg.lr
    p = (progress - decay_start) / max(1e-9, cfg.decay_frac)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * p))


def in_decay_phase(cfg: TrainConfig, progress: float) -> bool:
    return cfg.schedule == "wsd" and progress >= 1.0 - cfg.decay_frac


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


def _dist_info() -> tuple[int, int, int]:
    """(world_size, rank, local_rank) from torchrun's environment variables."""
    return (int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0")),
            int(os.environ.get("LOCAL_RANK", "0")))


def freeze_below(model: GPT, train_layers: int) -> None:
    """Freeze everything except the top ``train_layers`` blocks and the final norm."""
    for p in model.parameters():
        p.requires_grad_(False)
    for block in model.blocks[-train_layers:]:
        for p in block.parameters():
            p.requires_grad_(True)
    for p in model.norm.parameters():
        p.requires_grad_(True)


def compact_state(model: GPT) -> dict:
    """State dict with frozen weights in bf16 (they do not change) and trained ones in fp32."""
    trained = {name for name, p in model.named_parameters() if p.requires_grad}
    return {k: (v.to(torch.bfloat16) if k not in trained and v.is_floating_point() else v)
            for k, v in model.state_dict().items()}


def _local(t: torch.Tensor) -> torch.Tensor:
    """This rank's part of a sharded (DTensor) tensor; plain tensors pass through."""
    return t.to_local() if hasattr(t, "to_local") else t


def _shard_model(model: GPT, dev_type: str, world: int, param_dtype) -> None:
    """FSDP2: shard each transformer block (and the embeddings at the root) across all ranks.
    Weights are gathered layer by layer in ``param_dtype`` for compute; gradients are
    reduce-scattered in fp32, and each rank keeps only its shard of the optimizer state."""
    from torch.distributed.device_mesh import init_device_mesh

    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    except ImportError:  # PyTorch < 2.6
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
    mesh = init_device_mesh(dev_type, (world,))
    policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32)
    for block in model.blocks:
        fully_shard(block, mesh=mesh, mp_policy=policy)
    fully_shard(model, mesh=mesh, mp_policy=policy)


def _full_state_dict(model: GPT) -> dict:
    """Gather a sharded model's full weights onto rank 0's CPU (collective: every rank calls it)."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    return get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))


def _to_local_tree(obj):
    if isinstance(obj, dict):
        return {k: _to_local_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_local_tree(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return _local(obj).detach().clone().cpu()
    return obj


def _optim_shard_name(rank: int, world: int) -> str:
    return f"optim-rank{rank}-of-{world}.pt"


def _load_sharded_optimizer(optimizer, state: dict) -> None:
    """Load one rank's optimizer shard and turn its tensors back into DTensors."""
    from torch.distributed.tensor import DTensor

    optimizer.load_state_dict(state)
    for opt in optimizer.optimizers:
        for p, st in opt.state.items():
            if not isinstance(p, DTensor):
                continue
            local = p.to_local()
            for key, value in list(st.items()):
                if isinstance(value, torch.Tensor) and not isinstance(value, DTensor) and value.dim() > 0:
                    if value.shape != local.shape:
                        raise ValueError(f"optimizer shard shape {tuple(value.shape)} != {tuple(local.shape)}")
                    st[key] = DTensor.from_local(value.to(local.device, local.dtype), p.device_mesh,
                                                 p.placements, shape=p.shape, stride=p.stride(),
                                                 run_check=False)


class LossScaler:
    """Dynamic loss scaling for fp16 (GPUs without bf16, e.g. the T4).

    fp16 gradients underflow without scaling, so the loss is multiplied by a large factor
    before backward and the gradients divided by it afterwards. If any gradient overflows
    the step is skipped and the scale halved; after enough clean steps it doubles again.
    Works with any optimizer, including the Muon + AdamW combination."""

    def __init__(self, init_scale: float = 2.0 ** 16, growth_interval: int = 1000) -> None:
        self.scale = init_scale
        self.growth_interval = growth_interval
        self.good_steps = 0
        self.skipped = 0

    def unscale_and_check(self, params, all_ranks: Callable[[bool], bool] | None = None) -> bool:
        """Divide grads by the scale; return True if they are all finite.

        With sharded gradients each rank only sees its part, so ``all_ranks`` combines
        the verdicts (every rank must skip or apply the same step)."""
        finite = True
        for p in params:
            if p.grad is not None:
                grad = _local(p.grad)
                grad.div_(self.scale)
                if finite and not torch.isfinite(grad).all():
                    finite = False
        return all_ranks(finite) if all_ranks is not None else finite

    def update(self, finite: bool) -> None:
        if finite:
            self.good_steps += 1
            if self.good_steps % self.growth_interval == 0:
                self.scale = min(self.scale * 2.0, 2.0 ** 24)
        else:
            self.skipped += 1
            self.good_steps = 0
            self.scale = max(self.scale / 2.0, 1.0)


def train(data_dir: Path, out_dir: Path, cfg: TrainConfig, *, log: Log = print) -> dict:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    world, rank, local_rank = _dist_info()
    distributed = world > 1
    device = pick_device(cfg.device)
    dev_type = "cuda" if device.startswith("cuda") else device
    if distributed:
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if dev_type == "cuda" else "gloo")
        if dev_type == "cuda":
            device = f"cuda:{local_rank}"
            torch.cuda.set_device(local_rank)
    main = rank == 0
    shard = cfg.shard and distributed
    # Sharded models are built on the CPU and moved to the GPUs shard by shard.
    load_device = "cpu" if shard else device
    if not main:
        log = lambda *_: None  # noqa: E731 - only rank 0 talks
    torch.manual_seed(cfg.seed)
    if device == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // world))

    start_step = 0
    has_checkpoint = (out_dir / CHECKPOINT_NAME).is_file()
    if cfg.resume and not has_checkpoint and not cfg.init_from:
        log(f"No checkpoint in {out_dir} yet; starting a new run.")
    if cfg.init_from or (cfg.resume and has_checkpoint):
        # Resuming continues the run in out_dir; otherwise start from init_from.
        resumable = cfg.resume and has_checkpoint
        source = out_dir if (resumable or not cfg.init_from) else Path(cfg.init_from)
        model, tok, payload = load_checkpoint(source, load_device)
        if cfg.resume and source == out_dir:
            start_step = int(payload.get("step", 0))
        log(f"Loaded {source} (step {payload.get('step')}, stage {payload.get('stage')})")
    else:
        if cfg.stage == "sft":
            raise ValueError("SFT needs a pretrained model: pass --init-from <model dir>")
        tok = BPETokenizer.load(data_dir / "tokenizer.json")
        if cfg.preset not in PRESETS:
            raise ValueError(f"unknown preset {cfg.preset!r}; choose from {', '.join(PRESETS)}")
        model_cfg = GPTConfig(vocab_size=tok.vocab_size, **{**PRESETS[cfg.preset], **cfg.model_overrides})
        model = GPT(model_cfg).to(load_device)
    model.grad_checkpoint = cfg.grad_checkpoint
    if cfg.train_layers:
        freeze_below(model, cfg.train_layers)
    precision = pick_precision(cfg.precision, dev_type)
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    if shard:
        _shard_model(model, dev_type, world, amp_dtype if precision in ("fp16", "bf16") else None)

    block = model.cfg.block_size
    if cfg.stage == "sft":
        data = SFTData(data_dir, block, cfg.batch_size, device, pad_id=tok.eot_id, pack=cfg.pack)
    else:
        data = PretrainData(data_dir, block, cfg.batch_size, device)

    optimizer = build_optimizer(model, kind=cfg.optimizer, lr=cfg.lr, muon_lr=cfg.muon_lr,
                                weight_decay=cfg.weight_decay, fused=(dev_type == "cuda" and not shard))
    in_backward = cfg.optimizer == "lion"
    if in_backward and (cfg.grad_accum != 1 or distributed or precision == "fp16"):
        raise ValueError("--optimizer lion updates weights during backward: it needs --grad-accum 1, "
                         "a single process and no fp16")
    optim_shard = out_dir / _optim_shard_name(rank, world)
    if shard and cfg.resume and has_checkpoint:
        if optim_shard.is_file():
            try:
                _load_sharded_optimizer(optimizer, torch.load(optim_shard, map_location="cpu",
                                                              weights_only=True)["optimizer"])
            except (ValueError, KeyError, IndexError) as exc:
                log(f"Optimizer state does not match ({exc}); starting it fresh.")
                optimizer = build_optimizer(model, kind=cfg.optimizer, lr=cfg.lr, muon_lr=cfg.muon_lr,
                                            weight_decay=cfg.weight_decay)
        else:
            log(f"No optimizer state for {world} sharded ranks; starting it fresh.")
    elif cfg.resume and has_checkpoint and (out_dir / "optim.pt").is_file():
        state = torch.load(out_dir / "optim.pt", map_location=device, weights_only=True,
                           **({"mmap": True} if device == "cpu" else {}))["optimizer"]
        try:
            optimizer.load_state_dict(state if "optimizers" in state else {"optimizers": [state]})
        except (ValueError, KeyError, IndexError):
            log("Optimizer state does not match (different --optimizer?); starting it fresh.")

    if in_backward:
        optimizer.optimizers[0].attach()
    autocast = torch.autocast(device_type=dev_type if dev_type in ("cuda", "cpu") else "cpu", dtype=amp_dtype,
                              enabled=(precision in ("bf16", "fp16") and dev_type in ("cuda", "cpu")),
                              # The cast-weight cache is a second (16-bit) copy of the model: 3 GB at 1.5B.
                              cache_enabled=not in_backward)
    scaler = LossScaler() if precision == "fp16" else None
    # Each rank draws different batches; the eval generator is shared so evals are comparable.
    gen = torch.Generator().manual_seed(cfg.seed + start_step + 7919 * rank)
    eval_gen = torch.Generator().manual_seed(cfg.seed + 999)

    train_model = model
    if cfg.compile:
        try:
            train_model = torch.compile(model)
        except Exception as exc:  # noqa: BLE001 - compilation is an optional speed-up
            log(f"torch.compile unavailable ({exc}); continuing without it.")
    compiled = train_model is not model
    if distributed and not shard:
        from torch.nn.parallel import DistributedDataParallel as DDP

        train_model = DDP(train_model, device_ids=[local_rank] if dev_type == "cuda" else None)

    log(f"Model: YCode-LM v{model.cfg.arch_version}, {model.num_params() / 1e6:.2f}M parameters, "
        f"context {block}, device {device}{f' x{world}' if distributed else ''}, precision {precision}, "
        f"optimizer {cfg.optimizer}, stage {cfg.stage}{', compiled' if compiled else ''}"
        f"{', grad checkpointing' if cfg.grad_checkpoint else ''}{', sharded (FSDP)' if shard else ''}")

    def all_ranks_finite(finite: bool) -> bool:
        flag = torch.tensor([1.0 if finite else 0.0], device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def checkpoint(val_loss: float | None) -> None:
        """Save model + optimizer. Sharded: every rank joins the gather and writes its optimizer shard."""
        if not shard:
            save_checkpoint(out_dir, model, tok, step=step, stage=cfg.stage, val_loss=val_loss,
                            optimizer=optimizer, state=compact_state(model) if cfg.train_layers else None)
            return
        full = _full_state_dict(model)
        if main:
            save_checkpoint(out_dir, model, tok, step=step, stage=cfg.stage, val_loss=val_loss, state=full)
        del full
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = optim_shard.with_suffix(".tmp")
        torch.save({"optimizer": _to_local_tree(optimizer.state_dict()), "step": step}, tmp)
        os.replace(tmp, optim_shard)
        dist.barrier()
    t0 = time.time()
    tokens_seen = 0
    best_val = None
    history = []
    step = start_step
    model.train()
    annealing_logged = False
    stop_at = min(cfg.max_steps, cfg.until_step) if cfg.until_step else cfg.max_steps
    while step < stop_at:
        progress = step / max(1, cfg.max_steps)
        if cfg.max_minutes is not None:
            progress = max(progress, (time.time() - t0) / 60 / cfg.max_minutes)
        if distributed:
            # Every rank must use rank 0's clock: same LR, same stop decision, no hangs.
            synced = torch.tensor([progress], dtype=torch.float64,
                                  device=device if dev_type == "cuda" else "cpu")
            dist.broadcast(synced, 0)
            progress = float(synced.item())
        if cfg.max_minutes is not None and progress >= 1.0:
            log(f"Time limit of {cfg.max_minutes} minutes reached.")
            break
        lr = lr_at(step, cfg, progress)
        optimizer.set_lr_scale(lr / cfg.lr)
        mix = cfg.anneal_mix if (cfg.stage == "pretrain" and in_decay_phase(cfg, progress)) else 0.0
        if mix and not annealing_logged:
            log(f"Decay phase: annealing with {mix:.0%} instruction data.")
            annealing_logged = True
        loss_total = 0.0
        loss_scale = scaler.scale if scaler else 1.0
        for micro in range(cfg.grad_accum):
            x, y, m = data.batch("train", gen, mix=mix)
            last = micro == cfg.grad_accum - 1
            # DDP: all-reduce gradients only after the last micro-batch. (FSDP reduce-scatters every
            # micro-batch; keeping unsharded gradients around would defeat the point of sharding.)
            sync = contextlib.nullcontext() if (last or not distributed or shard) else train_model.no_sync()
            with sync:
                with autocast:
                    _, loss = train_model(x, y, loss_mask=m)
                (loss * (loss_scale / cfg.grad_accum)).backward()
            loss_total += loss.item() / cfg.grad_accum
            tokens_seen += x.numel() * world
        finite = (scaler.unscale_and_check(model.parameters(), all_ranks_finite if shard else None)
                  if scaler else True)
        if in_backward:
            pass  # Lion already updated every weight during backward (sign-based, so no clipping)
        elif finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        else:
            log(f"step {step + 1}: fp16 gradient overflow, skipping update (loss scale {loss_scale:g})")
        if scaler:
            scaler.update(finite)
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % cfg.log_interval == 0:
            elapsed = time.time() - t0
            log(f"step {step:>6} | loss {loss_total:.3f} | lr {lr:.2e} | "
                f"{tokens_seen / max(elapsed, 1e-9):,.0f} tok/s | {elapsed / 60:.1f} min")
        # Sharded models need every rank for a forward pass or a save; replicated ones only rank 0.
        acting = main or shard
        if acting and (step % cfg.eval_interval == 0 or step == cfg.max_steps):
            losses = estimate_loss(model, data, cfg, eval_gen, autocast)
            history.append({"step": step, **losses})
            log(f"eval step {step}: train {losses['train']:.3f} | val {losses['val']:.3f}")
            if best_val is None or losses["val"] < best_val:
                best_val = losses["val"]
            checkpoint(losses["val"])
        elif acting and cfg.save_interval and step % cfg.save_interval == 0:
            checkpoint(history[-1]["val"] if history else None)
            log(f"checkpoint saved at step {step}")

    summary: dict = {}
    if (main or shard) and (not history or history[-1]["step"] != step):
        losses = estimate_loss(model, data, cfg, eval_gen, autocast)
        history.append({"step": step, **losses})
        log(f"final eval step {step}: train {losses['train']:.3f} | val {losses['val']:.3f}")
        checkpoint(losses["val"])
    if main:
        summary = {"steps": step, "minutes": (time.time() - t0) / 60, "tokens": tokens_seen,
                   "final": history[-1], "params": model.num_params(), "out_dir": str(out_dir),
                   "world_size": world, "precision": precision,
                   "fp16_skipped_steps": scaler.skipped if scaler else 0}
        (out_dir / "train_log.json").write_text(json.dumps({"summary": summary, "history": history}, indent=2))
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return summary
