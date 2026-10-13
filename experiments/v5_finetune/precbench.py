"""fp32 vs bf16-autocast speed on this CPU at v5's real shapes (run with nothing else training)."""

import statistics
import sys
import time

import torch

sys.path.insert(0, "/home/user/Code-agent")
from ycode.lm.model import PRESETS, Block, GPTConfig, rope_tables  # noqa: E402

torch.set_num_threads(4)


def timed(fn, reps=5):
    fn()
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return statistics.median(out)


def matmuls():
    shapes = [(4096, 2048, 5504), (4096, 5504, 2048), (4096, 2048, 3072), (2048, 4096, 5504)]
    for m, k, n in shapes:
        a, b = torch.randn(m, k), torch.randn(k, n)
        fp32 = timed(lambda: a @ b)
        with torch.autocast("cpu", torch.bfloat16, cache_enabled=False):
            bf16 = timed(lambda: a @ b)
        print(f"matmul {m}x{k} @ {k}x{n}: fp32 {fp32:.3f}s  bf16 {bf16:.3f}s  fp32 speedup {bf16 / fp32:.2f}x",
              flush=True)


def block(seq=1024, batch=4):
    cfg = GPTConfig(vocab_size=8192, **PRESETS["v5-1.5b"])
    blk = Block(cfg)
    cos, sin = rope_tables(cfg.n_embd // cfg.n_head, cfg.block_size)
    cos, sin = cos[:seq], sin[:seq]
    x = torch.randn(batch, seq, cfg.n_embd, requires_grad=True)

    def step():
        blk(x, cos, sin).float().pow(2).mean().backward()

    fp32 = timed(step, reps=3)
    with torch.autocast("cpu", torch.bfloat16, cache_enabled=False):
        bf16 = timed(step, reps=3)
    print(f"block fwd+bwd (4x1024): fp32 {fp32:.2f}s  bf16 {bf16:.2f}s  fp32 speedup {bf16 / fp32:.2f}x", flush=True)


if __name__ == "__main__":
    if "--quick" in sys.argv:  # just the transformer block (~1 min)
        block()
    else:
        matmuls()
        block()
