# YCode-LM v4 (100M parameters)

The 100M-parameter YCode-LM, **grown from YCode-LM v3** rather than trained from scratch:
v3's 13 trained layers were kept and 21 new layers were inserted between them, each a copy of its
neighbour with zeroed output projections. The grown model computed exactly what v3 computed, and
continued pretraining then trained the new layers.

| | |
| --- | --- |
| Architecture | decoder-only transformer, 34 layers × 512 wide, 8 query / 2 key-value heads (GQA), QK-norm, RoPE, SwiGLU, RMSNorm |
| Parameters | 100,047,616 |
| Context | 1,024 tokens |
| Tokenizer | byte-level BPE, 8,192 tokens (shared with v2/v3) |
| Weights | bfloat16, **sharded** into `model-0000N-of-00005.pt` files (each < 45 MB, under GitHub's file limit); `model.pt` holds the config, the tokenizer and the shard index |
| Continued pretraining | 3,000 steps × 8,192 tokens ≈ 24.6M tokens on top of v3, Muon (lr 0.015) + AdamW (lr 1e-3), warmup-stable-decay with annealing |
| Instruction tuning | 700 packed steps on the corrected 87k-pair instruction set |
| Hardware | 4-core Intel Xeon CPU, ~760 tokens/s, about 13 hours |

## Results

All four models were evaluated the same way with `ycode-lm eval`. Bits per byte is measured on 11
packages none of them saw in training. The code tasks are executed against unit tests.

| | v1 (6.9M) | v2 (7.7M) | v3 (40.8M) | **v4 (100M)** |
| --- | --- | --- | --- | --- |
| Bits per byte on held-out code (lower is better) | 1.095 | 0.966 | 0.857 | **0.814** |
| Write a function from scratch, pass@1 (30 problems) | 0 / 30 | 0 / 30 | 0 / 30 | **1 / 30** |
| Fix an injected bug, fix@1 (15 functions) | 0 / 15 | 5 / 15 | 4 / 15 | **5 / 15** |

**What this means:**
- v4 models unseen code best of all versions: 5% better than v3 and 26% better than v1.
- It is the first YCode-LM to write a correct function from a description:
  `square(x)` → `return x * x`. That answer also passes extra tests (0, 5, 1.5, -7). One problem
  out of 30 is a small step, not a reliable skill.
- It fixed 5 of 15 injected bugs (`add`, `is_even`, `filter_even`, `celsius_to_fahrenheit`,
  `is_palindrome`).
- **The trade-off:** on the training corpus's own validation split, v4's pretraining loss ended
  slightly *worse* than v3's (1.832 vs 1.805 nats/token). Raising the learning rate again after
  growing disturbed v3's tuned weights, and 24.6M tokens is far too little to fully train 60M new
  parameters. It still generalises better to unseen packages (bits per byte above). More training,
  ideally on a GPU, is the clear next step.

## Use

```bash
pip install -e ".[local]"
ycode --local models/ycode-lm-v4
```

Keep all six `model*.pt` files together. Loading reassembles the shards automatically.
