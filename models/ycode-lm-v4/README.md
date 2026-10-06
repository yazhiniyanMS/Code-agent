# YCode-LM v4 (100M parameters)

> Evaluation in progress. Results will be added to this card.

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

## Use

```bash
pip install -e ".[local]"
ycode --local models/ycode-lm-v4
```

Keep all six `model*.pt` files together. Loading reassembles the shards automatically.
