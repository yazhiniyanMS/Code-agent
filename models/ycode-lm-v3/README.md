# YCode-LM v3 (40.8M parameters)

The largest YCode-LM, trained **from scratch** with this repository's `ycode-lm` pipeline: its own
BPE tokenizer, its own transformer and training loop, no pretrained weights.

| | |
| --- | --- |
| Architecture | decoder-only transformer, 13 layers × 512 wide, 8 query / 2 key-value heads (GQA), QK-norm, RoPE, SwiGLU, RMSNorm |
| Parameters | 40,836,608 |
| Context | 1,024 tokens |
| Tokenizer | byte-level BPE, 8,192 tokens (shared with v2, embedded in `model.pt`) |
| Weights | bfloat16 (81.8 MB), converted to float32 on load |
| Pretraining | 7,300 steps × 8,192 tokens ≈ 60M tokens of Python (same 176 MB corpus as v2), **Muon** optimizer for the transformer matrices + AdamW for embeddings/norms, warmup-stable-decay LR, annealing with 20% instruction data |
| Instruction tuning | 1,400 packed steps on 87k pairs generated from real code. Answers use compact one-line docstrings and only standalone functions for write/complete tasks |
| Hardware | 4-core Intel Xeon CPU (bf16, `torch.compile`), about 10 hours in total |

## Results

All three models were evaluated the same way with `ycode-lm eval`. Bits per byte is measured on 11
packages none of them saw in training (click, anyio, jsonschema, markdown_it, starlette, uvicorn, h11,
attr, pluggy, filelock, idna). The code tasks are executed against unit tests.

| | v1 (6.9M) | v2 (7.7M) | **v3 (40.8M)** |
| --- | --- | --- | --- |
| Bits per byte on held-out code (lower is better) | 1.095 | 0.966 | **0.857** |
| Base model before instruction tuning (pretraining held-out loss) | 2.07 nats/token* | 2.063 | **1.805** |
| Write a function from scratch, pass@1 (30 problems) | 0 / 30 | 0 / 30 | 0 / 30 |
| Fix an injected bug, fix@1 (15 functions) | 0 / 15 | 5 / 15 | 4 / 15 |

\*v1 used a different tokenizer and held-out split, so its per-token loss is not directly comparable.

**What this means:**
- v3 models real code clearly best: it predicts unseen code 11% better than v2 and 22% better than v1.
- It writes short, well-formed answers and can find and fix simple bugs. For example, given
  `result = 1` in a summing loop it answers "The bug is in `result = 1`. It should be `result = 0`."
  and returns the corrected function.
- It does **not** yet write correct functions from a description. On bug fixing, v2 and v3 are
  within noise of each other on 15 problems.
- More data and compute (a GPU) are the way forward. The pipeline supports both unchanged.

In YCode it runs in answer-only mode (no tools).

## Use

```bash
pip install -e ".[local]"
ycode --local models/ycode-lm-v3
ycode-lm chat --model models/ycode-lm-v3
```
