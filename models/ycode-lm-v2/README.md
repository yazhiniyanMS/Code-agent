# YCode-LM v2 (7.7M parameters)

A small programming language model trained **from scratch** with this repository's `ycode-lm`
pipeline: its own BPE tokenizer, its own transformer and training loop, no pretrained weights.
It is published here so `ycode --local` works right after cloning, with no API key and no training.

| | |
| --- | --- |
| Architecture | decoder-only transformer, 8 layers × 256 wide, 8 query / 2 key-value heads (GQA), QK-norm, RoPE, SwiGLU, RMSNorm |
| Parameters | 7,738,112 |
| Context | 512 tokens |
| Tokenizer | byte-level BPE, 8,192 tokens (embedded in `model.pt`) |
| Weights | bfloat16 (15.6 MB), converted to float32 on load |
| Pretraining | 10,000 steps, ~82M tokens of Python (176 MB: the standard library plus installed packages), warmup-stable-decay LR, annealing with 20% instruction data |
| Instruction tuning | 2,400 packed steps on 109k pairs generated from real code: write and explain functions, complete stubs, write docstrings, explain classes, fix injected bugs |
| Hardware | 4-core Intel Xeon CPU (bf16, `torch.compile`) |

## Results

Measured with `ycode-lm eval` on 11 packages the model never saw (click, anyio, jsonschema,
markdown_it, starlette, uvicorn, h11, attr, pluggy, filelock, idna):

| | v1 (6.9M) | **v2 (this model)** |
| --- | --- | --- |
| Bits per byte on held-out code (lower is better) | 1.095 | **0.966** |
| pass@1 on 30 executed coding problems | 0 / 30 | 0 / 30 |

It writes fluent, well-formed Python and follows the instruction formats, but its code is usually
wrong. It is a learning/research model, not a coding assistant. In YCode it runs in answer-only
mode (no tools).

## Use

```bash
pip install -e ".[local]"
ycode --local models/ycode-lm-v2
ycode-lm chat --model models/ycode-lm-v2
ycode-lm eval --model models/ycode-lm-v2 --heldout path/to/code/it/never/saw
```

To train your own (or continue from this one), see "Your own LLM" in the main README.
