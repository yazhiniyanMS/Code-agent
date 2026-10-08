# YCode-LM v5 (1.5B parameters)

The 1.5-billion-parameter YCode-LM, **grown from YCode-LM v4** and trained entirely on a CPU.
v4's 34 layers × 512 channels were widened 4× to 2048 channels: new channels and attention heads
started with zeroed outputs, so the grown model computed exactly what v4 computed (checked: identical
loss on the same batches before training). Continued pretraining then trained the new capacity.

| | |
| --- | --- |
| Architecture | decoder-only transformer, 34 layers × 2048 wide, 32 query / 8 key-value heads (GQA), QK-norm, RoPE, SwiGLU, RMSNorm |
| Parameters | 1,523,202,304 |
| Context | 1,024 tokens |
| Tokenizer | byte-level BPE, 8,192 tokens (shared with v2–v4) |
| Weights | bfloat16, **sharded** into 103 `model-*-of-00103.pt` files (each < 45 MB, under GitHub's file limit); `model.pt` holds the config, the tokenizer and the shard index |
| Continued pretraining | 600 steps × 4,096 tokens ≈ 2.5M tokens, Lion (lr 3e-5), warmup-stable-decay with 20% instruction data during the decay |
| Instruction tuning | 60 packed steps, Lion (lr 1.5e-5) |
| Hardware | 4-core Intel Xeon CPU, 15 GB of RAM. 55–75 tokens/s on a CPU with AMX bf16 units, 8–9 tokens/s on one without |

## Training a 1.5B model in 15 GB of RAM

Weights, gradients and AdamW state for 1.5B parameters take ~24 GB in fp32. v5 used a low-memory
mode (`ycode-lm train --optimizer lion`):

- **Lion** keeps one momentum value per weight (AdamW keeps two), stored in bf16: 3 GB.
- **Updates inside backward:** each weight is updated as soon as its gradient is ready, and the
  gradient is freed. A full set of gradients (6 GB) never exists.
- Gradient checkpointing, bf16 autocast without the cast-weight cache, memory-mapped checkpoints.

Peak memory was about 12 GB. Training ran in restartable segments that saved checkpoints, and it
survived three container restarts.

## Results

Evaluated with `ycode-lm eval`, the same way as every earlier version. Bits per byte is measured on
11 Python packages none of the models saw in training. The code tasks are executed against unit tests.

| | v1 (6.9M) | v2 (7.7M) | v3 (40.8M) | v4 (100M) | **v5 (1,523M)** |
| --- | --- | --- | --- | --- | --- |
| Bits per byte on held-out code (lower is better) | 1.095 | 0.966 | 0.857 | 0.814 | **0.758** |
| Write a function from scratch, pass@1 (30 problems) | 0 / 30 | 0 / 30 | 0 / 30 | **1 / 30** | 0 / 30 |
| Fix an injected bug, fix@1 (15 functions) | 0 / 15 | **5 / 15** | 4 / 15 | **5 / 15** | 4 / 15 |

**What this means:**
- v5 models unseen code best of all versions: **7% better than v4**, the largest step between any two
  versions, after only ~2.5M tokens of extra training.
- That has not yet turned into solving more tasks. v5 fixed 4 bugs (`total`, `filter_even`,
  `celsius_to_fahrenheit`, `is_palindrome`) and wrote no fully correct function. With 30 and 15 tasks,
  a difference of one is within noise. Its instruction tuning was cut to 60 steps by the slow CPU.
- A 1.5B model would ideally see tens of billions of training tokens; v5 saw a few million on top of
  v4. It is badly under-trained, and more training (ideally on GPUs: `ycode-lm autotrain`, `--shard`)
  is the clear next step.
- YCode uses v5 as its default model on machines with at least 4 GB of free memory, and v4 otherwise. v4 is just as good at the benchmark tasks and much faster on a CPU, so `ycode --local v4` is a good choice on slow machines.

## Use

```bash
pip install -e .
ycode                         # v5 is the default; loads in bfloat16: about 3 GB of RAM
ycode-lm chat --model models/ycode-lm-v5
```

Keep all 104 `model*.pt` files together. Loading reassembles the shards automatically.
