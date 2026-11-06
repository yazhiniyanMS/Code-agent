# YCode development log

One entry per development cycle, newest last. Read this and GitHub before starting a cycle; never assume a
previous container's or machine's local state still exists. (An earlier log was kept outside Git and was lost
with its container on 2026-11-03; this file replaces it.)

## Standing state (as of 2026-11-06)

- **Checkpoints:** round 54, C2, R1, S1 and every other checkpoint since V5 were only on the previous container's
  disk (`/home/user/lmwork`) and are gone. No copy exists elsewhere. Round 54 cannot be rebuilt from surviving
  records. Branch `claude/checkpoint-recovery-audit` (`fd9c270`, `experiments/recovery/README.md`) has the
  dependency matrix and the reproducibility experiment.
- **V13:** paused. It needs a verified baseline (S1 recovered, or a new baseline formally selected and
  pre-registered) and durable checkpoint storage tested by a full round trip.
- **Durable storage:** a private Hugging Face repository is the candidate.
  - Blocked: `huggingface.co` is not allowed by this environment's network policy, there is no token, and
    creating the repository needs approval.
  - The free tier gives 100 GB of private storage; the lost chain was about 8.3 GB.
- **Tokenizer:** regenerable from `models/ycode-lm-v5`. The full recorded hashes match
  (`b766ca56...` embedded, `7079ff7b...` as tokenizer.json).

## Cycle 1: 2026-11-06, 06:53-07:20 UTC

- **Machine:** cloud container, not the MacBook.
  - Ubuntu 24.04, x86_64, 4-core Intel Xeon at 2.10 GHz (AMX, avx512_bf16).
  - No GPU, no Metal/MPS.
  - 15 GiB RAM; 9.6-13 GB of disk free.
  - Python 3.11.18 venv with torch 2.14.1+cu130 (the recorded version) and numpy 2.4.6.
- **GitHub:** `gh` is not usable here (the environment's GH_TOKEN is invalid). The GitHub connector is
  authenticated as `yazhiniyanMS` (id 83769849), and PRs are created through it.
- **Start of cycle:**
  - Open PRs: #26 (V13 plan + identity-based checkpoint resolution) and #17 (memory guard).
  - No reviews, comments or CI runs on either; the repository has no CI.
  - Both merge into main without conflicts. main is `da84953` (unchanged since 2026-10-30).
- **Implemented:** `ycode-lm hardware` (`ycode/lm/hardware.py`):
  - machine report: chip, Apple Silicon performance/efficiency cores, memory and macOS memory pressure, disk,
    versions;
  - CPU/MPS/CUDA availability, and a correctness check of each device against the CPU;
  - with `--bench`, training and inference tokens/s per device and precision, a memory estimate that skips
    what will not fit, and a recommendation from the measurements;
  - `--model` measures a real checkpoint; `--train-layers` trains only the top blocks.
- **Files:** `ycode/lm/hardware.py` (new), `tests/test_hardware.py` (new), `ycode/lm/cli.py` (one
  subcommand), `README.md` (step 0 of the training walkthrough), `DEVLOG.md` (this file).
- **Tests:**
  - 9 new; the macOS paths run on captured sysctl, vm_stat and memory_pressure output.
  - Full suite: 458 passed, 2 skipped, 0 failed. One skip needs `huggingface_hub` (not installed); the other
    needs the production tokenizer at the lost `/home/user/lmwork/data`.
- **Measured here** (small preset):

  | | fp32 | bf16 |
  | --- | --- | --- |
  | training | 7,519 tokens/s | 16,600 tokens/s |
  | inference | 882 tokens/s | 614 tokens/s |

- **Commit and PR:** `4221001`. PR https://github.com/yazhiniyanMS/Code-agent/pull/33, author verified as
  `yazhiniyanMS`.
- **Found, not yet changed (inferred from code; needs a Mac to confirm):**
  1. **The local provider on MPS:** `LocalLM.precision()` uses CPU autocast for any non-CUDA device. On MPS
     the shipped 1.5B model loads in bf16, so generation likely fails with a dtype mismatch.
  2. **The trainer on MPS:** autocast is enabled only on CPU and CUDA, so `--precision bf16` on MPS silently
     trains in fp32.
- **Resources at the end:** 10.3 GB of disk free, 14 GiB RAM available.
- **Remaining:**
  - Run `ycode-lm hardware --bench` (and `--model models/ycode-lm-v5`) on the MacBook.
  - Durable storage decisions.
  - A V13 baseline decision.
  - #17, #26 and #33 await review.
- **Next recommended task:** make the local provider and the trainer use the model's own device for
  autocast. Fail loudly where a precision cannot run there, instead of silently using another. Add tests
  for that logic, and confirm on the MacBook with `ycode-lm hardware --bench --model models/ycode-lm-v5`.
