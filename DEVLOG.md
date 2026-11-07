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

## Cycle 2: 2026-11-06, 07:25 UTC onwards (5-hour engineering mission, cloud container)

- **Machine:** cloud container, not the MacBook.
  - Ubuntu 24.04, x86_64, 4-core Intel Xeon at 2.10 GHz (AMX, avx512_bf16).
  - 15 GiB RAM, no cgroup memory limit set.
  - ext4 disk with 9.6-10.3 GB free throughout; 16 GB RAM disk.
  - Python 3.11.18, torch 2.14.1+cu130.
- **GPU:** none. CUDA is built in, but `torch.cuda.is_available()` is False and there is no `/dev/nvidia*`.
  Everything ran on the CPU.
- **Network:** GitHub and PyPI reachable; huggingface.co blocked.
- **GitHub:** the connector is authenticated as `yazhiniyanMS`; `gh` is unusable (invalid GH_TOKEN).
  Every PR below was verified on GitHub as authored by `yazhiniyanMS` (id 83769849). Every commit's author and
  committer is `yazhiniyanMS <83769849+yazhiniyanMS@users.noreply.github.com>`.
- **Start of cycle:**
  - #33 was merged (main `936ac99`).
  - #26 and #17 open, with no reviews, comments or CI (the repository has no CI); both merge cleanly.

### A. Device-aware precision: PR #34 (`6563bb7`)
- **Broken, reproduced on main:**
  - `LocalLM(dtype="fp16")` on CPU loaded bf16 weights.
  - For a model on MPS, `LocalLM.precision()` built a CPU autocast context, which cannot apply to MPS tensors.
- **Broken, from the code:**
  - The trainer enabled autocast only for CPU and CUDA, so `--precision bf16`/`fp16` on MPS trained in fp32.
  - Explicit precisions were never checked.
  - PyTorch disables autocast with only a warning when it cannot run (CUDA with no GPU, verified here).
- **Fix:** `ycode/lm/precision.py` resolves (request, device) into a plan: actual precision, autocast on the
  device's own type (entered and checked), the reason for `auto`, and any fallback.
  - Explicit requests run as asked or raise `UnsupportedPrecision`.
  - The trainer resolves before loading anything, logs the plan and records it in `train_log.json`.
  - `LocalLM` and `ycode-lm info` use or show it.
- **Tests:** 9 new. MPS and CUDA are mocked (dispatch logic only, not Apple or NVIDIA execution). Full suite
  466 passed, 3 skipped.
- **Limitation:** MPS behaviour is unverified on Apple hardware.

### B. Coding-agent reliability: PR #35 (`d77ea3c`), PR #39 (`ce23161`, stacked on #35), PR #40 (`40cbe5b`)
- **#35, five failure modes, each with a test failing on main:**
  1. The same failing call repeated until the 50-step limit. Now it is flagged at the 2nd failure and the task
     stops as STUCK at the 3rd.
  2. Identical calls in one response all ran; now they run once.
  3. `write_file` overwrote files the agent had never read, or that changed since it read them. Now refused,
     using a SHA-256 of what was last seen.
  4. `write_file` with unencodable content truncated the file to 0 bytes (700 → 0 on main). Writes are now
     encoded first, then written atomically (temp file, fsync, rename, mode kept, read back).
  5. The summary said "completed" when the last command failed; it now says the command failed.
- **#39:** the conversation had no size limit, so a long session could outgrow the model's context, and after
  that every request fails. The oldest tool outputs now give way to a note past 400,000 characters; the newest
  two results and all user and assistant messages stay whole.
- **#40:** "Changes:" missed files changed through `run_command`. Now taken from before/after git status
  plus content hashes; the user's own earlier edits are not listed.
- **Tests:** #35 9 new, suite 467 passed, 2 skipped. #39 2 new, 469 passed. #40 2 new, 460 passed.
- **Audited, no change needed:** destructive git commands (`reset --hard`, `checkout --`, `clean`,
  force-push, `rebase`, `branch -D`) all require approval.
- **Limitations:**
  - Atomic replacement breaks hard links and needs a writable folder.
  - The context budget is counted in characters, not tokens.

### D. Checkpoint integrity: PR #36 (`6d7333d`)
- **Broken, each with a test failing on main:**
  - A resume could pair step N's weights with step N-k's optimizer state, after a crash between the two
    renames.
  - A new run silently replaced an existing checkpoint in `--out`.
  - Saves were not fsynced, and `info.json` was written in place.
- **Fix:**
  - On resume, the optimizer's step is checked against the model's; on a mismatch the optimizer state starts
    fresh, with a message.
  - `--overwrite` is required to replace an existing checkpoint.
  - Saves go through a fsynced temporary file, then a rename.
- **Tests:** 3 new, with failures injected part-way through a write. Suite 461 passed, 2 skipped.
- **Related:** identity-based loading and parent checks are in #26.

### C. Evaluation sets and tests: PR #37 (`ebc7799`)
- **Audit:** bench 65, dev 62, unseen 53 and unseen2 71 are pairwise disjoint by name and prompt, with no
  repeats within a split. Now guarded by 2 tests; no definitions changed.
- **Fix:** the contrast-token test had silently skipped since `/home/user/lmwork` was lost. It now uses the
  tokenizer embedded in `models/ycode-lm-v5`, checked against round 54's recorded hash; 0 skipped.

### E. Performance measurement: PR #38 (`8533112`, stacked on #34)
- **Change:** `ycode-lm hardware` now estimates memory before loading, loads as the local provider does, and
  reports load time, time to first token, decoding tokens/s and memory. `--save-bench` times a save while
  keeping the 5 GB buffer.
- **Measured here** (each configuration in its own process):

  | Model | Task | Result |
  | --- | --- | --- |
  | Real ycode-lm-v5 (1.52B), bf16 | inference | load 0.71 s, first token 0.36 s (64-token prompt), 7.1 tokens/s, 3.75 GB |
  | Real ycode-lm-v5, fp32 | inference | load 36.2 s, first token 0.82 s, 6.8 tokens/s, 6.7 GB (9.7 GB peak) |
  | Real ycode-lm-v5, top 2 layers, batch 1 × 128 | training step | 84 (fp32) / 110 (bf16) tokens/s, 7.6 GB |
  | Real ycode-lm-v5 | `eval --problems dev --limit 3` | 6 problems in 53.1 s (8.9 s each), 3.77 GB peak |
  | Synthetic `small` preset | training | 10.2k (fp32) / 13.3k (bf16) tokens/s |
  | Synthetic `small` preset | inference | 736 (fp32) / 500 (bf16) tokens/s |
  | Synthetic `v5-385m` preset | save | 1.51 GB in 2.23 s |

  The 6.1 GB v5 save was skipped by the 5 GB disk buffer. Synthetic numbers do not describe the 1.5B model.

### F. Shipped v5 evaluated on this host: PR #42 (`5ae89d6`)
- **Why:** v5 is the only intact model. This checks that evaluation reproduces here, and measures it for a
  baseline decision. Evaluation only: unseen-v2 stays sealed, and no definitions changed.
- **Reproduced exactly:** the published v5 standard benchmark, 0/30 writes and 4/15 bug fixes (`total`,
  `filter_even`, `celsius_to_fahrenheit`, `is_palindrome`). The 15 bug-fix answers were byte-identical across
  two runs.
- **Baseline:** bench 4/65 (clean 3/61); dev 3/62 (clean 3/44); unseen-v1 0/53.
  - Primary score (dev clean + unseen, /97): v5 3, round 54 (recorded) 29.
  - Round 54's fine-tune lifted dev from 3 to 39 but unseen-v1 only from 0 to 2: weak transfer to new task
    families.
- **Failures:** on unseen-v1, points are lost mostly to reasoning (22) and coding (11).
- **Record:** answers, per-item analysis and `summary.json` (model hashes, environment, timings) in
  `experiments/baselines/v5_2026-11-06/`.

### G. Repeated calls that make no progress: PR #43 (`2afc75b`, `c86abd8`; stacked on #35)
- **Broken:** #35 stopped repeated failing calls, but a call that succeeds without progress (re-reading an
  unchanged file) could repeat until the step limit.
- **Fix:** an identical call with an identical result is flagged from the 2nd time and stopped as STUCK at
  the 4th. A changed result resets the count.
- **Tests:** 2 new. Full suite 469 passed, 2 skipped. Merges cleanly with #39.

### End-to-end check
`ycode --local v5 -p "Write a Python function add(a, b) ..."` on the combined build answered in 13 s and exited
0, with the local-model warning shown. As v5's scores predict, the code was wrong.

### Integration
- All twelve open PRs (#17, #26, #34-#43) merge into main without conflicts, in the order #34, #35, #36, #37,
  #38, #39, #40, #43, #42, #41, #26, #17.
- Full suite on the combined result (local branch, not pushed): **511 passed, 3 skipped, 0 failed**. The skips:
  - real CUDA (no GPU);
  - `huggingface_hub` not installed;
  - #17's memory-guard test, which needs round 54 on the machine.

### Status at the end of the cycle
- **Model/checkpoints:** unchanged. Round 54, C2, R1 and S1 are missing; V13 stays paused; no training run.
- **Resources:** disk 10.29 GB free at the end, the same as the start of measurements (benchmark temp folders
  removed). Peak process memory: 9.7 GB (the fp32 load of v5).
- **Remaining:**
  - MPS verification on Apple hardware.
  - Durable checkpoint storage (network, token, approval).
  - A V13 baseline decision.
  - #17, #26 and #34-#43 await review.
- **Next recommended tasks:**
  1. Review and merge #34-#43 in the order above. #38 needs #34 first; #39 and #43 need #35 first.
  2. Run `ycode-lm hardware --bench --model models/ycode-lm-v5` on the MacBook to confirm the precision
     behaviour on MPS.
  3. Decide the V13 baseline and durable storage.
