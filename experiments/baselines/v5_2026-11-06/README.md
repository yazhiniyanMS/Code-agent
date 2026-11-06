# YCode-LM v5, evaluated on 2026-11-06 (cloud container)

- **What:** the shipped v5 (`models/ycode-lm-v5`, the only model intact after the checkpoint loss on
  2026-11-03), evaluated with the existing definitions (`eval-v5.1-1`) on bench, dev and unseen-v1.
- **Not included:** unseen-v2, which stays sealed.
- **Evaluation only:** no training, and no model selection. Choosing a new baseline is a separate decision.
- **Why:**
  - to check that evaluation reproduces in the new environment;
  - to measure v5 here for that decision;
  - to see where it fails.

## Identity and setup

- **Model:** 104 files. Combined `files_sha256` is `386660d44cec…` (the SHA-256 of each file, in
  `info.json` order); `model.pt`, the index file holding the tokenizer, is `68fe1ce80b6d…`. Full values are
  in `summary.json`.
- **Code:** main `936ac99`. Greedy decoding with bf16 weights under CPU autocast, the local provider's
  default for models over 500M parameters.
- **Commands:**
  `ycode-lm eval --model models/ycode-lm-v5 --problems {standard,fresh,dev,unseen} --device cpu --answers-out …`,
  then `ycode-lm analyze … --json …`.
  - `bench.jsonl` = standard's 45 answers plus fresh's 20 writes.
- **Host:** cloud container, Intel Xeon at 2.10 GHz, 4 cores, AMX, no GPU; Python 3.11.18, torch
  2.14.1+cu130.

## Results

| Set | Write a function | Fix a bug | Official | Clean (not contaminated) | Robust |
| --- | --- | --- | --- | --- | --- |
| bench (65) | 0 / 30 standard, 0 / 20 fresh | 4 / 15 (`total`, `filter_even`, `celsius_to_fahrenheit`, `is_palindrome`) | 4 | 3 / 61 | 4 |
| dev (62) | 2 / 47 (`all_odd`, `has_x`) | 1 / 15 (`product`) | 3 | 3 / 44 | 2 |
| unseen-v1 (53) | 0 / 53 | — | 0 | 0 / 53 | 0 |

**Reproducibility:**
- The published v5 model card (`models/ycode-lm-v5/README.md`) reports 0/30 standard and 4/15 bug fixes,
  with the same four functions. This reproduces exactly on this host.
- The standard and fresh runs each answered the 15 bug fixes, in separate processes; all 15 answers were
  byte-identical.

**Against round 54** (recorded in `experiments/v51/MISSION_REPORT.md`; round 54 itself is lost):

| Score | v5 (here) | Round 54 (recorded) |
| --- | --- | --- |
| dev clean / 44 | 3 | 27 |
| unseen-v1 / 53 | 0 | 2 |
| primary (dev clean + unseen) / 97 | 3 | 29 |
| bench / 65 | 4 | 22 |
| dev / 62 | 3 | 39 |

Round 54's 72-step fine-tune lifted dev from 3 to 39, but unseen-v1 only from 0 to 2. That is the
historical finding again: learning within trained task families, almost no transfer to new ones.

## Where v5 loses points (`ycode-lm analyze`)

| Set | Formatting | Instruction following | Hallucination | Coding | Factual | Reasoning |
| --- | --- | --- | --- | --- | --- | --- |
| bench | 5 | 13 | 18 | 6 | 2 | 17 |
| dev | 7 | 17 | 9 | 9 | 0 | 17 |
| unseen-v1 | 8 | 3 | 9 | 11 | 0 | 22 |

Every failed item, with its category and reason, is in `analysis_*.json`.

## Cost

The model loads in under a second (bf16 shards, memory-mapped).

| Run | Problems | Time | Per problem |
| --- | --- | --- | --- |
| standard | 45 | 616 s | 13.7 s |
| fresh | 35 (20 writes + the same 15 bug fixes) | 249 s | — |
| dev | 62 | 526 s | — |
| unseen-v1 | 53 | 539 s | — |

Peak memory for an evaluation run was about 3.8 GB.

## Limits

- Scores on 15-65 items move by one or two with any change; small differences are noise.
- These numbers describe v5, not round 54 or S1. A V13 baseline built on v5 would need its own
  pre-registration.
