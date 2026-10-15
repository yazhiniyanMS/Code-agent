# YCode-LM V5 → V6: training improvements

Status: **in progress**. Measured results are marked; nothing here comes from a run that did not happen.

## Starting point (measured)

- Model: YCode-LM v5, a **decoder-only** GPT (34 layers, width 2048, GQA 32/8, RoPE, SwiGLU, tied
  embeddings, 1,523,202,304 parameters, context 1024, byte-level BPE with 8,192 tokens). There is no
  encoder; "decoder targets" below means the causal next-token shift.
- Best checkpoint: round 54 (`/home/user/lmwork/loop/anchor`, never modified). On the clean
  evaluation sets: dev 27/44, unseen-v1 2/53, unseen-v2 0/71, so **all_clean = 29/168**;
  benchmark 22/65 (reported only).
- Host: 4 CPU cores, 15 GB RAM, no AMX/AVX512-BF16 (fp32 training is 3x faster than emulated
  bf16 here).

## What limits learning (measured)

1. **The objective is implemented as intended.** Targets are the inputs shifted by one with the
   mask shifted with them; packing fills every 1,025-token row (no padding); the loss is
   sum(loss × weight) / sum(weight) per step, prompt tokens weight 0, answer tokens 1–8 (body of a
   written function 3, changed tokens of a bug fix 8), end-of-answer marker trained. No change
   needed.
2. **The fine-tuning data has almost nothing left to teach.** One round of the v5 mix: 1,768 write
   examples with 717 distinct bodies, mostly one-line arithmetic (`return x + 10`), one prompt
   template. After 54 rounds the model's loss on it is ~0.07 per step (often 0.00); 24 steps of
   it cost round 54 1.5 dev items (control A, v51-m3).
3. **Compute.** Training the top 24 of 34 layers with activation checkpointing: 160 s per
   4 × 1024-token step (25.5 tokens/s), peak 11.3 GB. Without checkpointing the activations
   of 24 layers at this batch would need ~17 GB (estimated from the layer shapes, not run). Evaluating dev + unseen-v1 + unseen-v2 +
   benchmark takes ~65 min per checkpoint (greedy decoding, long answers).

## Changes to the pipeline

| Change | Where | Why |
| --- | --- | --- |
| Idiom curriculum: 76 families of verified programs built from general idioms, several constants, phrasings, implementations and prompt templates, plus bug-fix variants | `ycode/lm/idioms.py`, round-data options `idioms_write`, `idioms_bugfix` | New information the model can learn from (finding 2) |
| Held-out exclusion at AST-skeleton level (same code up to names, constants, operators, min/max, ...) | `leakage.skeleton`, `idioms.held_out_check` | No near-twin of any held-out item is trained on, so held-out gains measure transfer |
| unseen-v2: 71 tasks written by a separate agent that never read the repository, filtered automatically, frozen before the curriculum data was built | `ycode/lm/unseen2.py`, `experiments/v6/make_unseen2.py` | The curriculum was designed after reading unseen-v1's failures; v2 is the independent check |
| Learning probe: code-only loss on reference solutions (trained curriculum programs, new family members, clean dev, unseen-v1, unseen-v2) | `experiments/v6/probe_learning.py` | A held-out validation loss that moves (the data's own validation loss is ~0.01 and says nothing); 8 min instead of 65 |
| Items-lost category rule, fixed before any V6 result | `experiment.compare(category_rule="items_lost")` | The v5.1 rule fired on failures that only changed type |
| Runner keeps a bf16 copy of each trained model | spec `keep_models` | Candidates can be averaged, probed and delivered |

## Runs

RUNS_PENDING

## Decision

DECISION_PENDING

## Reproduction

REPRO_PENDING
