# YCode-LM v5.1: diagnosis, fixes and the controlled experiment

**Status:** phases 1, 2 and 4 are done, tested and pushed. The phase 3 runner is built and passed
a real-model smoke test. **The full A–E experiment has not been run: it needs about 15 to 23
hours of CPU and is waiting for approval.** No performance claim below comes from a run that has
not happened.

## 1. What the audit found (details: `AUDIT.md`)

| | Finding | Status |
| --- | --- | --- |
| C1, C2 | 16 benchmark and dev items had exact behavioural equivalents in the fine-tuning data ("sum of the non-zero numbers" for `total`, "less than zero" for `is_negative`, …). Contaminated slots: 18 of 62 on dev, 4 of 65 on the benchmark. | **Fixed** for future data. The round-54 checkpoint has seen them, so scores are also reported without them ("clean"). |
| C3 | Some held-out tests are too weak to catch wrong code. Wrong `first_word` (`split()[-2]`) passes in **every** run; wrong `fibonacci` passed in round 52. | Harness **unchanged**. A "robust" score is reported next to it. |
| C4 | 11–12% of each round's rows are exact duplicates. No true contradictions. | Dedup available as a data option (arms C–E). |
| C5, C6 | No validation loss was ever measured. The logged loss is one weighted batch. My earlier "the model memorised its data" was unsupported, and I withdrew it. | **Measured now** (section 4). |
| C7 | No NaN/Inf, gradient-norm, invalid-batch or checkpoint-integrity checks. | **Fixed.** |
| C8 | Skill functions were injected into every production round by default. | **Off by default**; opt in with `skills: true`. Whether they help is arm A vs B. |
| Verified correct | Causal masking (future tokens change nothing earlier, bit for bit). Target shift and answer-only loss mask. Padding never weighted. LR schedule. Deterministic evaluation. Round 53's rejection rule. | — |

## 2. Baseline, re-scored (existing runs: no new training)

The abl2 baseline: round-54 checkpoint, 72 more steps, skills off, two seeds.

| Seed | Dev | Dev clean (/44) | Dev robust | Bench | Bench clean (/61) | Bench robust | Hallucinations (dev / bench) | Invalid outputs (dev / bench) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1101 | 35 | 24 | 35 | 22 | 19 | 21 | 1 / 4 | 1 / 0 |
| 1102 | 38 | 28 | 35 | 22 | 19 | 20 | 0 / 3 | 0 / 0 |

**Seed variance:** official dev 35 vs 38, clean dev 24 vs 28; the benchmark is 22 under both
seeds. Three evaluation runs of one checkpoint are identical (Experiment A), so all of this
variance comes from training.

**Points lost by category** (mean of the two seeds):

| Set | Instruction following | Reasoning | Factual | Hallucination | Coding | Formatting |
| --- | --- | --- | --- | --- | --- | --- |
| Dev | 14 | 10.5 | 0 | 0.5 | 0 | 0.5 |
| Benchmark | 20.5 | 12 | 5.5 | 3.5 | 1.5 | 0 |

Per-example analyses are in `results/rescored_abl23.json` and the saved answer files.

**All 14 single changes of abl2/abl3, re-scored on clean dev:** 23 to 26 of 44, inside the
baseline's 24–28. None is distinguishable from seed noise (`results/rescored_abl23.json`).

## 3. Changes: retained, rejected, inconclusive

| Change | Verdict | Evidence |
| --- | --- | --- |
| Remove held-out equivalents from training data | **Retained** (correctness, not a score claim) | AUDIT C1; `test_no_training_function_computes_a_held_out_item` |
| Skill families off by default, opt-in kept | **Retained as a default**; performance effect **inconclusive** | abl2: −1 dev and +3 dev hallucinations on 1 seed. Under the v5.1 rule, one seed decides nothing; arm A vs B will. |
| Health checks (NaN stop, gradient norm, batch checks, hashes, per-step metrics) | **Retained** | tests in `tests/test_v51.py` |
| Manifests, hygiene reports, resumable runs | **Retained** | same |
| Dedup, contradiction filter, replay size | **Inconclusive** until arms C–E run | abl2's `data_dedup` also removed legitimate loop renderings, so it was not a clean test |
| Learning rate ×½/×2, steps ×2, batch 2, seq 512, averaging, fresh constants ×5, contrast ×3 | **Inconclusive** (no detectable effect) | abl2/abl3, clean dev 23–26 vs baseline 24–28 |

## 4. Training health

- **Tiny-dataset overfit:** a tiny model fine-tuned on 3 examples drives loss below 0.05 and
  answers them verbatim. The training path can learn.
- **Generalisation, measured on the round-54 checkpoint** (`results/anchor_diagnose.json`, 64
  rows per split, answer-token loss):

  | Rows from the training generators | Held-out rows, same generators | Unseen tasks (dev, reference answers) |
  | --- | --- | --- |
  | 0.094 | 0.035 | **0.220** |

  Rows the model never saw are predicted as well as training rows, so there is **no evidence of
  memorised rows**. The large gap is to new concepts (6× higher loss), which matches the error
  analysis: the model writes a trained neighbour's body.
- **Smoke run on the real model** (2 steps, arms A and E): gradient norm 0.08–0.11, no empty
  batches, no NaN, checkpoint hashes verified, weights finite. Training peak memory 11.3 GB
  (limit ≈ 14.3 GB), evaluation 7.7 GB. About 60 s per step (63–71 tokens/s).

## 5. The A–E experiment (ready, not run)

`experiments/v51/arms.json`. Every arm starts from the round-54 checkpoint with the same budget
(72 Lion steps, lr 1.5e-5, batch 4, top 24 blocks, bf16), the same decoding and harness, and seeds
1201/1202/1203. The leakage fix applies to all arms.

| Arm | Data | Compared with |
| --- | --- | --- |
| A | existing mix: skills on, duplicates kept, 200 replay rows | — |
| B | A without skill injection | A |
| C | B without exact duplicates or contradictions | B |
| D | C with a small replay set (60 rows, drawn like the 200) | C |
| E | C without replay | C, D |

Decision rule: clean dev, paired by seed. **Retained** only if the mean paired gain clears a
one-sided 95% t-test and no error category loses ≥ 2 points on average. **Rejected** on the
mirror image or category damage. **Inconclusive** otherwise. The benchmark is reported, never
used to decide.

**Cost, from the smoke run:** about 90 minutes per run (data 1.5–4 min, training about 74 min,
dev and benchmark evaluation about 13 min, diagnostics about 4 min). One run at a time
(memory). Restarts resume from a mid-run checkpoint.

| Plan | Runs | Wall time |
| --- | --- | --- |
| 5 arms × 3 seeds | 15 | **≈ 22–23 h** |
| 5 arms × 2 seeds, then a 3rd seed only where a comparison is close | 10 + up to 5 | ≈ 15 h + 1.5 h per extra run |

**Power:** with a clean-dev seed difference of about 4 points, three paired seeds can only
detect gains of about 4 or more clean dev points. Smaller real effects will show as
"inconclusive".

## 6. Unresolved risks

- **The starting checkpoint is contaminated.** It was trained on equivalents of 16 held-out
  items, and no data fix undoes that. Clean scores exclude those items, which leaves 44 dev and
  61 benchmark items.
- **The dev set is small and noisy** (clean seed-to-seed difference 4 of 44).
- **The held-out tests stay weak.** By the user's rule the harness is not changed; robust scores
  flag false passes, but do not alter the official score.
- **The leakage check is behavioural.** It probes typed inputs, so equivalence outside the probed
  types is not seen. Reviewed building blocks (`len`, `7 * x`, a three-way product, string
  length) are kept on purpose (`leakage.SHARED_OPERATIONS`).
- **Packing without a document mask (H1) and frozen lower layers (H2)** remain untested.
- **The 25/65 target:** the best run so far, `lr_half`, scored 25/65 (robust 24) on one seed.
  On clean, robust scoring the benchmark baseline is 19–21. None of the five arms is expected
  to add several points on its own.

## 7. Recommended next experiment

1. Run A–E. The A vs B comparison decides the skills question; C–E decide dedup and replay.
2. Then the next lever. The measured gap is to **unseen concepts** (loss 0.22 vs 0.035), not
   memorised rows. Every data-mix change so far left it unchanged. The next test should be **H2:
   train all 34 layers including embeddings vs the top 24**, with A–E's best data and three
   seeds. The words that separate neighbouring tasks ("fourth" vs "fifth") live in the frozen
   embeddings and lower layers. It needs a memory check first: about 1.5B trainable parameters
   on 15 GB.
