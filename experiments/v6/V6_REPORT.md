# YCode-LM V5 → V6: training improvements

Status: **in progress** (iteration 5 running). Measured results are marked; nothing here comes from a run that did not happen.

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
| Curriculum v2: 19 multi-step algorithm families; plausible-but-wrong repairs; worked input-output examples (opt-in) | `idioms.py` options `idioms_contrast`, `idioms_io` | The failure loop's top cause (68%: writing another known function) |
| Curriculum v3: 120 families written by a separate agent that read only `idioms.py` | `ycode/lm/idioms_extra.py` | Breadth |
| Failure loop: five root causes (misunderstood prompt, incorrect logic, edge cases, syntax, incomplete implementation) on the development tasks only | `failures.root_cause`, `experiments/v6/failure_loop.py` | Choose the next experiment without touching the holdout |
| Resumable runs that survive container restarts: checkpoint every `save_every` steps, optimizer state freed after training, disk check scaled to the starting checkpoint, curriculum family set pinned per spec | `experiment.py`, `rounddata.py` | Two container restarts and a worker restart during this mission |

## Runs

Protocol, fixed before the runs: **development tasks** (dev, unseen-v1, the probe's new curriculum
family members) choose every next experiment. **unseen-v2 is the fixed holdout**: never trained on,
never read, never used to choose (three of its names were exposed once, in held-out-check
messages that dropped curriculum programs equivalent to them). A candidate replaces round 54 only
if every seed beats its all_clean (29/168), the mean gain is >= 3 items, and no failure category
loses >= 3 clean items round 54 passes. The benchmark is reported only.

Every candidate fine-tunes round 54: Lion, lr 1.5e-5 (G: 3e-5), top 24 of 34 layers, batch
4 x 1024 packed tokens, fp32 on this host, seed 1301. Pass counts are official tests, clean items.

| Run | Change (one variable vs the previous best) | Steps / tokens | all_clean /168 | Dev /44 | Unseen-v1 /53 | Unseen-v2 /71 (holdout) | Bench /65 (clean /61) | Train min | Peak RAM train / eval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Round 54 | - | - | **29** | 27 | 2 | 0 | 22 (19) | - | - |
| A (v5.1 control) | v5 data mix, 24 steps | 24 / 98,304 | (26 on dev+v1) | 25 | 1 | - | 25 (22) | 53 | 11.3 / 7.7 GB |
| C | curriculum v1 replaces most v5 data | 24 / 98,304 | 24 | 23 | 1 | 0 | 18 (16) | 64 | 11.3 / 7.7 GB |
| D | 72-step budget, curriculum x2, half v5 data | 72 / 294,912 | **30** | **28** | 2 | 0 | 24 (22) | 181* | 11.3 / 7.7 GB |
| E | + algorithm families, wrong-code repairs, worked examples | 72 / 294,912 | 25 | 24 | 1 | 0 | 26 (23) | 147 | 10.8 / 7.7 GB |
| F | breadth: 215 families at D's curriculum budget | 72 / 294,912 | 28 | 26 | 2 | 0 | 24 (21) | 140 | 11.3 / 7.8 GB |
| G | D with learning rate x2 (3e-5) | 72 / 294,912 | **30** | 27 | **3** | 0 | 22 (20) | 158 | 11.3 / 7.7 GB |

\* D shared the CPU with analysis jobs for ~1 h. Checkpoints: 7.3 GB during training (weights +
optimizer state), 3.1 GB kept per candidate (bf16, the precision evaluation uses).

**What the probe shows** (code-only loss on reference solutions; lower is better; unseen-v2
shown for the record, not used to choose):

| Set | Round 54 | C | D | E | F |
| --- | --- | --- | --- | --- | --- |
| Curriculum v1 programs (trained) | 1.060 | 0.604 | 0.366 | 0.428 | 0.544 |
| New members of those families (never trained) | 1.320 | 0.714 | 0.451 | 0.540 | 0.645 |
| Clean dev references | 0.275 | 0.191 | 0.182 | 0.181 | 0.181 |
| Unseen-v1 references | 1.164 | 0.739 | 0.676 | 0.672 | 0.677 |
| Unseen-v2 references (holdout) | 1.279 | 0.986 | 0.962 | 0.931 | - |

- The curriculum is learned as patterns, not instances: new family members improve as much as
  trained ones.
- It transfers in likelihood to unseen tasks (-42% on unseen-v1, -25% on the holdout) but the
  transfer saturates at ~0.67 whatever the depth (D), formats (E) or breadth (F), and at that
  level greedy decoding still writes wrong code: new-concept pass rates stay at 1-2/53 and 0/71.
- Where pass rates move, they move on near-trained tasks: dev +1 (D), and the benchmark +2 to +4
  in the 72-step runs (reported only).
- Failure loop on the development tasks: 68% of round 54's failures write another known function
  (misunderstood prompt), 16% incorrect logic; C1301 the same. Wrong-code repairs aimed at that
  cost E four dev bug fixes (whole-body rewrites taught rewriting instead of minimal edits).
- Answers become shorter and better formed: on unseen-v2 (reported, not used to choose), calls to
  unknown names fall from 13 (round 54) to 4-7 and unparseable answers from 15 to 3-4 in every
  72-step candidate (D to F); D's answers average 100 characters against round 54's 311.

## Decision

DECISION_PENDING

## Reproduction

Every run is a spec in `experiments/v6/` run by the experiment runner from the main checkout;
its `manifest.json` records the seed, the options and the code commit, and rebuilding D's data at
its recorded commit (903c2a3) reproduces it byte for byte.

```bash
# round 54 on unseen-v2 (the other sets: experiments/v51)
python -m ycode.lm.cli eval --model /home/user/lmwork/loop/anchor --problems unseen2 --device cpu --answers-out unseen2.jsonl
# a candidate: data, training, the four evaluations and the diagnostic (resumes after interruption)
git checkout <code_version from the run's manifest.json>
python -m ycode.lm.experiment experiments/v6/v6_d_idioms_full.json
# the learning probe and the failure loop
python experiments/v6/probe_learning.py /home/user/lmwork/v6/runs/D/1301/model_bf16 D1301.json
python experiments/v6/failure_loop.py D1301 runs/D/1301/dev.jsonl runs/D/1301/unseen.jsonl
```

Per-run records (scores, answers, manifests, hygiene, diagnostics, step metrics) are in
`experiments/v6/results/runs/`; probe outputs in `experiments/v6/results/`.
