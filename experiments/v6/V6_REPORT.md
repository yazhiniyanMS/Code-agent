# YCode-LM V5 → V6: training improvements

Status: **complete**. Every number below is measured; nothing here comes from a run that did not happen.

**Outcome: no candidate beats round 54 under the pre-registered rule, so round 54 stays the best checkpoint
(unchanged, sha256 2538b55d...).** The best candidate, D (idiom curriculum, 72 steps), gains +1 clean item
on both seeds (30/168 and 30/168 against 29/168); the rule asks for +3. The D+G weight average scored 32
on seed 1301 but 28 on seed 1302. The remaining bottleneck is turning a description of a new task into a
new program: see [Decision](#decision).

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
| Base-referenced checkpoints: training checkpoints and kept models store only the tensors that differ from round 54 (6.4 GB instead of 7.3; 2.1 GB instead of 3.1), byte-identical weights | `train.py` (`compact_checkpoints`, `export_checkpoint(base=...)`), runner `train.compact`, `keep_models: "compact"` | Disk: a second seed of the soup needs two kept models next to a training run |
| Weight averaging of fine-tuned checkpoints (model soup) | `experiments/v6/soup.py` | D and G learn the same data at different rates; their average is more robust than either |
| Resumable runs that survive container restarts: checkpoint every `save_every` steps, optimizer state freed after training, disk check scaled to the starting checkpoint, curriculum family set pinned per spec | `experiment.py`, `rounddata.py` | Two container restarts and a worker restart during this mission |

## Runs

Protocol, fixed before the runs: **development tasks** (dev, unseen-v1, the probe's new curriculum
family members) choose every next experiment. **unseen-v2 is the fixed holdout**: never trained on,
never read, never used to choose (three of its names were exposed once, in held-out-check
messages that dropped curriculum programs equivalent to them). A candidate replaces round 54 only
if every seed beats its all_clean (29/168), the mean gain is >= 3 items, and no failure category
loses >= 3 clean items round 54 passes. The benchmark is reported only.

Every candidate fine-tunes round 54: Lion, lr 1.5e-5 (G: 3e-5), top 24 of 34 layers, batch
4 x 1024 packed tokens, fp32 on this host, seed 1301 (the last three rows: seed 1302, same specs).
Pass counts are official tests, clean items.

| Run | Change (one variable vs the previous best) | Steps / tokens | all_clean /168 | Dev /44 | Unseen-v1 /53 | Unseen-v2 /71 (holdout) | Bench /65 (clean /61) | Train min | Peak RAM train / eval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Round 54 | - | - | **29** | 27 | 2 | 0 | 22 (19) | - | - |
| A (v5.1 control) | v5 data mix, 24 steps | 24 / 98,304 | (26 on dev+v1) | 25 | 1 | - | 25 (22) | 53 | 11.3 / 7.7 GB |
| C | curriculum v1 replaces most v5 data | 24 / 98,304 | 24 | 23 | 1 | 0 | 18 (16) | 64 | 11.3 / 7.7 GB |
| D | 72-step budget, curriculum x2, half v5 data | 72 / 294,912 | **30** | **28** | 2 | 0 | 24 (22) | 181* | 11.3 / 7.7 GB |
| E | + algorithm families, wrong-code repairs, worked examples | 72 / 294,912 | 25 | 24 | 1 | 0 | 26 (23) | 147 | 10.8 / 7.7 GB |
| F | breadth: 215 families at D's curriculum budget | 72 / 294,912 | 28 | 26 | 2 | 0 | 24 (21) | 140 | 11.3 / 7.8 GB |
| G | D with learning rate x2 (3e-5) | 72 / 294,912 | 30 | 27 | **3** | 0 | 22 (20) | 158 | 11.3 / 7.7 GB |
| **D+G soup** | equal-weight average of D and G (no training) | - | **32** | **29** | **3** | 0 | 25 (23) | - | - / not measured |
| D, seed 1302 | replication | 72 / 294,912 | 30 | 28 | 2 | 0 | 26 (23) | 190 | 10.8 / 6.7 GB |
| G, seed 1302 | replication | 72 / 294,912 | 28 | 26 | 2 | 0 | 27 (25) | 167 | 11.3 / 6.7 GB |
| D+G soup, seed 1302 | replication | - | 28 | 26 | 2 | 0 | 28 (25) | - | - / not measured |

\* D shared the CPU with analysis jobs for ~1 h. Throughput 26.5-33.3 tokens/s. Checkpoints: 7.3 GB
during training (weights + optimizer state), 3.1 GB kept per candidate (bf16, the precision evaluation
uses); the seed-1302 runs used base-referenced checkpoints (6.4 GB and 2.1 GB, identical weights).
Evaluation peak RAM fell from 7.7 to 6.7 GB with them, because the frozen tensors are read from round 54
by memory map.

**What the probe shows** (code-only loss on reference solutions; lower is better; unseen-v2
shown for the record, not used to choose):

| Set | Round 54 | C | D | E | F | G | D+G soup | D+G soup, seed 1302 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Curriculum v1 programs (trained) | 1.060 | 0.604 | 0.366 | 0.428 | 0.544 | 0.217 | 0.274 | 0.276 |
| New members of those families (never trained) | 1.320 | 0.714 | 0.451 | 0.540 | 0.645 | 0.311 | 0.362 | 0.382 |
| Clean dev references | 0.275 | 0.191 | 0.182 | 0.181 | 0.181 | 0.180 | 0.177 | 0.178 |
| Unseen-v1 references | 1.164 | 0.739 | 0.676 | 0.672 | 0.677 | 0.671 | 0.665 | 0.661 |
| Unseen-v2 references (holdout) | 1.279 | 0.986 | 0.962 | 0.931 | 0.896 | 0.997 | 0.974 | 0.983 |

- The curriculum is learned as patterns, not instances: new family members improve as much as
  trained ones.
- It transfers in likelihood to unseen tasks (-42% on unseen-v1, -25% on the holdout) but the
  transfer saturates at ~0.67 whatever the depth (D), formats (E) or breadth (F), and at that
  level greedy decoding still writes wrong code: new-concept pass rates stay at 1-3/53 and 0/71.
- Where pass rates move, they move on near-trained tasks: dev +1 (D), and the benchmark +2 to +4
  in the 72-step runs (reported only).
- Failure loop on the development tasks: 68% of round 54's failures write another known function
  (misunderstood prompt), 16% incorrect logic; C1301 the same. Wrong-code repairs aimed at that
  cost E four dev bug fixes (whole-body rewrites taught rewriting instead of minimal edits).
- Answers become shorter and better formed: on unseen-v2 (reported, not used to choose), calls to
  unknown names fall from 13 (round 54) to 4-7 and unparseable answers from 15 to 3-4 in every
  72-step candidate (D to F); D's answers average 100 characters against round 54's 311.

## Decision

**Round 54 stays the best checkpoint.** No candidate meets the rule fixed before the runs. The rule
asks that every seed beat 29/168 all_clean, that the mean gain be at least 3 items, and that no failure
category lose a mean of 3 or more clean items that round 54 passes. Computed by
`experiment.compare(..., "all_clean", "items_lost")` against round 54's answers; the record is
`results/decision_compare.json`.

| Candidate | all_clean, seed 1301 / 1302 | Gain per seed | Mean gain (need >= 3) | Worst category, mean items lost (limit 3) | Meets rule |
| --- | --- | --- | --- | --- | --- |
| D: curriculum, 72 steps | 30 / 30 | +1 / +1 | +1.0 | instruction following 2.5 (3 on seed 1302) | no: gain too small |
| G: D at lr 3e-5 | 30 / 28 | +1 / -1 | 0.0 | reasoning 3.5 | no |
| D+G soup | 32 / 28 | +3 / -1 | +1.0 | reasoning 2.0 | no: seed 1302 below round 54 |

- `compare`'s own verdict for D is "retained", but only under the v5.1 statistical test, which with two
  identical differences (sd 0) passes any positive mean. The V6 rule's 3-item margin guards against
  seed-to-seed churn, which these runs show reaches 4 items (the soup: 32, then 28). A consistent +1 is real
  but too small to replace the checkpoint. D is kept as a candidate (below).
- **What persists across seeds.** Every candidate gains the same four dev items on both seeds:
  `first_three_chars`, `bugfix:first_negative`, `bugfix:product` and `double_plus_one` (D also
  the write version of `first_negative`). Each is a near neighbour of a curriculum family. All lose `count_greater` and
  `count_words_longer_than` on both seeds, and D also loses `any_zero` and `bugfix:remove_spaces`.
  The curriculum moves items between nearby concepts; it adds no new ones.
- **Nothing transfers to new concepts in pass rate.** Unseen-v1 is 2/53 for round 54 and for every
  candidate on seed 1302; the one +1 (`first_index_above`, seed 1301) did not replicate. The holdout
  unseen-v2 is 0/71 for every model. The benchmark (reported only) is 22-28/65 for the 72-step candidates
  against round 54's 22; it played no part in the decision.
- **Rejected for the reasons the mission names:** C and F lowered the scores; E improved the
  benchmark but lost 4 dev bug fixes; G lowered training loss most (0.23 against D's 0.29) without
  better held-out pass rates; the soup's seed-1301 gain was churn.

### The remaining bottleneck

The model cannot yet turn a description of an unfamiliar task into a new program. The evidence:

1. **The model learns what it is shown, as patterns.** Code loss on new members of curriculum
   families (never trained) falls from 1.32 to 0.31-0.45, as much as on the trained programs.
2. **Transfer to unseen concepts is real in likelihood but saturates.** Unseen-v1 code loss falls from
   1.16 to 0.67 by D and then stays at 0.66-0.68 whatever changes: depth (D), formats (E), breadth
   (215 families, F), learning rate (G), averaging or seed. The holdout follows (1.28 to 0.90-1.00).
3. **At 0.67 nats per code token, greedy decoding still writes the wrong function.** 68% of round 54's
   failures on the development tasks write another, familiar function (misunderstood prompt). The
   candidates' answers are shorter and better formed (unparseable holdout answers 15 down to 3-4 in D to F),
   but no more correct.

Within this fine-tuning budget (at most 295k tokens, the top 24 of 34 layers, fp32 at ~28 tokens/s
on 4 CPU cores), data content, mixture, curriculum breadth and learning rate do not move that floor.
The levers these runs leave untested are trained capacity and compute. Estimated costs:

- **Training all 34 layers.** From the measured 11.3 GB for 24 layers, this needs about 13.5 GB:
  6.1 GB of weights, 6.0 GB of Lion momentum and the activations. That leaves under 1.5 GB of the
  15 GB host, a real swap risk. Each step would also take about 40% longer, because the backward pass
  would run through 10 more layers. This is not estimated to be safe here.
- **Four times the tokens.** 288 steps would take about 12 h of training plus 1 h of evaluation per
  seed, and two seeds are needed. It would test whether the unseen-v1 floor is a budget limit. It is
  the more informative experiment, and it fits a longer allocation.

### Kept checkpoints

Round 54 (`/home/user/lmwork/loop/anchor/model.pt`, sha256 `2538b55d5aee3718...`) is unchanged; every
run recorded its hash, and the file still has it. The candidates are kept for reproduction and further averaging.
All four are base-referenced bf16 files of 2.1 GB, which store only the tensors that differ from round
54 and need it at that path to load:

| Model | Path | sha256 |
| --- | --- | --- |
| D, seed 1302 | `/home/user/lmwork/v6/runs/D/1302/model_bf16/model.pt` | `81425c6a288205eb76882b5bcc108e3f959d8c0ae453367193fb0c37b40ab75d` |
| G, seed 1302 | `/home/user/lmwork/v6/runs/G/1302/model_bf16/model.pt` | `489f6f65ac38f9d751190ab398458f1b1283d680b2461d7ed2f8d0124cfd071b` |
| D+G soup, seed 1301 | `/home/user/lmwork/v6/soups/DG1301c/model.pt` | `d84d20977148fbd148eb74440f9244b8018d7c68039c3e8a38826247ba6bfa68` |
| D+G soup, seed 1302 | `/home/user/lmwork/v6/soups/DG1302c/model.pt` | `357872cb796770e328fae4bc411cc2b1ea305fda15e7f10aab8c70c93025ecac` |

The seed-1301 D and G weights were deleted to make room for the replication: the session's disk
allowance had 3.7 GB left. Their soup is kept in base-referenced form, and both can be rebuilt from their specs and
manifests.

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
`experiments/v6/results/runs/`; the soups' answers and build records in `experiments/v6/results/soups/`;
probe outputs and the decision comparison in `experiments/v6/results/`. The seed-1302 replication ran
the same specs with `seeds: [1302]`: D, then G, then `soup.py` over the two kept models.
