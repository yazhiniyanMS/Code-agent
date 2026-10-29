# YCode V12: breaking the generalization ceiling

Status: **in progress.** P is complete; L was killed by container restarts; S1's first seed passes the
second-seed gate, and its second seed is blocked on storage (section 5). Round 54 remains the validated
checkpoint of record.

## 1. Starting point (verified 2026-10-28)

- **Every checkpoint exists and matches its recorded SHA-256.** Re-hashed after a container
  restart, compared with `experiments/v11/artifacts/CHECKPOINTS.json`:

  | Checkpoint | File SHA-256 | Role |
  | --- | --- | --- |
  | Round 54 `model.pt` | `2538b55d5aee3718...` | validated best, combined 29 |
  | Round 54 `optim.pt` | `bf51203b1f882b80...` | needed to train from round 54 |
  | C2 pack | `931ae406e1ff3f34...` | combined 31; parent of R1 and of every V12 run |
  | R1 pack | `eb7ec59511019869...` | strongest experimental candidate, combined 33, not validated |
  | R6, R7, V11-A, V11-B, V11-C packs | as recorded | - |

- **C2 was restored** to the RAM disk. The unpack checks every tensor against the pack's content
  hash.
- **R1 was restore-checked in V11** in a clean process: tensors bit-identical, all scores equal.
- **Hardware:**
  - 4 Xeon cores (AVX-512, no bf16 units), 15.7 GB RAM, a 16 GB RAM disk.
  - 0.67 GB free disk, so no new 1.3 GB pack fits.
  - Training runs at about 2.2 min per step of 4,096 tokens. Evaluation takes about 95 min.
- **The model is the decoder-only YCode-LM v5** (1.52B parameters), not an encoder-decoder.

## 2. What was tested and why

The choice of runs is in `experiments/v12/PLAN.md`, written before any V12 result, with one
amendment written after P and before L. The gates are V6/V10's, unchanged.

- **The dominant failure:** in 38 of round 54's 68 clean failures, the model writes the trained
  function whose description shares the most words with the prompt (V7).
- **Hypothesis C had never been tested at a learning rate that learns anything.** V7's only test of
  contrast ran at 1.5e-5, where nothing is learned; V7 itself found 1e-4 necessary. It was therefore
  the largest gap in the evidence per CPU hour.
- **Hypotheses A and B are largely settled.** Replay prevents forgetting only of what it contains
  (V8, V11). Diversity helps inside a grammar but not across families (V8).

## 3. Run P: contrastive minimal pairs at lr 1e-4 (hypothesis C)

**Setup.** V10's pre-registered R3 arm, never run before (`experiments/v12/v12_p.json`):

- R1's recipe from C2: the same 350 grammar v2 programs with 2 examples each, data seed 1302, 50%
  uniform replay, 36 steps, lr 1e-4 warmup-stable-decay, top 24 layers, seed 1301.
- Every example is written next to a minimal-pair sibling (exactly one part different, a different
  result), and the answer tokens that differ from the sibling's code weigh 20.
- All 700 curriculum answers were executed on their programs' inputs before training and passed.

**Result** (same pipeline and item sets as every earlier run):

| | Round 54 | C2 | R1 (same seeds) | V11-A | **V12-P** |
| --- | --- | --- | --- | --- | --- |
| Dev clean /44 | 27 | 28 | 30 | 30 | **26** |
| Unseen-v1 clean /53 | 2 | 3 | 3 | 1 | **2** |
| **Combined /97** | 29 | 31 | 33 | 31 | **28** |
| Neighbour probe /38 | 4 | 2 | 2 | 3 | **2** |
| New constants, trained families /24 (all 40 inputs) | 0 | 1 | 4 | 2 | **0** |
| Training-program reproduction /40 | 0 | 26 | 36 | 38 | **38** |
| Held-out combinations /60 | 0 | 32 | 54 | 51 | **50** |
| Held-out families /66 | 2 | 1 | 2 | 3 | **3** |
| Benchmark official · clean · robust (never used to select) | 22 · 19 · 20 | 21 · 18 · 20 | 23 · 20 · 23 | 27 · 24 · 24 | **29 · 25 · 27** |
| Failures: wrong function retrieved (V7 classes) | 38 | 28 | 31 | 33 | **37** |

- **Contrast weighting increased the failure it targets.** It produced 6 more wrong-function
  retrievals than R1 with the same seeds and programs, and 2 more neighbour-probe answers that write
  the trained neighbour (9 against 8).
- **It sharpened reproduction of the trained programs (38/40) and cost everything that measures
  generalization:**
  - new constants fell from 4 to 0;
  - held-out combinations fell from 54 to 50;
  - dev fell from 30 to 26.
- **On dev, P passes nothing R1 fails** and fails four items R1 passes: the `index_of_min` bug fix,
  and `first_negative`, `first_three_chars` and `longest_word` as written functions.
- **Gates:** combined 28 < 32. P fails the second-seed gate and is not a candidate.
- **The benchmark is the highest of any model.** It is reported only, and with dev and unseen-v1
  both lower it is not evidence of a better model.
- **Cost:**
  - 103 min of training. My concurrent test runs slowed it; about 80 min unloaded.
  - 207 min in total, a peak of 11,061 MB of RAM, 147,456 trained tokens.
  - Checkpoint `12b0dcc3a716...`, pack content `8c1cb2d6cf22...`.
  - The pack was kept on the RAM disk only; it is reproducible from the spec.

**Conclusion.** Weighting the tokens where an answer differs from a sibling teaches the model to
reproduce trained programs more exactly, not to read the decisive word. Hypothesis C in this form is
refuted at an effective learning rate.

## 4. Run L: R1 + string-literal bug fixes

**Container restarts and determinism.** The container restarted at about 23:05 while L was training
(step 24 of 42).

- The restart wiped the RAM disk, which held L's working checkpoint and P's pack.
- L was relaunched unchanged. Its dataset was rebuilt byte-identical (181,270 tokens).
- Its per-step training losses and gradient norms equal the first attempt's to every logged digit
  (steps 1-7, for example step 6: loss 0.02959, gradient norm 0.061001).
- **So a V12 run is reproducible from its spec and seeds alone.** That is the fallback while the disk
  (0.59 GB free) cannot hold a 1.3 GB pack.
- P's pack is gone with the RAM disk. P failed its gates, its records and hashes are committed, and
  rebuilding it costs about 80 min.

**L was not completed.** A second restart, at about 23:49, killed the relaunch at step 12.

- A run of L's size needs about 3 h of uptime.
- The disk cannot hold a resumable checkpoint (about 6 GB) or a pack (1.3 GB).
- Amendment 2 therefore replaced it with S1, before S1 started.

## 4b. Run S1: a short low-learning-rate stage after R1 (hypothesis D, with L's data)

**Setup** (`experiments/v12/v12_s1.json`):

- from R1, lr 3e-5 warmup-stable-decay, 7 steps (one epoch of 33,990 tokens, measured by a dry run);
- the 93 string-literal bug fixes (PR #15) plus uniform replay of round 54's production mix
  (63% of tokens), with no new curriculum;
- seed 1301, data seed 1302;
- 16.6 min of training, 28,672 trained tokens, peak RAM 10,907 MB;
- checkpoint `cbc133372f4f...`, pack content `5075d097a657...`.

The run was screened on dev + unseen-v1 first. It passed the screen, so the remaining sets followed
on the same model (`experiments/v11/eval_only.py`), 84 min in total.

| | Round 54 | C2 | R1 (start) | V11-A | P | **S1** |
| --- | --- | --- | --- | --- | --- | --- |
| Dev clean /44 | 27 | 28 | 30 | 30 | 26 | **30** |
| Unseen-v1 clean /53 | 2 | 3 | 3 | 1 | 2 | **3** |
| **Combined /97** | 29 | 31 | 33 | 31 | 28 | **33** |
| Robust-clean dev + unseen-v1 | 26 + 1 = 27 | 28 + 2 = 30 | 30 + 2 = 32 | 30 + 1 = 31 | 26 + 1 = 27 | **27 + 1 = 28** |
| Clean losses against round 54 (category rule) | - | 5 (passes) | 4 (instruction following 3: fails) | 5 (fails) | 4 (passes) | **1 (passes)** |
| Neighbour probe /38 | 4 | 2 | 2 | 3 | 2 | **3** |
| New constants, trained families /24 | 0 | 1 | 4 | 2 | 0 | **4** |
| Training-program reproduction /40 | 0 | 26 | 36 | 38 | 38 | **37** |
| Held-out combinations /60 | 0 | 32 | 54 | 51 | 50 | **55** |
| Held-out families /66 | 2 | 1 | 2 | 3 | 3 | **4** |
| Benchmark official · clean · robust (reported only) | 22 · 19 · 20 | 21 · 18 · 20 | 23 · 20 · 23 | 27 · 24 · 24 | 29 · 25 · 27 | 22 · 19 · 22 |
| Failures: wrong function retrieved | 38 | 28 | 31 | 33 | 37 | **32** |

**What S1 changed against R1** (dev answers, the same items):

- **All three instruction-following items R1 loses are passed again:**
  - `bugfix:longest_word` is a genuine fix: `<` back to `>`.
  - `bugfix:remove_spaces` and `write:remove_spaces` pass with `''.join(c for c in s if
    c.isalpha())`. That is the body of the training concept `letters_only`, retrieved, not the
    requested `s.replace(' ', '')`. **These are weak passes:** they pass the official tests and fail
    the robust ones.
  - Round 54's own `write:remove_spaces` pass was the same shortcut. Its `bugfix:remove_spaces` was a
    genuine `s.replace(' ', '')`, so on that item S1 is weaker than round 54, though both count as
    official passes.
- **New against both:** `write:last_word`.
- **Lost against R1:** `first_negative` (written and as a bug fix), the `index_of_min` bug fix and
  `write:first_three_chars`. These were grammar gains of R1's.
- **R1's other learned skills are kept.** New constants stay at 4/24, reproduction is 37/40 and
  held-out combinations are 55/60.
- **Held-out families:** 4/66 (2 of the 7 the grammars can express), the highest of any model.
- **Robust-clean combined: 28.** That is 1 above round 54 and 4 below R1. The official gain over
  round 54 (+4) is mostly weak passes.
- **Failure mechanisms** (`results/mechanism_s1.json`): 32 wrong function retrieved (R1 31), 14
  correct concept with a wrong implementation (R1 18), 9 prompt misunderstood (R1 11), 5 edge cases
  (R1 4) and 4 syntax or format (R1 0).

**Audit (pre-registered: hygiene, duplicates, robustness):**

- The hygiene check removed 0 examples; no training example shares a name, AST skeleton or
  behaviour with any evaluation item.
- The literal bug fixes come only from `basics.training_concepts()`, which excludes the 15 dev
  concepts. `remove_spaces` and `longest_word` are among them, so neither was trained on.
- The restored `remove_spaces` items are retrievals of a training concept, not leakage of the
  evaluation item. They are weak gains and reported as such; by the rule, a weak gain does not void
  the candidate.

**Gates:**

- S1 seed 1301 **passes the second-seed gate**: combined 33 >= 32, and at most 1 clean loss in any
  category.
- Promotion needs the second seed, the same recipe applied to V11-A (`v12_s1_1302.json`, 8 steps).

**The second seed was killed by the kernel's OOM killer** during its final save (17 min in).

- The cause: S1's 1.3 GB pack sat on the RAM disk next to the trainer (10.8 GB) and the 2.1 GB save.
- PR #17 adds a pre-flight memory check that refuses such a run before it trains.
- The second seed needs that pack off the RAM disk. It cannot be moved to the disk (0.59 GB free)
  and was not deleted, because deleting a checkpoint needs your approval (section 5).

## 5. Comparison, decision and next experiment

Full tables: `experiments/v12/results/compare.md`. Failure mechanisms: `results/mechanism.json`
and `results/mechanism_s1.json`.

**Decision (provisional until S1's second seed runs):**

- **Round 54 stays the validated checkpoint of record.** No V12 candidate has completed the
  promotion rule, which needs every seed.
- **Strongest V12 candidate: S1 (seed 1301).**
  - It is the first post-round-54 model to pass the category rule while keeping combined 33.
  - It keeps R1's learned skills and has the best held-out-family score of any model.
  - But its official gain over round 54 rests largely on weak passes: robust combined 28 against
    27.
- **R1 remains the best on robust scores** (32), with its 3 instruction-following losses.
- **Hypothesis C** (contrast weighting, run P) is refuted at lr 1e-4. It increased the failure it
  targets.

**Next step, already specified and blocked on storage:** S1's second seed, from V11-A, seed 1302 /
data 1303 (`v12_s1_1302.json`). It needs one of:

- **(a) Disk space.** Approval to remove old V6/V7 lossless packs (`lmwork/v6/runs/{D,G}/1302`,
  `v6/soups/DG130{1,2}c`, `v7/runs/L2/1301`; about 5.7 GB). Then S1's pack and future packs can be
  kept on disk.
- **(b) Approval to drop RAM-disk packs of runs that rebuild exactly.** S1 rebuilds from R1 and its
  spec in about 16 min, and determinism was shown bit for bit: L's restarted run matched its first
  attempt on every logged step. Its pack would then be dropped before the second seed trains.

**Next experiment justified by the evidence, after S1's seeds:** make the restored skill genuine.

- S1 restored `remove_spaces` by retrieving `letters_only`. The literal bug fixes taught which
  literal a `join` needs, not the `replace` pattern.
- No training concept calls `replace` with literal arguments. The only one using `replace` is
  `replace_word`, with variables, and `remove_spaces` is a dev concept held out of training.
- The next data change is therefore new verified training concepts with literal `replace(...)`,
  `split(...)` and `strip(...)` calls, and literal bug fixes on them.
  - None may share a name, skeleton or behaviour with an evaluation item; the hygiene check
    enforces this.
  - Success is measured on the robust tests, not only the official ones.
