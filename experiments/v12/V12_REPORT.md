# YCode V12: breaking the generalization ceiling

Status: **S1 is promoted under the pre-registered rules.**

- Its two seeds score combined 33 and 31 (mean 32.0, exactly the threshold), with at most 1.0 mean clean
  loss in any category, and its audit finds no leakage.
- **S1 seed 1301 replaces round 54 as the checkpoint of record** (section 5).
- The gain is modest: on the robust tests, the two seeds average 29 against round 54's 27.
- P refuted hypothesis C at lr 1e-4. L was lost to container restarts.
- unseen-v2 was not evaluated. It stays reserved for the final protocol you authorize.

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

**Gates:** S1 seed 1301 **passes the second-seed gate**: combined 33 >= 32, and at most 1 clean
loss in any category.

## 4c. S1's second seed, from V11-A

**Setup** (`experiments/v12/v12_s1_1302.json`): S1's stage applied to V11-A, which is R1's recipe at
seed 1302 / data seed 1303.

- 8 steps (one epoch of 33,758 training tokens), seed 1302, data seed 1303.
- 32,768 trained tokens; 11.8 min of training steps, 13.6 min with loading; peak RAM 10,768 MB.
- Checkpoint `3fdbf898bc3e...`, pack content `9a198bc80668...`.
- Code: commit 527444b, which is main plus PR #20. PR #20 changes only the final save's memory use.

**Getting it to run took three attempts:**

1. **Killed by the kernel's OOM killer in its final save.** The kernel's report shows a 14.34 GB limit
   for this session's processes:
   - 11.43 GB in processes;
   - 2.86 GB on the RAM disk (S1 1301's 1.32 GB pack and 1.54 GB of the save being written).
2. **Two fixes, so that nothing had to be deleted:**
   - **PR #20** frees the optimizer's 2.1 GB momentum before the final weights-only save. The saved
     weights are unchanged.
   - **PR #19** packs a checkpoint as a lossless delta from its parent. S1 1301 went to the disk as a
     0.42 GB pack against R1, verified bit for bit.
3. **A container restart killed the relaunch** at about step 3 and wiped the RAM disk, including S1
   1301's RAM-disk pack.
   - S1 1301 survived only as the delta pack on disk. It was restored from it and R1's pack, content
     `5075d097...` verified.
   - The third launch completed.

| | Round 54 | R1 | V11-A | **S1 1301** | **S1 1302** | **S1 mean** |
| --- | --- | --- | --- | --- | --- | --- |
| Dev clean /44 | 27 | 30 | 30 | 30 | 30 | 30 |
| Unseen-v1 clean /53 | 2 | 3 | 1 | 3 | 1 | 2 |
| **Combined /97** | 29 | 33 | 31 | **33** | **31** | **32.0** |
| Robust-clean dev + unseen-v1 | 27 | 32 | 31 | 28 | 30 | 29.0 |
| Clean losses against round 54, by category | - | 4: instruction following 3 | 5: instruction following 3 | 1: hallucination 1 | 3: instruction following, reasoning, hallucination 1 each | max 1.0 |
| Gains against round 54 that are weak (robust tests fail) | - | - | - | 3 of 5 | 0 of 5 | - |

**What S1's stage changed on V11-A** (dev answers):

- **Restored:**
  - `bugfix:longest_word`, a genuine fix (`<` back to `>`);
  - `write:remove_spaces`, with the same weak `isalpha` shortcut as round 54's own answer.
- **Still lost:** `bugfix:remove_spaces`. The answer returns the buggy `s.replace(' ', '_')` unchanged.
- **Lost against V11-A:** `bugfix:index_of_min` and `write:longest_word`, two of R1's grammar gains.
- **All five of its gains against round 54 are robust:** `first_negative` (written and as a bug fix),
  `product`, `first_three_chars` and unseen `sum_at_odd_positions`.

**The remaining sets** were evaluated after the promotion rule had been applied. They are reported
only, never used to choose (`experiments/v11/eval_only.py`, 31 min):

| | S1 1301 | S1 1302 | V11-A (its start) |
| --- | --- | --- | --- |
| New constants, trained families /24 | 4 | 3 | 2 |
| Neighbour probe /38 | 3 | 3 | 3 |
| Held-out families /66 | 4 | 2 | 3 |
| Training-program reproduction /40 | 37 | 38 | 38 |
| Held-out combinations /60 | 55 | 50 | 51 |
| Benchmark official · clean · robust | 22 · 19 · 22 | 23 · 20 · 21 | 27 · 24 · 24 |
| Failures: wrong function retrieved | 32 | 31 | 33 |

Failure mechanisms, from `results/mechanism_s1_1302.json`:

- wrong function retrieved: 31;
- correct concept with a wrong implementation: 23;
- prompt misunderstood: 5;
- edge cases: 6;
- syntax or format: 1.

On seed 1302 the stage cost 4 benchmark points against its start (27 → 23). That is reported only;
the benchmark varied by 3-4 items between seeds in V11.

## 4d. Promotion and audit

**The promotion rule** (V6, unchanged since V11 and fixed in `PLAN.md` before any V12 result):

| Condition | S1 1301 | S1 1302 | Result |
| --- | --- | --- | --- |
| Every seed beats 29 | 33 | 31 | passes |
| Mean combined >= 32 | | | 32.0, passes (at the threshold) |
| Mean clean losses < 3 in every category | hallucination 1 | 1 each in 3 categories | max 1.0 (hallucination), passes |

The R1 recipe alone, R1 and V11-A, also scored 33 and 31. It failed only on 3.0 mean instruction-following
losses. S1's stage brings that category to 0.5.

**The audit of every gained item** (V11's, before promotion; `experiments/v12/audit_s1.py` →
`results/audit_s1.json`):

- **The S1 data is the data trained on.** Each seed's dataset was rebuilt from its recorded seed and
  options, and matched the run's manifest byte for byte.
- **Hygiene:** the runs' check removed 0 examples in both seeds. It compares each example's name, AST
  skeleton and behaviour with every dev, unseen and benchmark item.
- **Duplicates:**
  - No gained item's name occurs in either seed's training tokens as a function.
  - The only match is the English phrase "product of", in other functions' descriptions (`times`,
    `calc`, `scale_by_five`, `dot_product`) from round 54's own production replay. That is not the dev
    item `product`.
- **Robustness:**
  - Seed 1301: 3 of its 5 gains are weak (`write:last_word`, unseen `count_capitalized` and
    `sum_at_odd_positions`). Its other weak passes are the two `remove_spaces` items.
  - Seed 1302: all 5 of its gains are robust. Its only weak pass is `write:remove_spaces`, which round
    54 passes the same way.
- **Starting points:** R1 and V11-A were audited in V10 and V11 (hygiene removed 0).
- **Conclusion:** no leakage. Seed 1301's weak gains are reported as weak; by the rule, they do not
  void the candidate.

## 5. Comparison, decision and next experiment

Full tables: `experiments/v12/results/compare.md`. Failure mechanisms: `results/mechanism.json` and
`results/mechanism_s1.json`. Audit: `results/audit_s1.json`.

### Decision

- **S1 is promoted.** It passes the promotion rule on both seeds and its audit (section 4d).
- **The new checkpoint of record is S1 seed 1301.**
  - Between the two seeds it is chosen by the established selection metric, combined clean dev +
    unseen-v1: 33 against 31.
  - Robust scores are "always reported, never used to select".
- **Caveats:**
  - **The promotion is at the threshold:** mean 32.0 against the required 32.
  - **The robust gain is small:** a mean of 29 robust-clean passes against round 54's 27.
  - **Seed 1301's official gain is mostly weak passes.** Its robust-clean score is 28, against round
    54's 27. Seed 1302's is 30, with every gain robust.
  - **unseen-v2 has not been evaluated.** It is reserved for the final protocol you authorize, which
    is the confirmation this result now needs.
- **Hypothesis C** (contrast weighting, run P) is refuted at lr 1e-4. It increased the failure it
  targets.
- **Hypothesis D** (a short low-learning-rate second stage) is supported in this form. It removed
  the instruction-following losses that blocked R1's recipe, at a cost of 2 of R1's grammar gains per
  seed.

### Comparison against round 54, C2 and the strongest V11 candidate

The strongest V11 candidate is the R1 recipe: R1 at seed 1301 and V11-A at seed 1302. It was not
promoted.

| | Round 54 | C2 | R1 1301 | V11-A 1302 | **S1 1301 (new record)** | S1 1302 |
| --- | --- | --- | --- | --- | --- | --- |
| Combined clean /97 | 29 | 31 | 33 | 31 | **33** | 31 |
| Dev · unseen-v1 clean | 27 · 2 | 28 · 3 | 30 · 3 | 30 · 1 | **30 · 3** | 30 · 1 |
| Robust-clean combined | 27 | 30 | 32 | 31 | **28** | 30 |
| Category rule (most clean losses in one category) | - | 2 | 3: fails | 3: fails | **1** | 1 |
| New constants /24 · neighbours /38 · held-out families /66 | 0 · 4 · 2 | 1 · 2 · 1 | 4 · 2 · 2 | 2 · 3 · 3 | **4 · 3 · 4** | 3 · 3 · 2 |
| Benchmark official · clean · robust (reported only) | 22 · 19 · 20 | 21 · 18 · 20 | 23 · 20 · 23 | 27 · 24 · 24 | **22 · 19 · 22** | 23 · 20 · 21 |

### Configurations, hashes and cost

| Run | Spec | Start | Steps · tokens | Training | Peak RAM | Checkpoint SHA-256 | Pack content SHA-256 | Stored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 1301 | `v12_s1.json` | R1 (pack `eb7ec595...`) | 7 · 28,672 | 16.6 min | 10,907 MB | `cbc133372f4f785f08192ab9b13400e15d6a1dabf3c37f6f8dceac8ad051a20d` | `5075d097a6571cc964907a011b6d4f634100bfb393b61bf5a18b897abb7837f7` | `lmwork/v12/ckpt/S1_1301.ypk`, 0.42 GB, delta against R1 (file `4b9f4898c35a...`) |
| S1 1302 | `v12_s1_1302.json` | V11-A (pack content `944ded91...`) | 8 · 32,768 | 13.6 min | 10,768 MB | `3fdbf898bc3ea2b2b3748a767e43eb7a3c95843f767155557afa439a343abc73` | `9a198bc80668d9ae9c29eabc117f3419129d9cc40bcf46d6efb010f4a0267360` | RAM disk only (1.32 GB); rebuilds in about 14 min |
| P 1301 | `v12_p.json` | C2 | 36 · 147,456 | 103 min | 11,061 MB | `12b0dcc3a716...` | `8c1cb2d6cf22...` | lost with the RAM disk; failed its gates |

- **Shared settings:** lr 3e-5 warmup-stable-decay, 1 warmup step, batch 4 × 1,024, top 24 of 34 layers,
  fp32, Lion, compact bf16 checkpoints.
- **Data:** the 93 string-literal bug fixes plus a 50% share of uniform replay from round 54's
  production mix (`literal_bugfix: 3`, `legacy_share: 0.5`, no curriculum).
- **What the record depends on:** the S1 1301 checkpoint is its delta pack plus R1's pack
  (`lmwork/v10/ckpt/R1_1301.ypk`, 1.32 GB) plus round 54. All three are on disk.
- **Unpacking:** `python -m ycode.lm.packckpt unpack lmwork/v12/ckpt/S1_1301.ypk OUT_DIR` (needs
  PR #19) checks both packs' content hashes.
- **Full chain from round 54:** C2 (56 steps, 166 min), then R1 (36 steps, 74 min), then S1 (7 steps,
  17 min).
- **Lost to failures:** 2.4 h of training.
  - L's two attempts, 88 and 33 min, were killed by restarts.
  - S1 1302's first two attempts, 17 min (OOM) and 6 min (restart), were also lost.
- **Evaluation:** S1 1301's took 67 min for all sets, and P's took 104 min.

### Next experiment

**Make the restored skills genuine.** Seed 1301 restored `remove_spaces` by retrieving `letters_only`.
Seed 1302 did not restore the bug fix at all.

- No training concept calls `replace` with literal arguments. The only one using `replace` is
  `replace_word`, with variables, and `remove_spaces` is a dev concept held out of training.
- **The change:** new verified training concepts with literal `replace(...)`, `split(...)` and
  `strip(...)` calls, and literal bug fixes on them.
  - None may share a name, skeleton or behaviour with an evaluation item; the hygiene check
    enforces this.
  - The same S1 stage, on the same two starting points.
  - Success is measured on the robust tests as well as the official ones.
- **Before that:** the final protocol on unseen-v2 for S1 1301, if you authorize it.
