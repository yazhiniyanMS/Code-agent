# YCode V11: replicating R1, finding the forgetting, and publishing

Status: complete. **No V11 candidate passes the pre-registered gates. Round 54 remains the validated checkpoint of record.** The checkpoint upload to GitHub is blocked (section 7).

## Executive summary

- **V11-A replicated R1's recipe at seed 1302 (data seed 1303).**
  - Dev replicated exactly: the same 30 clean items, against round 54's 27.
  - Unseen-v1 did not: 1 against R1's 3.
  - Combined 31; over two seeds the recipe averages 32.0.
  - Both seeds lose the same 3 instruction-following items round 54 passes, so the recipe fails
    the promotion rule's category part (mean 3.0, needs < 3). R1's forgetting pattern is reproduced.
- **Why the items are forgotten** (section 3). Two of the three are bug fixes where every model since
  C2 hands the buggy code back unchanged.
  - The targets are correct, the decisive tokens weigh 8, and frequency is not the cause.
  - The model's preference at the one decisive token flipped. Round 54 makes the fix; C2 gives the
    fixing token `>` only 0.20 against the buggy `<`, and R1 0.40.
  - Round 54's own training mix has **no** bug fix that turns a string literal into `''`, which is
    the `remove_spaces` fix. Replay cannot restore a skill it does not contain.
  - The third item, and unseen `count_words_longer_than`, were weak passes of round 54: its own code
    fails the robust tests.
- **Three further pre-registered experiments** (sections 4-5), none passing:

  | Run | Change | Combined | Result |
  | --- | --- | --- | --- |
  | V11-B | replay drawn by C2's forgetting score | 31 | restores the `longest_word` bug fix, loses 2 reasoning items |
  | V11-W | 0.5·R1 + 0.5·round 54 | 27 | restores `longest_word`, loses R1's gains |
  | V11-C | one stage from round 54, retention from the start | 25 | forgets more |

- **Named artifacts:**
  - **Validated best: round 54** (unchanged, hashes rechecked).
  - **Best experimental candidate: R1** (combined 33, unvalidated). It was restored from its
    lossless pack in a clean process:
    - every tensor is bit-identical;
    - dev 62/62 and unseen-v1 52/53 answers are byte-identical, with all scores equal;
    - the one differing answer fails either way. It is a floating-point near-tie on this new host.
- **Preservation.**
  - Every new run has a verified lossless pack, configuration, seeds, logs and records.
  - Nothing of round 54, C2, R1, R6 or R7 was modified, moved or deleted.
- **Publishing.**
  - Pushed to GitHub: the report, plan, configurations, manifests with SHA-256 hashes, and all
    evaluation records.
  - **The checkpoint file could not be uploaded.** This session type may not create releases, and
    the network policy refuses `lfs.github.com`. The fix and a ready script are in section 7.
- **Next:** V11-D, string-literal bug-fix examples (section 8). It needs about 1.4 GB of disk freed
  first.

## 1. Starting state (verified before any training)

- **Repositories:**
  - Git repository: `/home/user/Code-agent`, branch `claude/affectionate-rubin-7rl1am`, remote
    `https://github.com/yazhiniyanMS/Code-agent` (public; the session's account `yazhiniyanMS` has
    admin rights).
  - `/home/user/lmwork` is the non-Git working area for checkpoints, data and runs.
- **Rules:** the selection and promotion rules are V6's and V10's, unchanged. They are written out in
  `experiments/v11/PLAN.md`, committed before any V11 result.
- **Checkpoints and SHA-256 hashes** (full list: `experiments/v11/inventory_sha256.txt`):

  | Checkpoint | Path | File SHA-256 | Tensor content SHA-256 | Size |
  | --- | --- | --- | --- | --- |
  | Round 54 (checkpoint of record) | `lmwork/loop/anchor/model.pt` (fp32) | `2538b55d5aee3718...` | - | 5.21 GB |
  | Round 54 optimizer state (needed to rebuild C2) | `lmwork/loop/anchor/optim.pt` | `bf51203b1f882b80...` | - | 2.13 GB |
  | C2 | `lmwork/v10/ckpt/C2_1301.ypk` | `931ae406e1ff3f34...` | `cc2f2fa07777e69e...` | 1.32 GB |
  | R1 | `lmwork/v10/ckpt/R1_1301.ypk` | `eb7ec5951101986...` | `9909c511043b8a16...` | 1.32 GB |
  | R6 | `lmwork/v10/ckpt/R6_1301.ypk` | `fcc97a345a292d1c...` | `0e8bf2a6a881b174...` | 1.35 GB |
  | R7 | `lmwork/v10/ckpt/R7_1301.ypk` | `cd3d6c78e329ff4b...` | `2d75e24a54747...` | 1.34 GB |

  - The `.ypk` files are lossless packs (`ycode/lm/packckpt.py`): bf16 bits XOR round 54's, compressed.
  - Restoring one needs round 54 at `lmwork/loop/anchor` and the tokenizer at
    `lmwork/data/tokenizer.json` (`7079ff7b...`).
  - The tensor content hash is checked on every unpack. C2's pack was made from the checkpoint whose
    file hash is `9c138498...`.
- **Restoring was verified end to end.** C2 was unpacked and evaluated on dev by the runner's own
  command, and all 62 answers are byte-identical to V9's record.
- **Resources:**
  - 4 CPU cores, no native bf16, 15.7 GB RAM, a 16 GB RAM disk.
  - The disk had **448 MB free** at the start.
  - Space came from the main worktree's copy of `models/` (the published v2-v5 models, 3.2 GB). It now
    uses the same sparse checkout as the staging worktree. All 120 blobs were checked present in Git
    first and are also on GitHub; `git sparse-checkout disable` restores them. No checkpoint was moved
    or deleted.

## 2. V11-A: R1 replicated at another seed

**Setup.** R1's spec (`experiments/v10/v10_r1.json`) with only the seeds changed (`experiments/v11/v11_a.json`):

- training seed 1302 and data seed 1303 (`data_seed_offset` 1);
- from the verified C2: the pack was unpacked with its content hash checked, and the trainer's log
  records loading that directory;
- otherwise identical: 36 steps, lr 1e-4 with warmup-stable-decay to 1e-5, batch 4, the top 24
  layers trained, fp32, no optimizer state carried over, 2 new examples for each of grammar v2's 350
  programs, and a 50% replay share from one production round of round 54's own mix.
- The data manifest gives 160,912 tokens: 700 curriculum examples (46.0%), 1,017 production
  examples (49.5%) and 43 original replay rows (4.5%).

**Result** (identical evaluation pipeline, greedy decoding, the same item sets):

| | Round 54 | C2 | R1 (seed 1301) | V11-A (seed 1302) | R1 recipe, mean of 2 seeds |
| --- | --- | --- | --- | --- | --- |
| Dev clean /44 | 27 | 28 | 30 | 30 | 30.0 |
| Unseen-v1 clean /53 | 2 | 3 | 3 | 1 | 2.0 |
| **Combined /97** | **29** | 31 | **33** | **31** | **32.0** |
| Lost round-54 passes, instruction_following | - | 2 | 3 | 3 | **3.0** |
| family_new /24 | 0 | 1 | 4 | 2 | 3.0 |
| Neighbours /38 | 4 | 2 | 2 | 3 | 2.5 |
| Held-out families /66 | 2 | 1 | 2 | 3 | 2.5 |
| Reproduction c2 /40 | 0 | 26 | 36 | 38 | 37.0 |
| Held-out combinations c2 /60 | 0 | 32 | 54 | 51 | 52.5 |
| Benchmark official · clean · robust | 22 · 19 · 20 | 21 · 18 · 20 | 23 · 20 · 23 | 27 · 24 · 24 | 25 · 22 · 23.5 |
| Robust passes dev /62 · unseen-v1 /53 (official items) | 38 · 1 | - | 42 · 2 | 41 · 1 | 41.5 · 1.5 |

V11-A cost 83 minutes of training (228 minutes with evaluation), 147,456 trained tokens, a peak of
10,891 MB of RAM, and its checkpoint hash is `c369abd0823f...` (the pack's content hash is
`944ded915c01...`).

**What replicated and what did not:**

- **Dev replicated exactly.** V11-A passes the same 30 clean dev items as R1, including R1's gains
  `write:first_negative` and `write:longest_word` and the three bug fixes C2 learned.
- **Unseen-v1 did not replicate.** It fell from 3 to 1; only `sum_at_odd_positions` remains.
  R1's `count_capitalized` did not come back at this seed. The R1 recipe's mean on unseen-v1 is 2.0,
  round 54's level, so the recipe's whole mean gain (+3) is on dev.
- **The forgetting replicated exactly.** These round-54 passes were lost in both seeds:
  - `bugfix:longest_word`
  - `bugfix:remove_spaces`
  - `write:remove_spaces`
  - unseen `count_words_longer_than`

  - The three dev items fall in the instruction-following category, so both seeds have 3 losses
    there.
  - V11-A also lost unseen `keys_above` (coding), which C2 loses and R1 had restored.
- **The held-out probes did not move:** new-family 2/24 (R1 4/24), neighbours 3/38 (round 54
  4/38), held-out families 3/66 (round 54 2/66). These are reported, never used to select.
- **The benchmark is the highest of any model at 27 · 24 · 24.** It is reported only, and the gain
  is within the seed-to-seed churn.

**Gates (PLAN.md, unchanged):**

- **Second-seed gate for V11-A:** combined 31 < 32, and 3 instruction-following losses → fails.
- **Promotion of the R1 recipe:**
  - every seed > 29: yes (33, 31);
  - mean combined >= 32: yes (32.0);
  - mean `items_lost` < 3 in every category: **no** (instruction_following 3.0).

  → **not promoted.** Round 54 stays the checkpoint of record.

**Conclusion:** R1's forgetting pattern is reproduced, not a single seed's accident. Whatever is
lost is lost in C2 already (the 4 shared items) and R1's consolidation does not restore it. That
triggers the forgetting investigation (section 3) and a retention-aware experiment (section 4).

Exact outputs of every changed item: `experiments/v11/results/changed_outputs.md`.

## 3. Why `remove_spaces` and the `longest_word` bug fix are lost

There are four items. Round 54 passes them; C2, R1, V11-A and every V10 continuation fail them.

| Item | Round 54's answer | C2 / R1 / V11-A answer | Round 54's pass under the robust tests |
| --- | --- | --- | --- |
| dev `bugfix:longest_word` | the fix (`<` back to `>`) | **the buggy code returned unchanged** | robust |
| dev `bugfix:remove_spaces` | the fix (`'_'` back to `''`) | **the buggy code returned unchanged** | robust |
| dev `write:remove_spaces` | `''.join(c for c in s if c.isalpha())` | unrelated code (`s.count('s')`, an indexing expression) | **weak**: drops digits and punctuation too |
| unseen `count_words_longer_than` | `sum(1 for c in text.split()[2:] if c.isalpha())`, which never reads `n` | `len(text.split())` and similar | **weak**: passes the official tests by coincidence |

So only the two bug fixes are robust skills that were really lost. The other two were coincidental
passes of wrong code. The pre-registered gate counts official passes, and it is applied unchanged.

**What was inspected** (`experiments/v11/forgetting_probe.py`, the production-data checks below, and
`experiments/v11/results/forgetting_probe.json`):

1. **Training targets.**
   - Round 54's own mix has 1,500 bug-fix examples per production round (seed 1302).
   - **None of them has the buggy code as its target.** All parse, and 1,475 of the 1,500 change
     exactly one line.
   - The target is always the concept's reference implementation, whose tests `verify_concepts()`
     runs. The bug is kept only if the buggy code fails those tests
     (`ycode/lm/basics.py: bugfix_examples`).
   - 291 fixes change only a comparison and 549 only a literal.
   - No production example uses the names `longest_word`, `remove_spaces`,
     `count_words_longer_than` or `keys_above`. The runner's hygiene check also removes any example
     sharing a name, AST skeleton or behaviour with a dev or unseen item.
   - The grammar v2 curriculum has no bug-fix examples, so nothing in C2's or R1's data teaches a
     wrong fix.
2. **Tokenization and loss masking.**
   - Prompt tokens weigh 0. In a bug fix the tokens the fix changes weigh 8 (`DIFF_WEIGHT`), the rest
     of their line 3, and operators kept elsewhere 2 (`ycode/lm/data.py: answer_weights`).
   - The decisive tokens (`>` and `''`) are therefore the most heavily trained tokens of such an
     example. Masking is not the cause.
3. **Sampling frequency.**
   - Round 54's production mix is 34% bug fixes by example (1,500 of 4,378).
   - C2's data holds about 1,473 production examples, a third of a round, beside 1,050 curriculum
     examples. R1's holds 1,023 beside 700.
   - So bug fixes fall to about 18-20% of the examples (about 500 in C2's data, 320 in R1's), and none of the curriculum examples is a bug
     fix.
   - Adding 350 correct bug-fix examples (V10-R5) did not restore a single item, so frequency alone
     is not the cause either.
4. **Per-token log-probabilities of the correct answer.** Exact items, read only for diagnosis,
   never trained on:

   | Item | Model | log p(correct fix) | log p(buggy code echoed) | At the decisive position |
   | --- | --- | --- | --- | --- |
   | `bugfix:longest_word` | round 54 | **-0.95** | -3.75 | fixes |
   | | C2 | -2.63 | **-1.25** | p(`<`, buggy) 0.79 vs p(`>`) 0.20 |
   | | R1 | -1.69 | **-1.29** | p(`<`) 0.59 vs p(`>`) 0.40 |
   | `bugfix:remove_spaces` | round 54 | **-1.97** | -2.90 | fixes |
   | | C2 | -3.45 | **-1.41** | prefers `'_'` (buggy) over `'')` |
   | | R1 | -2.68 | **-1.28** | the same |
   | `write:remove_spaces` | round 54 | -36.2 (its own `isalpha` answer: -23.0) | - | a weak pass |
   | | C2 | -21.7 | - | its own answer writes `s.replace('_'` where the fix has `s.replace(' ', '')` |

   - The fix is never far away: the correct token keeps 20-40% of the probability.
   - What changed is a **copy bias at the one decisive token**: the model hands the buggy token back.
   - R1's consolidation moves `longest_word` halfway back (0.20 → 0.40), not over the line.
5. **Generated outputs.** All four models' exact outputs are in
   `experiments/v11/results/changed_outputs.md`. C2, R1 and V11-A answer both dev bug fixes with the
   prompt's code verbatim.

6. **Curriculum interference** was checked and ruled out.
   - C2's 1,050 grammar v2 examples contain no `'_'`, no `replace(' ', ...)` and no
     `len(x) < len(y)`.
   - So the wrong tokens are not taught by similar-looking curriculum code.
   - The model's behaviour on bug-fix prompts shifted from editing toward copying.
7. **What C2 forgot on round 54's own training mix.**
   - Every example of one production round (seed 1302, 4,378 examples; no dev, unseen or benchmark
     item) was scored under round 54 and under C2. The score is the weighted answer loss, exactly
     the training objective (`experiments/v11/forgetting_scores.py`, both passes in fp32 compute on
     the bf16 weights).
   - The scores are in `experiments/v11/results/forgetting_1302.json`; the summary below is in
     `forgetting_1302_summary.txt`.

   | Kind | Examples | Loss round 54 | Loss C2 | Mean rise | Share whose loss rose |
   | --- | --- | --- | --- | --- | --- |
   | write | 2,878 | 0.005 | 0.025 | +0.019 | 92% |
   | bug fix, other edit | 660 | 0.001 | 0.011 | +0.010 | 100% |
   | bug fix, literal only | 549 | 0.003 | 0.009 | +0.006 | 99% |
   | bug fix, comparison only | 291 | 0.001 | 0.004 | +0.004 | 99% |
   | all | 4,378 | 0.004 | 0.019 | +0.015 | 95% |

   - C2 got worse on 95% of round 54's own training examples. The most forgotten are write examples
     of non-grammar concepts (Collatz steps, divisors, element-wise products) and bug fixes such as
     `remove_vowels` and `multiply_lists`.
   - **By average loss, bug fixes are forgotten least.** The decisive token is one of about 60
     answer tokens, so the average hides the one place the model now copies.
   - The score that matters for the lost items is therefore not the one V11-B's sampler uses.
     Drawing replay by this score shifts it toward write examples:

   | Replay draw (same 50% token share, data seed 1302) | Examples | Comparison bug fixes | All bug fixes | Write |
   | --- | --- | --- | --- | --- |
   | uniform (R1) | 959 | 56 | 320 | 639 |
   | by forgetting score (V11-B) | 976 | 20 | 205 | 771 |

   - V11-B therefore tests whether restoring the broadly forgotten write skills helps. The plan
     defined it this way before these numbers were known, and it runs as pre-registered.
   - It is not expected to restore the bug-fix echo, which an average-loss score cannot see.

## 4. Experiments after the replication

### 4.0 Model and methodology (common to every run)

- **Model.** YCode-LM v5, 1,523,202,304 parameters, unchanged architecture:
  - decoder-only, 34 layers × 2048 wide;
  - 32 query and 8 key-value heads, QK-norm, RoPE, SwiGLU, RMSNorm;
  - context 1,024;
  - byte-level BPE tokenizer with 8,192 tokens.
- **Training** (`ycode.lm.cli sft` through `ycode.lm.experiment`):
  - only the top 24 of 34 layers are trained;
  - Lion, fp32 master weights with gradient checkpointing;
  - lr 1e-4 warmup-stable-decay: 2 warmup steps, stable, then decay to 1e-5 over the last 20%;
  - 4 packed rows of 1,024 tokens per step, one epoch of the training split. The first 5% of rows
    are a validation split, never trained.
  - **Loss weights:**
    - prompt tokens weigh 0;
    - a written function's body weighs 3;
    - in a bug fix, the changed tokens weigh 8, the rest of their line 3, the def line 2, and kept
      operators and literals 2.
  - **Checkpoints** are compact bf16 (the trained tensors, referencing round 54). They are stored on
    disk as lossless packs.
- **Data.** Curriculum examples from grammar v2's 350 training programs, plus 50% of the tokens
  replayed from one production round of round 54's own mix, plus a few rows of the original
  instruction data.
  - The hygiene check removes any example sharing a name, AST skeleton or behaviour with a dev,
    unseen or benchmark item.
  - Training never sees the unseen-v2 holdout.
- **Evaluation** (identical for every model):
  - Greedy decoding, at most 200 new tokens, no repeated 12-gram.
  - The extracted code must pass every unit test, run in a subprocess with a 5 s timeout.
  - **Clean** scores leave out the items listed as contaminated. **Robust** scores rerun the
    passing code on extra inputs.
  - Task sets (`experiments/v7/prompt_diagnostic.py`) count a program correct only if it is right
    on all of its probe inputs.
  - Unseen-v2 was never evaluated.

### 4.1 The matrix

Every run was defined in `experiments/v11/PLAN.md` (and its amendments 1 and 2) before it started.
The gates were fixed before any V11 result and never changed.

| Run | Hypothesis | Changed variable (one) | Expected | Forgetting risk | Seeds | Resources (measured or estimated) | Stop / next |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | R1's +4 is not one seed's luck | seeds only | combined about 33 with the same losses | as R1 | train 1302, data 1303 | 83 min of training plus 145 min of evaluation, 10.9 GB peak | always run |
| B | replaying what C2 forgot restores it | how the 50% replay is drawn: by C2's forgetting score (section 3) | combined >= 32 with the losses restored | narrower replay (fewer bug fixes, see section 3) | 1301 (data 1302) | 2 h of scoring, about 80 min of training, about 2 h of evaluation | second seed if both gates pass, else W |
| W | round 54's behaviour and the R1 recipe's gains sit close together in weight space | the weights: 0.5 R1-recipe + 0.5 round 54, fixed in advance | bug fixes back, combined >= 32 | the gains halve | R1 (1301) and V11-A (1302) | no training, about 2.5 h of evaluation per seed | if both seeds fail, stop W |
| C | the loss happens while C2 trains, so retention must start at round 54 | one stage from round 54 (C2's and R1's examples together) with replay drawn by forgetting score within each kind | the instruction-following items kept, combined >= 32 | 5 examples per program in one stage | 1301 (data 1302) | 88 steps (about 3 h) plus about 2 h of evaluation | only if B and W fail; second seed if it passes |
| (D) | grammar v3 compositions add transfer once retention holds | + grammar v3's programs | unseen-v1 up (R6 reached 5/53) | dev fell in R6 | - | about 4 h | only after a run passes |

- The correct bug-fix contrast pairs moved behind C (amendment 1). V10-R5 had added 350 correct
  bug-fix examples without restoring a single item.
- Learning rate, architecture and full retraining were not touched.
- Each run's configuration file is listed below.

| Run | Configuration | Records |
| --- | --- | --- |
| A | `experiments/v11/v11_a.json` | `experiments/v11/results/runs/A/1302/` |
| B | `experiments/v11/v11_b.json` (weights `experiments/v11/results/forgetting_1302.json`) | `experiments/v11/results/runs/B/1301/` |
| W | `experiments/v6/soup.py`, `experiments/v11/drivers/run_w.sh`, evaluated by `experiments/v11/eval_only.py` | `experiments/v11/results/runs/W/<seed>/` |
| C | `experiments/v11/v11_c.json` | `experiments/v11/results/runs/C/1301/` |

### 4.2 Scores (the same pipeline and item sets for every row)

| Model | Dev /44 | Unseen-v1 /53 | Combined /97 | family_new /24 | Neighbours /38 | Held-out families /66 (expressible 7 · other 59) | Reproduction c2 /40 · c3 /40 | Held-out combinations c2 /60 · c3 /60 | Bench official · clean · robust |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Round 54 | 27 | 2 | **29** | 0/24 | 4/38 | 2/66 (0/7 · 2/59) | 0/40 · 0/40 | 0/60 · 0/60 | 22 · 19 · 20 |
| C2 | 28 | 3 | **31** | 1/24 | 2/38 | 1/66 (0/7 · 1/59) | 26/40 · 0/40 | 32/60 · 0/60 | 21 · 18 · 20 |
| R1 1301 | 30 | 3 | **33** | 4/24 | 2/38 | 2/66 (1/7 · 1/59) | 36/40 · - | 54/60 · - | 23 · 20 · 23 |
| R6 1301 | 26 | 5 | **31** | 3/24 | 1/38 | 2/66 (1/7 · 1/59) | 39/40 · 30/40 | 46/60 · 43/60 | 21 · 18 · 20 |
| R7 1301 | 27 | 2 | **29** | 2/24 | 2/38 | 0/66 (0/7 · 0/59) | 40/40 · - | 55/60 · - | 24 · 21 · 21 |
| V11-A 1302 | 30 | 1 | **31** | 2/24 | 3/38 | 3/66 (1/7 · 2/59) | 38/40 · - | 51/60 · - | 27 · 24 · 24 |
| V11-B 1301 | 29 | 2 | **31** | 1/24 | 3/38 | 2/66 (0/7 · 2/59) | 35/40 · - | 50/60 · - | 26 · 23 · 25 |
| V11-C 1301 | 23 | 2 | **25** | 1/24 | 1/38 | 2/66 (0/7 · 2/59) | 38/40 · - | 51/60 · - | 24 · 20 · 24 |
| V11-W 1301 | 26 | 1 | **27** | 0/24 | 4/38 | 2/66 (0/7 · 2/59) | 21/40 · - | 18/60 · - | 25 · 24 · 23 |

| Model | Lost against round 54 (by category) | Gained against round 54 | Lost / gained against C2 | Category gate (>= 3 fails) |
| --- | --- | --- | --- | --- |
| C2 | 5: dev:bugfix:longest_word, dev:bugfix:remove_spaces, dev:write:remove_spaces, unseen:write:count_words_longer_than, unseen:write:keys_above ({'instruction_following': 2, 'reasoning': 2, 'coding': 1}) | 7: dev:bugfix:first_negative, dev:bugfix:index_of_min, dev:bugfix:product, dev:write:first_three_chars, unseen:write:count_capitalized, unseen:write:double_plus_one, unseen:write:sum_at_odd_positions | 0 / 0 | passes |
| R1 1301 | 4: dev:bugfix:longest_word, dev:bugfix:remove_spaces, dev:write:remove_spaces, unseen:write:count_words_longer_than ({'instruction_following': 3, 'hallucination': 1}) | 8: dev:bugfix:first_negative, dev:bugfix:index_of_min, dev:bugfix:product, dev:write:first_negative, dev:write:first_three_chars, dev:write:longest_word, unseen:write:count_capitalized, unseen:write:sum_at_odd_positions | 1 / 3 | fails |
| R6 1301 | 5: dev:bugfix:longest_word, dev:bugfix:remove_spaces, dev:write:count_greater, dev:write:remove_spaces, unseen:write:count_words_longer_than ({'coding': 1, 'instruction_following': 3, 'reasoning': 1}) | 7: dev:bugfix:first_negative, dev:bugfix:product, dev:write:first_three_chars, unseen:write:count_capitalized, unseen:write:double_plus_one, unseen:write:is_vowel, unseen:write:sum_at_odd_positions | 2 / 2 | fails |
| R7 1301 | 5: dev:bugfix:longest_word, dev:bugfix:remove_spaces, dev:write:remove_spaces, dev:write:starts_with_the, unseen:write:count_words_longer_than ({'instruction_following': 3, 'reasoning': 2}) | 5: dev:bugfix:index_of_min, dev:bugfix:product, dev:write:first_three_chars, dev:write:longest_word, unseen:write:count_capitalized | 4 / 2 | fails |
| V11-A 1302 | 5: dev:bugfix:longest_word, dev:bugfix:remove_spaces, dev:write:remove_spaces, unseen:write:count_words_longer_than, unseen:write:keys_above ({'instruction_following': 3, 'reasoning': 1, 'coding': 1}) | 7: dev:bugfix:first_negative, dev:bugfix:index_of_min, dev:bugfix:product, dev:write:first_negative, dev:write:first_three_chars, dev:write:longest_word, unseen:write:sum_at_odd_positions | 2 / 2 | fails |
| V11-B 1301 | 5: dev:bugfix:remove_spaces, dev:bugfix:sum_of_squares, dev:write:remove_spaces, dev:write:scale_each_by_three, unseen:write:count_words_longer_than ({'instruction_following': 2, 'reasoning': 3}) | 7: dev:bugfix:first_negative, dev:bugfix:index_of_min, dev:bugfix:product, dev:write:first_negative, dev:write:first_three_chars, dev:write:index_of_min, unseen:write:sum_at_odd_positions | 4 / 4 | fails |
| V11-C 1301 | 7: dev:bugfix:longest_word, dev:bugfix:remove_spaces, dev:bugfix:sum_of_squares, dev:write:remove_spaces, dev:write:scale_each_by_three, dev:write:subtract_three_each, unseen:write:count_words_longer_than ({'coding': 1, 'instruction_following': 2, 'hallucination': 2, 'reasoning': 2}) | 3: dev:bugfix:first_negative, dev:bugfix:product, unseen:write:count_capitalized | 7 / 1 | passes |
| V11-W 1301 | 3: dev:bugfix:remove_spaces, dev:write:remove_spaces, unseen:write:count_words_longer_than ({'instruction_following': 1, 'reasoning': 2}) | 1: dev:write:longest_word | 7 / 3 | passes |

| Run | Start | Trained tokens | Steps | Train min | Total min | Peak RAM MB | Checkpoint sha256 | Packed content sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C2 | round 54 | 229,376 | 56 | 166 | 268 | 11311 | `9c1384982c9b` | `cc2f2fa07777` |
| R1 1301 | C2 | 147,456 | 36 | 74 | 130 | 10827 | `0758684e76d6` | `9909c511043b` |
| R6 1301 | C2 | 270,336 | 66 | 128 | 212 | 10810 | `bc429d345b36` | `0e8bf2a6a881` |
| R7 1301 | C2 | 303,104 | 74 | 196 | 326 | 10941 | `0001c192288a` | `2d75e24a5474` |
| V11-A 1302 | C2 | 147,456 | 36 | 83 | 228 | 10891 | `c369abd0823f` | `944ded915c01` |
| V11-B 1301 | C2 | 147,456 | 36 | 82 | 177 | 10954 | `462168ece2ac` | `3861bcb030f1` |
| V11-C 1301 | round 54 | 360,448 | 88 | 224 | 366 | 11316 | `f9cbdd6fe2a0` | `14e953032857` |
| V11-W 1301 | R1 + round 54 (no training) | 0 | 0 | 0 | 91 | 16448 | `0b4f6c8f994b` | - |

Notes on the tables:

- **Selection metric:** clean dev /44 plus clean unseen-v1 /53. "Lost" lists clean items round 54
  passes and the model fails, grouped by the failure category of the model's answer; three or more
  in one category fails the rule.
- **Rows marked `-`:** that set was not run for that model.
- **V11-W:**
  - Its evaluation took 91 minutes, and building it took 0.5 minutes.
  - Its peak RAM (16,448 MB) is the averaging process's high-water mark, which counts the two
    memory-mapped checkpoints' pages. These are file cache, not allocations.
- **Time per stage:** each run's training and evaluation times are in its `result.json`. For
  example, V11-B took 82 minutes of training, then dev 10.8, unseen 6.9, benchmark 7.3, and about
  70 minutes for the five task sets.
- **Robust passes** (dev / unseen-v1, clean): round 54 26 / 1, R1 30 / 2, V11-A 30 / 1,
  V11-B 29 / 2, V11-W 26 / 1, V11-C 23 / 1.
- **Exact outputs** of every changed item for every model: `experiments/v11/results/changed_outputs.md`
  (108 items).

### 4.3 What each run showed

**V11-B (replay drawn by forgetting score): combined 31 (dev 29, unseen-v1 2). Fails the second-seed
gate on both counts.**

- **Restored:** the `longest_word` bug fix (it returns `>` again), so instruction-following losses fall
  from 3 to 2.
- **Newly lost, both clean reasoning items:**
  - `bugfix:sum_of_squares`
  - `write:scale_each_by_three`

  With unseen `count_words_longer_than` that makes 3 reasoning losses, so the category rule fails in
  another category.
- **Still lost:** `remove_spaces`, both as a bug fix and written.
- **Reported, never used to select:** the benchmark reaches 26 · 23 · 25, and robust dev / unseen-v1
  is 29 / 2.
- **Reading:** redrawing the replay toward the examples C2 forgot most moves which items are lost.
  It does not reduce the count.

**V11-W (0.5 R1 + 0.5 round 54, no training): combined 27 (dev 26, unseen-v1 1). Fails, and below
round 54.**

- **Restored:** the `longest_word` bug fix. Only 3 clean losses against round 54, so the category
  rule passes.
- **Lost:** nearly all of R1's gains.
  - The bug fixes `first_negative`, `index_of_min` and `product` are gone, and so is
    `first_three_chars`.
  - Reproduction of C2's programs falls from 36/40 to 21/40, and held-out combinations from 54/60 to
    18/60.
- **Kept:** round 54's neighbour score (4/38) is back, and the benchmark is 25 · 24 · 23.
- **Reading:** halfway in weight space keeps round 54's behaviour and loses the new skills. The R1
  recipe's gains and its losses come together.
- **Stop:** under amendment 3, W's second seed is not run. Every seed must beat 29 and seed 1301
  scored 27.

**V11-C (retention from the start, one stage from round 54): combined 25 (dev 23, unseen-v1 2).
Fails. It is the worst V11 result.**

- **Configuration:**
  - five grammar v2 examples per program in one 88-step stage from round 54, with round 54's
    optimizer state;
  - replay drawn by forgetting score within each kind, keeping round 54's 35% bug-fix share.
- **Lost against round 54:** 7 clean items:
  - six dev items: both bug fixes (`longest_word`, `remove_spaces`), `write:remove_spaces`,
    `bugfix:sum_of_squares`, `write:scale_each_by_three` and `write:subtract_three_each`;
  - unseen `count_words_longer_than`.
- **The losses are spread across categories:** at most 2 in any one (instruction following 2,
  hallucination 2, reasoning 2, coding 1). So the category rule alone would pass, but combined 25 is
  far below 32.
- **Gained:** only 3 clean items (the `first_negative` and `product` bug fixes and unseen
  `count_capitalized`, the last a weak pass).
- **Reading:**
  - Retention from the start does not prevent the loss. More curriculum in a single stage forgets
    more.
  - C2's two-stage path, a short stage then consolidation, is the better trade.

**Across all runs (seed replication and generalization):**

- **Dev:** the R1 recipe's dev gain is the one replicated result: 30 and 30 over two seeds, against
  round 54's 27. These are the same 30 items, including `first_negative` (written and as a bug fix),
  `first_three_chars`, `index_of_min` and `product`.
- **Unseen-v1:** no run gains reliably. The scores are 3, 1, 2, 1 and 2 against round 54's 2. R1's
  `count_capitalized` came back only in C (a weak pass), and R1's `sum_at_odd_positions` pass is weak
  under the robust tests.
- **Held-out probes** (reported, never used to select):
  - New-family probe: 0-4 of 24.
  - Neighbours: 2-4 of 38; no run beats round 54's 4.
  - Held-out families: 2-3 of 66.

  These do not show transfer beyond the trained grammar.
- **The benchmark** rose in every R1-recipe run (23, 27, 26 official, against round 54's 22). It is
  reported only, never used to choose, and the gain is within the 3-4 item churn seen between seeds.
- **Leakage and duplicates.**
  - For every run, the runner's hygiene check removed 0 training examples. It checks each example's
    name, AST skeleton and behaviour against every dev, unseen and benchmark item.
  - All gains counted above are clean items, not in the contamination lists.
  - No production example uses the names of the four lost items (section 3).
- **Weak passes** (the robust tests fail the code):
  - R1's `sum_at_odd_positions` and C's `count_capitalized` among the gains;
  - round 54's own `write:remove_spaces` and `count_words_longer_than` among the losses.

## 5. Decision: validated best and best experimental candidate

Under the rules in `PLAN.md`, fixed before any V11 result:

| Candidate | Seeds run | Combined per seed | Second-seed gate (>= 32, no category >= 3) | Promotion (every seed > 29, mean >= 32, mean losses < 3 per category) |
| --- | --- | --- | --- | --- |
| R1 recipe (R1, V11-A) | 1301, 1302 | 33, 31 | R1: score yes, category no (3 instruction following); A: no on both | **no**: mean 32.0 but instruction-following losses 3.0 |
| V11-B | 1301 | 31 | no (31; reasoning 3) | - |
| V11-W | 1301 | 27 | no | no (27 <= 29); second seed not run (amendment 3) |
| V11-C | 1301 | 25 | no (25) | - |

- **Validated best: Round 54.** It is the checkpoint of record, unchanged.
  - File SHA-256: `2538b55d5aee371899a988799168042af4da27c03cafff3417a3c9925b8e8e98`.
  - Combined 29, benchmark 22 · 19 · 20.
  - Manifest: `experiments/v11/artifacts/validated_best.json`.
- **Best experimental candidate: R1** (V10 seed 1301). Highest combined score of any checkpoint (33).
  - **Not validated.** It fails the category rule, and its replication reached 31.
  - Pack SHA-256: `eb7ec5951101986951521256ae98ee40e81ef673724058c3c89e7d001d408eb5`.
  - Tensor content SHA-256: `9909c511043b8a166b4a713e50ad206d84a47cbc927e9ed17bfc9b0e161c6ec3`.
  - Manifest: `experiments/v11/artifacts/best_experimental_candidate.json`.
  - Restore check: `experiments/v11/results/restore_check_R1*.json`.
- **R1 over V11-A as the candidate:** both fail. R1 is higher on the primary metric (33 against 31),
  and A has the higher benchmark, which is never used to choose.
- **What V11 establishes:**
  - The R1 recipe's dev gain is real and reproducible (+3 on two seeds).
  - Its unseen-v1 gain is not.
  - Its forgetting is systematic, and V11 identified its cause.

## 6. Checkpoints, restoration and disk

The full list, with hashes recomputed from disk when it was written, is in
`experiments/v11/artifacts/CHECKPOINTS.json`.

| Checkpoint | Stored as | Path | Size | File SHA-256 | Tensor content SHA-256 | Parent |
| --- | --- | --- | --- | --- | --- | --- |
| Round 54 (validated best) | fp32 training checkpoint (+ optimizer state) | `lmwork/loop/anchor/model.pt` (+ `optim.pt`, `info.json`, `sampler.json`) | 5.21 GB (+ 2.13 GB) | `2538b55d5aee3718...` | - | YCode-LM v5 + 72 steps |
| C2 | lossless pack | `lmwork/v10/ckpt/C2_1301.ypk` | 1.32 GB | `931ae406e1ff3f34...` | `cc2f2fa07777e69e...` | round 54 |
| R1 (best experimental) | lossless pack | `lmwork/v10/ckpt/R1_1301.ypk` | 1.32 GB | `eb7ec59511019869...` | `9909c511043b8a16...` | C2 |
| R6 | lossless pack | `lmwork/v10/ckpt/R6_1301.ypk` | 1.35 GB | `fcc97a345a292d1c...` | `0e8bf2a6a881b174...` | C2 |
| R7 | lossless pack | `lmwork/v10/ckpt/R7_1301.ypk` | 1.34 GB | `cd3d6c78e329ff4b...` | `2d75e24a54747...` | C2 |
| V11-A | lossless pack | `lmwork/v11/ckpt/A_1302.ypk` | 1.32 GB | `45c7eeb8153c99b5...` | `944ded915c016b9d...` | C2 |
| V11-B | lossless pack | `lmwork/v11/ckpt/B_1301.ypk` | 1.32 GB | `7b9babf50f731fab...` | `3861bcb030f17a0c...` | C2 |
| V11-C | lossless pack | `lmwork/v11/ckpt/C_1301.ypk` | 1.39 GB | `33dcebe32341...` | `14e953032857ae3b...` | round 54 |
| V11-W | not stored: deterministic from R1 + round 54 | rebuild below | (2.13 GB) | `0b4f6c8f994b267409fb...` (model.pt) | stored tensors `5254e2a3f72d9c28...` | R1, round 54 |

**Restoring and evaluating.** Every pack needs round 54 at `/home/user/lmwork/loop/anchor` and the
tokenizer at `/home/user/lmwork/data/tokenizer.json` (`7079ff7b...`). From the repository root:

```bash
python -m ycode.lm.packckpt unpack /home/user/lmwork/v10/ckpt/R1_1301.ypk /dev/shm/r1   # checks every tensor
python -m ycode.lm.cli eval --model /dev/shm/r1 --problems dev --device cpu --answers-out dev.jsonl
python experiments/v11/verify_restore.py R1 /home/user/lmwork/v10/ckpt/R1_1301.ypk \
    experiments/v10/results/runs/R1/1301 out.json        # unpack + dev + unseen, compared with the record
# V11-W: unpack R1 as above, then
python experiments/v6/soup.py /dev/shm/w /dev/shm/r1:0.5 /home/user/lmwork/loop/anchor:0.5   # stored tensors' content sha256 5254e2a3... (file 0b4f6c8f... when built)
# a standalone bf16 copy (round 54 folded in), with MANIFEST.json and a reload check:
python experiments/v11/package_checkpoint.py /dev/shm/r1 /dev/shm/pub --name ycode-v11-r1 \
    --record experiments/v10/results/runs/R1/1301/result.json
```

- **Restoration checks done in V11:**
  - C2: unpacked, dev reproduced 62/62 on the earlier host (section 1).
  - R1: unpacked in a clean process. Tensors bit-identical; dev 62/62 and unseen-v1 52/53 answers
    byte-identical; all scores equal. The one differing answer was reproduced identically in a
    second clean restore. It fails either way, a near-tie that depends on the CPU.
  - V11-A, V11-B, V11-C: each pack was compared bit for bit with the model that was evaluated.
- **Configurations and seeds:** `experiments/v11/v11_{a,b,c}.json` and the run records'
  `result.json` (training seed, data seed and every option). Data manifests with token counts and
  file hashes are under `experiments/v11/results/runs/*/*/manifest.json`. Training logs are under
  `train.log` / `train_log.json` / `metrics.jsonl`.

**Disk** (`/` is a fixed per-session allowance):

- **Start of V11:** 448 MB free.
- **Space freed:**
  - the main worktree's copy of `models/` (3.2 GB, still in Git and on GitHub);
  - V7's L2 checkpoint packed losslessly in place (0.8 GB saved; `PACKED.txt` beside it records
    both hashes).
- **Now:** 663 MB free after the three new V11 packs (4.0 GB).
- **Retained:** round 54 (7.3 GB with its optimizer state), V10 packs (5.0 GB), V11 (3.8 GB), V6
  (4.2 GB) and V7 (1.2 GB) packs, and the Git store (2.6 GB).
- **Nothing was deleted to make room.** The RAM disk is empty at the end.

## 7. Publishing to GitHub

**Published to GitHub** (branch `claude/affectionate-rubin-7rl1am` of
`https://github.com/yazhiniyanMS/Code-agent`, pushed and checked):

- this report;
- the plan and its three amendments;
- every configuration (`v11_*.json`) and driver (`experiments/v11/drivers/`);
- all records (`experiments/v11/results/`: answers, scores, task-set rows, manifests, training
  logs, forgetting scores);
- the checkpoint manifest with SHA-256 hashes (`experiments/v11/artifacts/`);
- the tools (`forgetting_scores.py`, `forgetting_summary.py`, `forgetting_probe.py`,
  `eval_only.py`, `compare.py`, `changed_outputs.py`, `verify_restore.py`, `package_checkpoint.py`,
  `write_manifests.py`, `publish_lfs.sh`).

The commit is listed at the end of this section.

**The checkpoint itself is NOT on GitHub. Two exact blockers:**

1. **Release assets.**
   - The API answers `403` when creating a release: "Creating, editing, or deleting releases is not
     permitted for this session type."
   - The repository's only release, `v5.0.0`, has no assets. Attaching a V11 checkpoint to it would
     mislabel it, so it was not used.
2. **Git LFS.**
   - The LFS batch API accepts an upload (tested in V11 with a 1.35 GB object).
   - The environment's network policy refuses `lfs.github.com` (`CONNECT` answered `403`), where
     GitHub's LFS verify step goes. So an uploaded object is never registered, and a later download
     returns 404. This was rechecked on 2026-10-28 after the container moved hosts, with the same
     result.

**Two rejected alternatives:**

- **Ordinary Git:** the repository's own convention for v2-v5 is bf16 shards under 50 MB committed
  to ordinary Git (`models/ycode-lm-v5`, 3.3 GB). It is not used for V11, because you asked for no
  multi-gigabyte checkpoint in ordinary Git history.
- **Another host:** no other host was used, because you asked for GitHub.

**To publish:**

1. Allow `lfs.github.com` in the environment's Network access settings: the environment menu in
   the session's title bar → Edit → Network access, then add `lfs.github.com` under Allowed domains
   (or choose a broader level). The steps are at
   https://code.claude.com/docs/en/cloud-environments#network-access.
2. Run `experiments/v11/publish_lfs.sh`. It:
   - exports a standalone bf16 checkpoint with round 54 folded in (about 3.05 GB, shards under
     1.9 GB), so restoring needs nothing local;
   - writes `MANIFEST.json` with every file's SHA-256 and the tensor content hash;
   - reloads it in a separate process;
   - commits only the LFS pointers and the manifest, and pushes;
   - downloads every object back and checks it against the manifest.

   Its pointer handling (an object store on the RAM disk) was tested locally. The push and download
   steps could not be tested here.

No upload is claimed.

COMMIT_LINE

## 8. Next experiment

### V11-D: string-literal bug fixes, from C2, R1's recipe plus one addition

**The evidence:**

- `bugfix:remove_spaces` (fix `'_'` back to `''`) is lost by every model trained after round 54: C2,
  R1, R4-R7, V11-A, B, C, and W's interpolation.
- In a whole production round of round 54's own mix, **only 29 of 1,500 bug fixes change a string
  literal, and none fixes a literal to the empty string**. The grammar v2 curriculum has no bug
  fixes at all.
- So no replay of round 54's data, however it is drawn, contains the skill. Round 54's ability here
  came from earlier training and is the first thing any fine-tuning wears away.
- The other real loss, `longest_word` (a comparison fix, 291 of them per round), comes back easily:
  B and W both restored it.

**The run:**

- **Hypothesis:** correct bug-fix examples whose fix is a string literal, including the empty
  string, restore the edit in the class that replay cannot supply.
- **Changed variable:** R1's recipe from C2, plus string-literal mutations of round 54's own
  training concepts in its bug-fix generator (`ycode/lm/basics.py: _TEXT_MUTATIONS`), for example
  `''` ↔ `'_'` and `' '` ↔ `''`.
  - The targets are always the verified reference code. The decisive tokens are weighted 8 as usual.
  - The runner's hygiene check keeps `remove_spaces`' name, skeleton and behaviour out.
  - Pairs of fixes with opposite directions are the contrast.
- **Expected:** `bugfix:remove_spaces` restored. With R1's other items unchanged that leaves 2
  instruction-following losses, inside the gate, at about 33-34 combined.
- **Risk:** this targets a failure class seen on dev. A gain on that single item is weak evidence,
  so it needs the second seed and an unchanged unseen-v1 before any claim.
- **Seeds:** 1301 (data 1302), then 1302 (data 1303) if the second-seed gate passes.
- **Cost:** about 3 h per seed.

**Prerequisite: disk.**

- With V11's packs kept, the disk has 0.66 GB free, and a run needs 1.3-1.4 GB for its durable
  pack.
- V11 did not delete anything to make room. The candidates are the V6 and V7 packs, about 5.7 GB:
  - `v6/runs/{D,G}/1302`;
  - `v6/soups/DG130{1,2}c`;
  - `v7/runs/L2/1301`.

  Their removal is your decision, or more storage can be added.

**One question to settle before V12, not after its results:**

- Two of the three items that block the R1 recipe at the category rule are weak passes of round 54
  (`write:remove_spaces`, `count_words_longer_than`): round 54's own code fails the robust tests.
- The rule counts official passes and was applied unchanged in V11.
- Whether V12 should count only robust passes as retention is a choice to make and pre-register
  before any V12 run.
