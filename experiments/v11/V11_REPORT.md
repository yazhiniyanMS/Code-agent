# YCode V11: replicating R1, finding the forgetting, and publishing

Status: STATUS

## Executive summary

SUMMARY

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
| Robust dev /62 · unseen-v1 /53 | 38 · 2 | - | 40 · 3 | 41 · 1 | |

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

  V11-A also lost unseen `keys_above` again; C2 loses it, and R1 had restored it. That is 3
  instruction-following losses in both seeds.
- **The held-out sets and the benchmark rose a little.** These sets are reported but never used to
  select. V11-A's benchmark of 27 · 24 · 24 is the best of any model. The gain is not significant,
  and the benchmark is never used to choose.

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

RESULTS_SECTION

## 5. Decision: validated best and best experimental candidate

DECISION_SECTION

## 6. Checkpoints, restoration and disk

ARTIFACTS_SECTION

## 7. Publishing to GitHub

PUBLISH_SECTION

## 8. Next experiment

NEXT_SECTION
