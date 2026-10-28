# YCode V10: fine-tuning C2 for generalization, not memorization

Status: **complete**. Seven fine-tuning runs, all from C2, one change at a time. No second seed was run:
no candidate passed the pre-registered gate.

**About the model:** YCode-LM v5 is a decoder-only GPT with 1.52B parameters, not an encoder-decoder.
Fine-tuning trains the top 24 of its 34 blocks.

**Hygiene:**
- Every number here is measured on dev, unseen-v1 and the development task sets.
- The benchmark is reported only and played no part in any choice.
- unseen-v2 was never evaluated. The only use of it is the training-data decontamination that has run
  since V6, which drops any example whose name, skeleton or behaviour matches a held-out item.
- Round 54 is unchanged (sha256 `2538b55d...`).
- C2 was never overwritten. It was rebuilt bit for bit (R0) and kept.
- The decision rules were written before any V10 result (`experiments/v10/PLAN.md`). Each of the three
  amendments, adding R5, R6 and R7, was written before its run.

## Answer

**No candidate passes the pre-registered rules, so there is no reliable improvement to declare.
Round 54 stays the official checkpoint. The best candidate is R1, which is not yet validated.**

- **C2 reproduces bit for bit.** The rebuilt checkpoint matches V9-1's sha256.
- **R1 is the best candidate:** C2 plus one epoch on 2 new examples of each of its 350 programs.
  - Combined 33/97: dev 30/44 and unseen-v1 3/53, against C2's 31 and round 54's 29.
  - New-family probe 4/24 (C2 1, round 54 0).
  - Training-program reproduction 36/40, held-out combinations 54/60.
  - Benchmark 23/65 (round 54 22, reported only).
  - It cost 74 minutes of training.
- **R1 fails the pre-registered category gate,** which this plan used to decide second seeds:
  - It loses 3 instruction-following dev items round 54 passes. C2 had already lost the same 3.
  - One of them is round 54's weak pass, an answer that is wrong in general.
  - So R1 got no second seed, and its +4 over round 54 is one seed's result.
- **Replay (A) and learning rate (D) do not restore those items:**
  - R7 (75% replay): combined 29.
  - R4 (lr 3e-5): combined 29.
  - R5 (verified bug-fix examples): combined 27.
  - All three still lose them.
- **Diverse cross-family examples (B):**
  - Learning composes: R6 learns both grammars and reaches unseen-v1 5/53, the highest of any model.
  - Held-out families: round 54 2/66, R2 3/66. The tasks gained are exactly the ones that are new
    combinations of trained parts, and nothing moves on the 59 that are not. Neighbours stay at
    1-2/38, against round 54's 4.
- **Contrastive pairs (C)** were built and unit-tested, but not run. R1 had already removed most part-swap
  errors, and the blocker turned out to be forgetting.

## 1. Starting point (R0)

- **C2's checkpoint no longer existed.** V9-1 deleted it after evaluation: the disk had 1.7 GB free,
  and a candidate is 2.1 GB.
- **R0 reran C2's recorded spec unchanged** (`v10_c2_rebuild.json`). The data reproduced exactly
  (2,587 examples, 238,237 tokens), and the final checkpoint's sha256 equals V9-1's record
  (`9c138498...`).
  - So **C2 reproduces bit for bit**, and every V9-1 measurement of it still holds.
  - R0 also measured the new-family probe, which V9-1 had not run: C2 scores 1/24, round 54 0/24.
- **Keeping checkpoints:**
  - `ycode/lm/packckpt.py` stores a fine-tune as the XOR of its bf16 bits with round 54's, split into
    byte planes and zlib-compressed: 50-62% of the size, restored and verified tensor by tensor.
  - With your approval, the four rejected V6 checkpoints were packed (their originals removed after
    verification). Later, R2's and R5's packs were deleted.
  - C2, R1, R6 and R7 are kept as verified packs in `/home/user/lmwork/v10/ckpt`.
- **What the C2 recipe is:** 350 grammar v2 programs x 3 examples, 50% replay from round 54's own
  training distribution, 56 steps, Lion. The learning rate is warmup-stable-decay: 1e-4 for 80% of
  the run, then cosine to 1e-5. It is not a constant high rate.

## 2. Experiments (each from C2, one change at a time)

Produced by `experiments/v10/compare.py`, under identical evaluation conditions. Sets are scored strict:
correct on all 40 inputs. Combined = dev clean + unseen-v1 clean.

| Model | Dev /44 | Unseen-v1 /53 | Combined /97 | family_new /24 | Neighbours /38 | Held-out families /66 (expressible 7 · other 59) | Reproduction c2 /40 · c3 /40 | Held-out combinations c2 /60 · c3 /60 | Bench official · clean · robust |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Round 54 | 27 | 2 | **29** | 0/24 | 4/38 | 2/66 (0/7 · 2/59) | 0/40 · 0/40 | 0/60 · 0/60 | 22 · 19 · 20 |
| C2 | 28 | 3 | **31** | 1/24 | 2/38 | 1/66 (0/7 · 1/59) | 26/40 · 0/40 | 32/60 · 0/60 | 21 · 18 · 20 |
| R1 1301 | 30 | 3 | **33** | 4/24 | 2/38 | 2/66 (1/7 · 1/59) | 36/40 · - | 54/60 · - | 23 · 20 · 23 |
| R2 1301 | 26 | 1 | **27** | 0/24 | 2/38 | 3/66 (2/7 · 1/59) | 1/40 · 25/40 | 0/60 · 30/60 | 25 · 22 · 23 |
| R5 1301 | 25 | 2 | **27** | 1/24 | 2/38 | 0/66 (0/7 · 0/59) | 38/40 · - | 56/60 · - | 27 · 24 · 25 |
| R6 1301 | 26 | 5 | **31** | 3/24 | 1/38 | 2/66 (1/7 · 1/59) | 39/40 · 30/40 | 46/60 · 43/60 | 21 · 18 · 20 |
| R7 1301 | 27 | 2 | **29** | 2/24 | 2/38 | 0/66 (0/7 · 0/59) | 40/40 · - | 55/60 · - | 24 · 21 · 21 |
| R4 1301 (partial) | 28 | 1 | **29** | 3/24 | (1/8 scored) | (not run) | 33/40 · - | 48/60 · - | 25 · 21 · 24 |

- **What each run changed:**
  - R1: 2 new examples of each C2 program.
  - R2: grammar v3 only.
  - R4: R1's recipe at lr 3e-5.
  - R5: R1 plus bug-fix examples.
  - R6: R1 plus grammar v3.
  - R7: R1's recipe at 75% replay.
- **R4 is partial.** A container restart cut it off during its neighbour set, and its RAM-disk model
  was lost before packing. Its selection metrics are complete (`results/runs/R4/1301/PARTIAL.txt`).
- **The benchmark is reported only.**

Forgetting and the pre-registered category gate (`items_lost`: 3 or more lost in one failure
category fails):

| Model | Lost against round 54 (by category) | Gained against round 54 | Lost / gained against C2 | Category gate (>= 3 fails) |
| --- | --- | --- | --- | --- |
| C2 | 5: dev:bugfix:longest_word, dev:bugfix:remove_spaces, dev:write:remove_spaces, unseen:write:count_words_longer_than, unseen:write:keys_above ({'instruction_following': 2, 'reasoning': 2, 'coding': 1}) | 7: dev:bugfix:first_negative, dev:bugfix:index_of_min, dev:bugfix:product, dev:write:first_three_chars, unseen:write:count_capitalized, unseen:write:double_plus_one, unseen:write:sum_at_odd_positions | 0 / 0 | passes |
| R1 1301 | 4: dev:bugfix:longest_word, dev:bugfix:remove_spaces, dev:write:remove_spaces, unseen:write:count_words_longer_than ({'instruction_following': 3, 'hallucination': 1}) | 8: dev:bugfix:first_negative, dev:bugfix:index_of_min, dev:bugfix:product, dev:write:first_negative, dev:write:first_three_chars, dev:write:longest_word, unseen:write:count_capitalized, unseen:write:sum_at_odd_positions | 1 / 3 | fails |
| R2 1301 | 6: dev:bugfix:longest_word, dev:bugfix:remove_spaces, dev:bugfix:sum_of_squares, dev:write:count_greater, unseen:write:count_words_longer_than, unseen:write:keys_above ({'coding': 2, 'instruction_following': 3, 'formatting': 1}) | 4: dev:bugfix:first_negative, dev:bugfix:last_word, dev:write:first_three_chars, unseen:write:digits_of | 7 / 3 | fails |
| R5 1301 | 6: dev:bugfix:count_greater, dev:bugfix:longest_word, dev:bugfix:remove_spaces, dev:write:remove_spaces, dev:write:scale_each_by_three, unseen:write:count_words_longer_than ({'instruction_following': 4, 'reasoning': 2}) | 4: dev:bugfix:first_negative, dev:bugfix:index_of_min, dev:bugfix:product, unseen:write:count_capitalized | 5 / 1 | fails |
| R6 1301 | 5: dev:bugfix:longest_word, dev:bugfix:remove_spaces, dev:write:count_greater, dev:write:remove_spaces, unseen:write:count_words_longer_than ({'coding': 1, 'instruction_following': 3, 'reasoning': 1}) | 7: dev:bugfix:first_negative, dev:bugfix:product, dev:write:first_three_chars, unseen:write:count_capitalized, unseen:write:double_plus_one, unseen:write:is_vowel, unseen:write:sum_at_odd_positions | 2 / 2 | fails |
| R7 1301 | 5: dev:bugfix:longest_word, dev:bugfix:remove_spaces, dev:write:remove_spaces, dev:write:starts_with_the, unseen:write:count_words_longer_than ({'instruction_following': 3, 'reasoning': 2}) | 5: dev:bugfix:index_of_min, dev:bugfix:product, dev:write:first_three_chars, dev:write:longest_word, unseen:write:count_capitalized | 4 / 2 | fails |

R4 (partial) loses the same `remove_spaces` and `longest_word` items.

Cost and identity:

| Run | Start | Trained tokens | Steps | Train min | Total min | Peak RAM MB | Checkpoint sha256 | Packed content sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C2 | round 54 | 229,376 | 56 | 166 | 268 | 11311 | `9c1384982c9b` | `cc2f2fa07777` |
| R1 1301 | C2 | 147,456 | 36 | 74 | 130 | 10827 | `0758684e76d6` | `9909c511043b` |
| R2 1301 | C2 | 147,456 | 36 | 75 | 154 | 10876 | `a0514c173028` | `e0908fe70fe4` |
| R5 1301 | C2 | 233,472 | 57 | 128 | 194 | 10884 | `ffa75c1e54f7` | - |
| R6 1301 | C2 | 270,336 | 66 | 128 | 212 | 10810 | `bc429d345b36` | `0e8bf2a6a881` |
| R7 1301 | C2 | 303,104 | 74 | 196 | 326 | 10941 | `0001c192288a` | `2d75e24a5474` |

- All runs were on this host's 4 cores, with greedy bf16 evaluation.
- R4's training took about 86 minutes; its result record was never written.
- Kept packs: C2, R1, R6 and R7 in `/home/user/lmwork/v10/ckpt`. R2's and R5's packs were deleted with
  your approval. R4's model was lost in the restart.

## 3. What each experiment showed

- **R1, consolidation (more examples of C2's own programs), gives the best selection score.**
  - Combined 33: dev 30/44 and unseen-v1 3/53, against C2's 31 and round 54's 29.
  - The new-family probe goes from 1/24 to 4/24.
  - Reproduction of trained programs goes from 26/40 to 36/40, and held-out combinations of trained
    parts from 32/60 to 54/60.
  - Against C2 it loses 1 item (`double_plus_one`) and gains 3. Training took 74 minutes.
  - It fails the pre-registered category gate. It loses 3 instruction-following dev items that round
    54 passes, the same 3 that C2 loses:
    - `remove_spaces`, written and as a bug fix;
    - the `longest_word` bug fix.
  - One of those 3, round 54's `write:remove_spaces`, is a weak pass: its answer keeps only letters,
    passes the official tests and fails the robust ones. The gate counts official passes and was not
    changed.
- **Cross-family compositions (B, grammar v3) reach held-out families only through combinations of
  trained parts.**
  - R2 trained grammar v3 alone, at 77% of an epoch because of a step-count slip. It solves
    `window_max_2` and `window_min_2`, the only 2 of the 66 held-out-family tasks grammar v3 can
    express, and so goes from round 54's 2/66 to 3/66.
  - R2 also erased C2's grammar v2: C2's programs fell from 26/40 to 1/40, and it often wrote invalid
    Python in grammar v3's shape. Combined score 27.
  - R6 mixed both grammars and learned both: reproduction 39/40 and 30/40, held-out combinations 46/60
    and 43/60. It reached unseen-v1 5/53, the highest of any model. But dev fell to 26 (combined 31),
    and held-out families were 2/66.
  - On the 59 held-out-family tasks neither grammar can express, no run beats round 54's 2/59. R1
    swaps one pass for another: it gains `keep_even_values` and loses `keep_big_values`.
- **Neither verified bug-fix examples (R5) nor a lower learning rate (R4) restores the lost dev items.**
  - R5 still returns the buggy code unchanged on the same 3 of 15 dev bug fixes. Combined 27, though
    its grammar learning is the strongest (38/40, 56/60) and its benchmark 27/65.
  - R4 at 3e-5 keeps most of R1's grammar learning (33/40, 48/60), but combined 29, unseen-v1 1/53.
  - C2 had already lost these items. A continuation from C2 inherits the loss unless the training puts
    the skill back, and neither change did.
- **The neighbour set never improves.** It stays at 1-2/38 in every run, against round 54's 4/38.
- **Contrastive minimal pairs (C, R3) were built and unit-tested, but not run.** R1 had already removed most
  part-swap errors (reproduction 36/40). The remaining failures were forgetting, not confusion between
  parts, so R4, R6 and R7 took precedence.

## 4. Decision

- **The pre-registered rules decide.**
  - A second seed requires combined >= 32 and no failure category losing >= 3 of round 54's items.
    Only R1 reaches 32, and it fails the category part.
  - Replacing round 54 requires the V6 rule: every seed beats 29, the mean gain is >= 3, and the
    category rule holds.
  - So no candidate is validated, and no second seed was run.
- **The checkpoints stay as they are:**
  - Round 54 is unchanged and remains the checkpoint of record.
  - C2 is unchanged (packed, bit-identical to V9-1's).
  - R1 is kept, packed, as the best unvalidated candidate.
- **I am not calling R1 a generalization gain.** Its dev and unseen-v1 gains include grammar-expressible
  items it was not trained on (`sum_at_odd_positions`, `count_capitalized`), but they are one seed, and
  reproduction and held-out combinations are within-grammar measures.
- **Things the rules did not change but you should know:**
  - Every V10 run inherits C2's three losses, because all start from C2.
  - One of the three, round 54's `write:remove_spaces`, passes the official tests but fails the robust
    ones. Scored by robust passes, R1 would pass the category gate. The gate was fixed before the
    results, so it was not changed.

## 5. Exact configuration of the best candidate

**R1** (`experiments/v10/v10_r1.json`, arm R1, seed 1301):

- **Start:** C2. Its checkpoint sha256 is `9c1384982c9b95eb67819fd60b9b5fe5ced0973356dc16199eb3b4c683160d6f`,
  built from round 54 by `experiments/v10/v10_c2_rebuild.json`, identical to V9-1's C2. C2's optimizer
  state no longer exists, so Lion starts fresh (`init_optim` false).
- **Data:**
  - `idioms_select v9_grammar_train`: C2's 350 grammar v2 programs.
  - `idioms_write 2`, no bug-fix examples.
  - `legacy_share 0.5`: replay from round 54's production mix plus original SFT replay rows.
  - Data seed 1302 (`data_seed_offset 1`): 1,767 examples, 159,094 tokens.
- **Training:**
  - 36 steps x 4 rows x 1,024 tokens = 147,456 trained tokens, about one epoch.
  - Lion on the top 24 of 34 blocks, fp32, gradient checkpointing.
  - lr 1e-4 warmup-stable-decay: 2 warmup steps, stable to 80%, cosine to 1e-5.
  - Seed 1301.
  - The loss and masking are V5's, unchanged: prompt tokens weigh 0, answer tokens 1-8.
- **Result:**
  - 73.7 minutes of training, peak RAM 10,827 MB.
  - Final bf16 compact checkpoint sha256 `0758684e76d6b4e98df74a510f90af3174365790e2077418eb25a81b9f433a93`.
- **Kept as** `/home/user/lmwork/v10/ckpt/R1_1301.ypk`, tensor content sha256
  `9909c511043b8a166b4a713e50ad206d84a47cbc927e9ed17bfc9b0e161c6ec3`. Restore with
  `python -m ycode.lm.packckpt unpack /home/user/lmwork/v10/ckpt/R1_1301.ypk <dir>`; this needs round 54
  at `/home/user/lmwork/loop/anchor`.
- **Rerun:** `python -m ycode.lm.experiment experiments/v10/v10_r1.json --arms R1`, with C2 unpacked at
  the spec's `init_from` path.

## 6. Next experiment

**Recommended: a second seed of the whole pipeline, C2 then R1, at seed 1302.**

- **What it costs:** about 3.5 hours of training (C2 recipe 2-2.8 hours, then R1 about 1.3 hours) plus
  about 2.5 hours of evaluation.
- **What it answers:**
  - whether R1's +4 over round 54 replicates on another seed, which is the evidence a replacement needs;
  - whether C2's three losses are specific to seed 1301 (all of V10 inherited them).
- **What would follow:**
  - If seed 1302 reaches combined >= 32 without those losses, the V6 rule can be applied to both
    seeds.
  - If it loses them again, the forgetting belongs to the grammar curriculum itself. The next step is
    then to mix the C2 and R1 curricula with round 54's bug-fix examples from the start, rather than
    trying to repair them afterwards.
- **One operational requirement:** the container restarted twice while idle, killing two runs. Runs of
  this length must be watched actively, as R7 was.

**Not worth repeating:** higher replay (R7), a lower rate (R4) and grammar bug-fix examples (R5) all
leave the lost items lost and score below R1. Grammar v3 alone (R2) erases grammar v2. On cross-family
transfer, every gain so far is a held-out task built from trained parts; widening the parts without
designing them from the test families is the only route the evidence supports.
