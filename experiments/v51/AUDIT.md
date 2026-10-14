# YCode-LM v5 audit (Phase 1)

What the code actually does, traced from the production driver
(`/home/user/lmwork/until_target2.py` → `build_round.py` → `ycode-lm sft` → `average.py` →
`ycode-lm eval` → `ycode-lm analyze`), with every claim checked by reading the code or by running
it. The production scripts are copied in `../v5_finetune/`. Commands to reproduce the checks are
at the end.

**Corrections to the task brief:**
- The model is a **decoder-only** GPT, not an encoder-decoder.
- The best benchmark score so far is **26/65** (rounds 36 and 52). 22/65 is the round-54 anchor
  and the latest ablation baseline.

## 1. Model, tokenizer, pretraining

| | |
| --- | --- |
| Architecture | `ycode/lm/model.py` `GPT`. 34 blocks of [RMSNorm → causal self-attention (RoPE, 32 query / 8 KV heads, QK-norm) → RMSNorm → SwiGLU], tied embedding and LM head. |
| Size | 1,523,202,304 parameters; width 2048; context 1,024 tokens. |
| Tokenizer | Byte-level BPE (`ycode/lm/tokenizer.py`), 8,192 tokens: 7,932 merges plus 4 special tokens (`<\|endoftext\|>`, `<\|user\|>`, `<\|assistant\|>`, `<\|end\|>`). Trained on code, so common English words split into several tokens (`largest`, `vowels`, `Fahrenheit`). |
| Pretraining | Next-token prediction on 53.3M tokens of Python (stdlib and site-packages) for v1 to v4. v5 was grown from v4 (width ×4) and given only about 2.5M tokens of continued pretraining (600 steps × 4,096 tokens). That is about 0.002 tokens per parameter, so the base model is heavily under-trained. |

## 2. Fine-tuning path

| | |
| --- | --- |
| Data | `build_round.py`: `basics_examples` (6 writes + 4 bug fixes per concept) + `compose_examples` (3 per exercise) + `compose_bugfix_examples` + `contrast_examples` + `novel_examples`, then `write_basics_dataset` with 200 replay rows from the original stdlib SFT set. 4,458 generated examples per round without skills, 4,684 with. |
| Encoding | `encode_example(weighted=True)`. Prompt tokens weigh 0. Answer tokens weigh 1, or 2/3/8 for the def line, the changed line and the changed tokens of a bug fix, and 3 for a written function's body. The answer's end marker weighs 1. |
| Batches | `SFTData`, packed: each 1,025-token row holds several whole examples back to back. Training order is a seeded permutation, without replacement within an epoch. The first 5% of the shuffled rows are a validation split. |
| Optimizer | Lion. lr 1.5e-5 (min lr 1.5e-6), warmup 2 steps, warmup-stable-decay schedule (cosine over the last 20%). Updates are applied inside backward, so there is no gradient clipping and no gradient-norm measurement. Momentum is carried over from the anchor (`--init-optim`). |
| Budget | 72 steps × batch 4 × 1,024 tokens per round (288 packed rows ≈ 295k tokens ≈ 3,700 examples of about 80 tokens, i.e. most of a round's data once). Only the top 24 of 34 blocks are trained; embeddings and the bottom 10 blocks stay frozen. bf16 autocast, gradient checkpointing. |
| Checkpoints | `compact_state`: frozen tensors stored in bf16, trained tensors in fp32. From round 56, each round is averaged 50/50 with its anchor (`average.py`, fp32 tensors only). Saves are mid-round and in place. |
| Acceptance | A round is accepted if its dev score is ≥ anchor − 3 and ≥ best − 5, or if new-concept plus bug-fix dev items improve by 2 or more. The benchmark is run only for accepted rounds and is never used to accept. |

## 3. Skill functions

`ycode/lm/skills.py` defines 49 skills (50 task phrasings). They enter training through
`compose._skills()` → `_raw_exercises()` → `all_exercises()`, and from there into
`compose_examples`, `compose_bugfix_examples` and `contrast_examples`. That adds 226 examples per
round. The switch is the module flag `compose.INCLUDE_SKILLS`, which defaults to **True**, and
`build_round.py` never sets it. **Skills have therefore been in every production round since they
were added.** The dev set builder reads `_raw_exercises(seed=0)` only for its own held-out keys.

## 4. Confirmed problems

| # | Finding | Evidence |
| --- | --- | --- |
| C1 | **Benchmark and dev items leak into training through behavioural equivalents.** Name and concept filters (`held_out_names`, `HELD_OUT_KEYS`, `DEV_KEYS`) cannot see a training function that computes a held-out item under another name or wording: "sum of the nonzero numbers" (= `total`), "less than zero" (= `is_negative`), `scale_by_seven` (= `weeks_to_days`), `n >= 7` (= `above_six`), `len(items)` (= `dict_size`). 15 held-out items have exact equivalents in every round's data: benchmark `total`, `sum_of_cubes`, `dict_size`; dev `cube`, `is_negative`, `double_all`, `smaller`, `count_uppercase`, `weeks_to_days`, `multiply_three_numbers`, `longer_than_three`, `smallest_positive`, `below_two`, `above_six`, `keep_gt5`. That is 129 rows in round 55's data (seed 1055), the same with skills on or off. A 16th item, dev `keep_negative`, leaked in other rounds: the novel condition pair "less than 3 ... also negative" was drawn in 23 of the 56 rounds, and "greater than 5 ... also positive" (= `keep_gt5`) in 25. | `leakage_audit.py`, `ycode/lm/leakage.py`. Each flagged function agrees with the item's reference solution on all 120 probe inputs: the test's own arguments, typed variations and seeded random values. |
| C2 | **Contamination inflates dev and benchmark scores.** 17 of the 62 dev slots (18 counting `keep_negative`) and 4 of the 65 benchmark slots are contaminated. In abl2/abl3 they supply 9 to 11 dev points (of 34 to 38) and 2 to 4 benchmark points (of 17 to 25). Since rounds are accepted on dev, every accept/reject decision included these items. | Pass lists in `../v5_finetune/ablations/abl2/*.json` and `abl3/*.json`. |
| C3 | **Some held-out tests are too weak to catch wrong code.** `fibonacci` is checked only at 0, 1 and 10, so `n * (n + 1) // 2` passes. `second_largest` has a single assertion. Round 52's 26/65 includes 2 wrong functions counted as passes (`fibonacci` above, and `first_word` returning `text.split()[-2]`). Round 54's 22/65 includes 2 (that `first_word`, and a `maximum` fix starting from `items[1]`). The scoring rules are **not** changed. A robust sub-score is reported next to the official one instead. | `leakage.agreement` on the passing answers in `loop/bench52.jsonl` and `loop/bench54.jsonl`. |
| C4 | **11 to 12% of each round's rows are exact duplicates** (510 of 4,458 for seed 1055; 562 for seed 1101). | `datacheck.find_duplicates`. |
| C5 | **No validation loss is ever measured.** All rounds and ablations run with `--eval-iters 0`. The 5% validation split is held out of training but never evaluated, and it is drawn from the same templates as training, so it would measure fit, not generalisation. | `until_target2.py`, `ablate.py`, `train.py`. |
| C6 | **The logged training loss is the weighted loss of one 4-row batch.** It swings between steps (0.077, 0.177, 0.181 at steps 50/60/70) and between data mixes (0.001 to 0.18 at step 70 across abl2 arms), and it cannot be compared across mixes whose answer weights differ. **My earlier "the model has memorised its data" conclusion (`ABL2_RESULT.md`) is not supported without validation evidence, and I withdraw it.** | `train.py` loop. |
| C7 | **Training has no numerical health checks**: no NaN/Inf check on the loss, no gradient norm, no check for invalid batches, and no checksum of checkpoints written or loaded. | `train.py`. |
| C8 | **Skills are on by default in production** (section 3). In abl2's single-seed arm, adding them cost 5 benchmark points and 1 dev point, and raised dev hallucination failures from 1 to 4. That is one seed and inside dev noise, so it is a hypothesis, not proven harm. | `abl2/data_skills.json`. |

## 5. Verified correct

- **Causal attention:** changing tokens after position 25 leaves logits at positions 0 to 24
  bit-identical (max difference 0.0). The KV cache matches a full recompute (existing tests).
- **Target alignment and loss masking:** `x = t[:-1]`, `y = t[1:]`, `mask = m[1:]`. On the real
  tokenizer, no weighted target is prompt text, no padding token is weighted, and the weights used
  are {0, 1, 2, 3, 8}. In unpacked mode, padding positions carry weight 0.
- **Learning-rate schedule:** the logged lr matches the formula (1.468e-5 at step 60, 2.89e-6 at
  step 70 with min lr 1.5e-6).
- **The evaluation is deterministic:** three runs gave identical answers on every item (Experiment
  A). It decodes greedily, with at most 200 new tokens and `no_repeat_ngram` 12, and runs tests in
  an isolated subprocess with a 5 s timeout. Dev and benchmark points from `eval` and from
  `analyze` agree.
- **Round 53's rejection:** dev 36 < best 40 − 3 (the `tol_max` in force then). Re-scoring its
  saved answers reproduces all 62 verdicts. Its weights cannot be regenerated: its anchor (round
  52) was overwritten.
- **No true contradictions:** in seeds 1055 and 1101, with or without skills, no task text is
  answered by functions that compute different things. The "94 tasks with two answer bodies" were
  loop and one-liner renderings that agree wherever both are defined; one raises on an empty list
  where the other returns `None`. The abl2 `data_dedup` arm therefore removed legitimate
  variations, not contradictions.
- **The replay pool has no behavioural equivalents of held-out items** in round 55's sample (134
  replayed functions, each run in a subprocess).

## 6. Likely causes of the plateau (evidence-weighted, not proven)

1. **Selection on a noisy, contaminated dev set.** Seed noise is 3 dev points, and 17 dev slots
   test items the model saw in equivalent form. Accept/reject decisions and the anchor's 39 are
   partly noise. No fresh run of the recipe reaches 39 again.
2. **Narrow data.** 78% of answers are one-liners, with 844 distinct bodies. Held-out errors are
   dominated by writing a neighbouring trained function's body (the error analysis). No data-mix
   change in abl2 or abl3 moved this beyond noise.
3. **An under-trained base model** (section 1). Reasoning and factual errors (about a third of
   lost points) have not responded to fine-tuning in 50 rounds.

## 7. Unverified hypotheses

- H1. Packing with plain causal attention, with no document mask, lets later examples in a row
  attend to earlier unrelated ones. This may encourage copying from context.
- H2. Freezing the embeddings and bottom 10 blocks limits learning of lexical distinctions
  (`fourth` vs `fifth`).
- H3. Skills hurt (C8). This needs several seeds.
- H4. Low training loss reflects saturation on the training distribution. This needs training
  loss vs validation loss vs unseen-task loss from the same checkpoint (Phase 4).

## Reproduce

```
python3 experiments/v51/leakage_audit.py 1055 [--no-skills]   # C1, replay check
python3 -m pytest tests/test_v51.py                            # masking, alignment, leakage, hygiene tests
```
