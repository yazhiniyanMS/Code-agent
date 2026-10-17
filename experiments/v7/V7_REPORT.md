# YCode V7: why the model writes a familiar function instead of the requested one

Status: **complete**. Every number is measured, on development tasks only. The fixed holdout
(unseen-v2) was not read, analysed or used to choose anything in V7. The benchmark was not used
either. Round 54 is unchanged (sha256 `2538b55d...`).

## Answer

- **The mechanism is retrieval by wording, of functions trained hundreds of times.** When the model
  has not learned a task's mapping, it writes the heavily trained v5 function whose description
  shares the most words with the prompt. The word that differs is ignored (`fourth_item` gets
  `fifth_item`). Training frequency plays no part. This is 38 of round 54's 68 clean dev + unseen-v1
  failures.
- **Prompt fixes do not work.** Input-output examples or behavioural constraints in the prompt help
  no task. Describing a neighbouring function makes the model write it (17 of 38 answers).
- **The targeted intervention fails its screen.** P1 trains on sibling pairs with the deciding
  tokens weighted. It moves likelihood but flips no answer: neighbour set 2/38 against round 54's
  4/38.
- **The real bottleneck is learning strength.** At the V6 recipe (Lion, learning rate 1.5e-5), no
  run learns even its own training programs to generation level. Code loss stays near 0.4-0.5
  nats/token, and the trained programs are written correctly 0/30 times. At learning rate 1e-4, the
  same 24 steps and data (L2) write **29/30** of them.
- **Once a family is learned, generalization within it is real.** L2 writes **13/22** constants of
  its families that no run trained on, correct on all 40 inputs each. Round 54, D, P1 and L1 write
  0/22.
- **It does not cross families.** No model's learned families improve the neighbour set (2-4/38) or
  the held-out sets. Those concepts are, by design, skeleton-disjoint from every training family.
- **Decision: no candidate replaces round 54.** P1 failed its pre-registered screen. L1 and L2 are
  diagnostics; L2 also forgets (dev 15/44 against 27).

## 1. Failure mechanism (round 54, clean dev /44 + unseen-v1 /53)

`experiments/v7/mechanism.py` sorts every failure into the five V7 classes, derived from the v5.1
classifier's evidence (rules in the script's docstring).

| Class | Round 54 | V6 C | V6 D |
| --- | --- | --- | --- |
| Wrong function retrieved (a trained function's body or behaviour) | **38** | 39 | 28 |
| Correct concept, incorrect implementation | 13 | 15 | 26 |
| Prompt misunderstood (another coherent task, not a trained one) | 11 | 16 | 10 |
| Edge cases (>= 75% agreement with the reference) | 4 | 2 | 2 |
| Syntax or output format | 2 | 1 | 1 |
| Passed | 29/97 | 24/97 | 30/97 |

The catalogue of trained functions is the v5 mix, so D's retrievals of its own curriculum programs
count as "wrong implementation". Part of D's shift may therefore be counting.

**Which trained function is retrieved?** For the 35 write items where round 54 wrote a trained
function, each feature ranks that function among the trained functions of the same arity (up to
~725; 1.0: the most similar; 0.5: chance):

| Feature | Mean percentile | Retrieved function is the top match |
| --- | --- | --- |
| Shared content words of the prompt and the function's training description | **0.90** | 16/35 |
| Code similarity to the task's reference | 0.81 | 9/35 |
| Shared words of the function names | 0.80 | 15/35 |
| Examples of the function per training round | 0.49 | 1/35 |

Typical pairs: `index_of_min` gets `index_of_max`, `longest_word` gets `shortest_word`,
`hours_to_seconds` gets `hours_to_minutes`, `double_plus_one` gets `double`.

Among the failures, a close training neighbour makes retrieval the more likely failure: AUC 0.65 by
wording, 0.67 by code, 0.53 by name, and 0.59 by the closest-worded neighbour's training frequency.
Items whose neighbour is a near-twin mostly pass (wording AUC 0.18, against passed items).

## 2. Controlled prompt diagnostic (`ycode/lm/neighbours_v1.json`, 38 dev tasks)

The set has two balanced groups:
- **Same wording (19):** worded like a trained function, with one detail that changes the code
  (`first_four_chars` next to the trained `first_five_chars`).
- **New wording (19):** a trained function's behaviour under a new name and new wording.

Every task is verified by execution. Tasks matching a benchmark or unseen-v2 item by name, behaviour
or skeleton were dropped (25 of 71 drafts; counts only, no holdout item was seen). There are four
prompt variants per task. Decoding uses the evaluation's own settings (greedy, 200 tokens, bf16).

| Prompt | Round 54: same / new wording | Writes the neighbour | V6 D: same / new | Writes the neighbour |
| --- | --- | --- | --- | --- |
| Original | 1 / 3 | 9 | 1 / 1 | 10 |
| + two input-output examples | 0 / 0 | 6 | 0 / 1 | 7 |
| + one behavioural constraint | 0 / 1 | 6 | 1 / 0 | 6 |
| + a neighbouring function's description | 0 / 1 | **17** | 0 / 2 | 15 |

- **More specification does not help.** Examples and constraints never gain a task.
- **Naming a neighbour pulls the model to it.** The model's preference for the reference over the
  neighbour's code falls on 27 of 38 tasks (p = 0.014). On new wording it falls on 17 of 19 tasks
  for both models (p = 0.0007).
- **The likelihood gap is large.** On same-wording tasks round 54 prefers the neighbour's code by
  15.9 nats on average and prefers the reference on only 4 of 19 tasks.
- **New wording fails differently.** Round 54 slightly prefers the reference over the distractor
  (+2.6 nats), yet writes neither. It cannot map new words to code it knows.

## 3. Training runs (from round 54; seed 1301; Lion; top 24 layers; 4 x 1024-token batches)

The two control columns are scored on all 40 program inputs per task:
- **Trained programs:** the 36 programs of the 12 constant-parameterised curriculum families
  (30 scored).
- **New constants:** constants of the same families that no run trained on.

| Run | Variable | Steps x LR | Dev clean /44 | Unseen-v1 /53 | Neighbours /38 (same-wording margin) | Trained programs /30 | New constants /22 | Train min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Round 54 | - | - | 27 | 2 | 4 (-15.9) | 0 | 0 | - |
| **P1** | sibling pairs, deciding tokens weigh 20 | 24 x 1.5e-5 | 23 | 0 | 2 (-8.9) | (0/17 other curriculum programs) | 0 | 48.6 |
| L1 | 36 programs, ~24 exposures each | 24 x 1.5e-5 | 21 | 2 | 4 (-9.3) | 0 | 0 | 50.3 |
| **L2** | L1 at learning rate 1e-4 | 24 x 1e-4 | 15 | 1 | 2 (-10.2) | **29** | **13** | 47.0 |
| V6 D | curriculum v1 (existing model) | 72 x 1.5e-5 | 28 | 2 | 2 (-9.2) | 0 | 0 | 190 |
| V6 G | D at twice the learning rate (existing model) | 72 x 3e-5 | 26 | 2 | 4 (-9.9) | 10 | 6 | 167 |

- **P1, the targeted intervention, fails its pre-registered screen.** It needed to beat round 54 on
  the neighbour set's original prompts and got 2/38 against 4/38, so P0 and the second seed were not
  run. Weighting the 961 deciding tokens (3.3% of answer tokens) moved the same-wording margin in 24
  steps as far as D's curriculum did in 72 (-15.9 to -8.9 nats), but no answer flipped.
- **Under the V6 recipe nothing new is learned to generation level.** L1 saw each program about 24
  times, under several names and templates, the evaluation prompt included. Code loss on the references stayed at 0.49, and it
  still wrote `return sum(numbers) - sum(numbers)` for `sum_after_first_two`. Round 54's v5 functions
  were seen hundreds of times, which is why every new task falls back to them by wording. V6's
  likelihood gains without pass-rate gains had this cause.
- **Learning strength is the lever.** L1 against L2 is controlled: same data and seed, learning rate
  only. The trained programs go from 0/30 to 29/30, and code loss from 0.49 to 0.07. V6 D against G
  (72 steps, 1.5e-5 against 3e-5) gives the same direction: 0/30 against 10/30.
- **Learned families generalize to new members.** L2 writes 13/22 never-trained constants correctly
  on all 40 inputs, against 0/22 for round 54 and L1. Examples: `shift_by_5` and `shift_by_6` from
  trained k = 1, 3, 13; `digits_above_2` and `digits_above_6` from k = 4, 5, 7.
- **Learned families do not reach other families.** Every model stays at 2-4/38 on the neighbour
  set, and L2 loses round 54's new-wording passes. The held-out dev and unseen-v1 concepts share no
  skeleton with any training family by design, and they do not improve. L2 forgets (dev 27 to 15)
  because it has almost no legacy data at a high learning rate.

## 4. Decision

The V6 rule is unchanged. A candidate replaces round 54 only if:
- every seed beats all_clean 29/168,
- the mean gain is >= 3 items,
- no failure category loses >= 3 items that round 54 passes.

**No candidate qualifies, and round 54 stays.**
- **P1** failed its screen (development tasks), so it never reached the holdout evaluation.
- **L1 and L2** were diagnostics, declared as such before running. L2's dev + unseen-v1 total is
  16, against round 54's 29.
- **The fixed holdout** was not evaluated in V7, because no candidate passed screening.

### What the evidence rules out

| Ruled out | Evidence |
| --- | --- |
| Frequency-driven retrieval | Frequency percentile 0.49 |
| Prompt-side fixes | I/O examples, constraints: 0 tasks gained, round 54 and D |
| Focusing the loss on deciding tokens, at the V6 learning rate | P1: margin moves, 0 answers flip, dev and unseen-v1 lose 6 |
| More of the V6 kind of data at the V6 learning rate (depth, formats, breadth) | V6 C-G; and now L1: not even trained programs are learned |
| "The model cannot generalize at all" | L2: 13/22 never-trained constants of learned families |

### The remaining bottleneck and the next test

Generalization is local to learned families. The held-out concepts share no training family, and
under the V6 recipe no family is learned well enough to be written.

The most informative next test combines L2's learning strength with breadth and forgetting control.
Learn many families to generation level while keeping round 54's skills, then ask whether transfer
to new families appears (the neighbour set, then dev and unseen-v1). V6 F tested breadth only under
the recipe that learns nothing, so this question is still open. Estimated cost, from the measured
36 tokens/s:

- **Data:** 100-150 curriculum programs at ~10 exposures each, plus half legacy and replay to hold
  dev, at learning rate 5e-5 to 1e-4. That is about 250k tokens, 72 steps, ~1.9 h of training per
  seed, plus ~25 min of development evaluation.
- **Screening:** one seed on the neighbour set and dev + unseen-v1, about 2.3 h.
- **Adoption:** a second seed and the full evaluation (holdout included) only if screening shows
  cross-family gains with dev held. That adds about 3 h.

A learning-rate sweep without breadth is not worth running: L1 against L2 already settles that
learning strength is the lever.

## Resources and artefacts

- **Training** (4 CPU cores, fp32): 36-37 tokens/s, 98,304 tokens per run, 47-50 min, peak RAM
  11.1-11.3 GB.
- **Evaluation:** bf16, ~9 min dev, ~8 min unseen-v1, 3-7 min per diagnostic set, peak about 6.7 GB.
- **Disk:**
  - The 3.7 GB free fit no ordinary 6.4 GB training checkpoint, so V7 added `train.final_bf16`. The
    last save writes the trained tensors in bf16, the precision evaluation loads, with no optimizer
    state. That is 2.1 GB, and it is tested to load the same weights as the fp32 checkpoint cast to
    bf16. A run interrupted before its last step restarts.
  - P1's and L1's model files were removed after their records were archived (`REMOVED.txt` next to
    each says how to rebuild).
  - L2's model is kept at `/home/user/lmwork/v7/runs/L2/1301/model_bf16` (sha256 `fdf3d162...`,
    references round 54).
  - The V6 checkpoints are untouched.

## Reproduction

```bash
python experiments/v7/train_frequency.py train_frequency.json                     # bodies per round
python experiments/v7/mechanism.py train_frequency.json out.json round54=DEV.jsonl,UNSEEN.jsonl
python experiments/v7/neighbors.py                                                 # rebuilds ycode/lm/neighbours_v1.json
python experiments/v7/prompt_diagnostic.py MODEL_DIR out.jsonl [--set SET.json] [--variants original,...]
python experiments/v7/diag_summary.py summary.json out.jsonl ...
python experiments/v7/family_new_set.py SET.json [--trained | --k-families]       # control sets (in experiments/v7)
python -m ycode.lm.experiment experiments/v7/v7_pairs.json --arms P1
python -m ycode.lm.experiment experiments/v7/v7_learnability.json --arms L1        # or --arms L2
```

Records are in `experiments/v7/results/`:
- the mechanism analysis;
- every diagnostic row, for every model and set;
- run records (scores, answers, manifests, step metrics).
