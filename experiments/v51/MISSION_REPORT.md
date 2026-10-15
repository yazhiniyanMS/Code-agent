# YCode-LM v5.1 mission: reliable evaluation and unseen-concept generalisation

Measured results are marked **measured**; hypotheses say so. Nothing below comes from a run that
did not happen. Raw data is in `results/`.

## 1. Evaluation audit (phase 1)

**Measured.** All checks are in `tests/test_v51.py` (39 tests, all passing).

| Question | Finding |
| --- | --- |
| Are the sets defined and versioned? | Yes, now (`ycode/lm/evalsets.py`, `eval-v5.1-1`). Benchmark: 65 items, 61 clean. Dev: 62 items, 44 clean. Unseen: 53 items, all clean. Each set is SHA-256-fingerprinted over names, prompts, official tests and robust tests. "Clean" removes the 16 items v5's checkpoints saw in equivalent form. |
| Training overlap with held-out items? | No generator emits a function named like a held-out item, or one computing the same thing on every probe input, apart from 8 reviewed shared building blocks (`len`, `7 * x`, a three-way product, string length). A test enforces this. The unseen set started at 82 candidates. 29 were dropped: 17 because a trained function shares the name, 12 because a trained function or benchmark/dev item computes the same thing (`halfway` = `mean_of_two`, `strip_vowels` = `remove_vowels`, `position_of_max` = `index_of_max`). |
| Are the three loss measurements consistent? | Yes. Same encoder (`encode_example` for tasks, the stored encoding for data rows), the same target shift, every answer token counted once, and the same token-weighted mean with a bootstrap 95% interval. |
| Why was held-out loss (0.035) below training loss (0.094)? | **It was not a train vs held-out comparison.** The round-54 checkpoint never trained on that data (built for seed 1201), so both splits were unseen rows from the same generators. With 200 rows each, the intervals overlap: 0.107 [0.052, 0.176] vs 0.074 [0.028, 0.129]. Per-row loss is heavy-tailed (per-row mean 0.06, token-weighted mean 0.11) and depends on the row mix: written functions cost 0.157 vs 0.019 for bug fixes. The diagnostic now records `trained_on`, a per-row-type breakdown and intervals. |
| Do weak tests accept wrong code? | They did, and they still do for the official score, which is unchanged. `robust_tests_v1` adds independent assertions for every item, computed from the reference inside the official tests' domain and scale, plus edge cases (`first_word`, `fibonacci`, `maximum`, `minimum`, `second_largest`, `is_between`). Tested: 7 known-wrong historical passes fail it; 5 alternative correct solutions pass. |
| Denominators | Every score is reported as official, clean and robust, each with its denominator. Historical official scores are unchanged. |

## 2. Unseen-task failures of the round-54 checkpoint (phase 2)

**Measured, no training.** `results/m2/m2_failures.json` holds every failure: prompt, expected
solution, generated answer, test outcome, group, class and evidence. Ranking:
`results/m2/m2_failures.md`.

- **Clean dev: 27/44. Unseen concepts: 2/53** (robust 1/53). The dev set, whose items are one word
  away from trained concepts, hugely overstates generalisation. Genuinely new concepts are almost
  never solved.
- **Classes of the 68 clean failures:**

| Class | Clean dev | Unseen | Share |
| --- | --- | --- | --- |
| Incorrect task interpretation | 12 | 33 | **66%** |
| Algorithmic or logic error | 3 | 9 | 18% |
| Incorrect code generation | 0 | 5 | 7% |
| Edge-case failure | 2 | 2 | 6% |
| Formatting or parsing | 0 | 2 | 3% |
| Evaluation failure | 0 | 0 | 0% |

- Nearly every task-interpretation failure is the **body of the nearest trained function**:
  `hours_to_seconds` → `hours_to_minutes`, `middle_of_three` → `min_of_three`, `swap_halves` →
  `swap_case`, `fourth_item` → `fifth_item` (`items[4]`), `first_three_chars` → `s[:2]`. The model
  retrieves by surface similarity and ignores the word that differs.

## 3. Controlled experiment (phase 3)

`m3_spec_first.json`, run with `python -m ycode.lm.experiment experiments/v51/m3_spec_first.json`.

| | Control A | Treatment B |
| --- | --- | --- |
| Data | Current clean pipeline: leak-free, skills off, 200 replay rows; written functions answered as bare code. | Identical examples and code. Each written function first restates the task from the prompt in a docstring (`spec_first`). |
| Everything else | Same round-54 checkpoint, token budget, optimizer (Lion, lr 1.5e-5, warmup-stable-decay schedule), trained layers (top 24), evaluation (greedy, unchanged harness), matched seeds. | Same |

- **Why this intervention:** two-thirds of clean failures are the nearest trained body with the
  distinguishing word ignored. Restating the task makes the model copy that word ("seconds",
  "middle", "fourth") into its answer just before writing the code. It changes one thing (the
  answer format) and adds no new concepts, so it cannot leak a held-out item. The data-mix
  changes tried in abl2 and abl3 (more constants, contrast bug fixes, dedup, replay) did not
  move unseen performance.
- **Hardware constraint (measured):** after a container restart this host has no AMX or
  AVX512-BF16. Emulated bf16 training ran at 11 tokens/s. fp32 is 3× faster here (block forward
  and backward 3.61 s vs 11.07 s), but still 2.9× slower than the AMX host v5 trained on, so a
  production 72-step run would take about 3.3 hours. To fit matched seed pairs into the mission,
  **both arms use 24 steps (a third of the production budget), fp32 training and 2 seeds each.
  These results are preliminary and not directly comparable with 72-step runs.** Generation stays
  bf16: same answers as fp32 on a sample, at similar speed.
- **Smoke tests:** treatment B (bf16, 2 steps) passed every stage, including the unseen
  evaluation, scoring with denominators, failure classes and the diagnostic. Control A in fp32
  measured 29–32 tokens/s, about 145 s per step, peak memory 11.3 GB.
- **Incident:** the first launch of A1301 died at its step-12 checkpoint save because the disk was
  full. I had stopped the fp32 smoke runner by PID before it deleted its 7 GB checkpoint. The
  leftover was removed, and the runner now refuses to train with under 8 GB free. A1301 restarted
  from scratch at 04:32. The same seed and data reproduce the same losses (step-1 loss 0.09438 in
  both attempts). Its first attempt also shared the CPU with the test suite for 15 minutes; that
  attempt is discarded.

## 4. Results (phase 4)

**Measured, preliminary** (24 steps, fp32, 2 matched seeds per arm; see section 3). Full tables:
`results/m4/m4_report.md`, generated by `python experiments/v51/m4_report.py
experiments/v51/m3_spec_first.json`. Every run's scores, answers, data manifest, hygiene
record, diagnostic and per-step metrics: `results/m4/runs/<arm>/<seed>/`.

### 4.1 Per run

Primary = clean dev + unseen, out of 97. Hallucinations and invalid outputs are summed over dev,
unseen and benchmark (v5 error categories: code that calls an unknown name, and answers that do not parse or define no function).

| Arm | Seed | Primary /97 | Dev clean /44 | Unseen /53 | Unseen robust /53 | Dev /62 (robust /62) | Bench /65 | Bench clean /61 | Bench robust /65 | Halluc. | Invalid | Train / total min | Peak RSS train / eval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 1301 | 26 | 25 | 1 | 0 | 37 (37) | 25 | 22 | 24 | 7 | 5 | 53 / 88 | 11.3 / 7.7 GB |
| A | 1302 | 27 | 26 | 1 | 0 | 38 (38) | 23 | 20 | 22 | 7 | 4 | 57 / 98 | 11.3 / 7.7 GB |
| B | 1301 | 29 | 27 | 2 | 1 | 39 (38) | 24 | 22 | 23 | 7 | 8 | 61 / 105 | 11.3 / 7.7 GB |
| B | 1302 | 28 | 26 | 2 | 1 | 38 (37) | 21 | 19 | 20 | 9 | 7 | 57 / 106 | 11.3 / 7.7 GB |

Every run trained on 98,304 tokens (24 steps x 4 x 1024), with no empty batches, a finite loss
and gradient norm at every step (largest gradient norm 0.20), and finite weights after training
(checked by the runner). B's data is 12% longer (the docstrings), so in the same token budget B sees
about 11% fewer examples; that is part of the treatment as specified (same token budget).

Answer-token loss (diagnostic: 16 rows per data split, every clean dev and unseen item; bootstrap
95% intervals in `diagnose.json`):

| Arm | Seed | Train rows | Validation rows | Unseen: dev clean | Unseen: new concepts | Training loss, last 10 steps |
| --- | --- | --- | --- | --- | --- | --- |
| A | 1301 | 0.329 | 0.009 | 0.233 | 0.977 | 0.112 |
| A | 1302 | 0.196 | 0.009 | 0.219 | 0.939 | 0.081 |
| B | 1301 | 0.290 | 0.011 | 0.622 | 1.364 | 0.121 |
| B | 1302 | 0.171 | 0.014 | 0.600 | 1.345 | 0.100 |

B's higher unseen loss is mostly the format: the references are bare code and B expects a
docstring first. The format-matched, code-only loss (recorded from the seed-1302 runs on) is
in section 4.4.

### 4.2 Per arm and the pre-registered comparison

| Arm | Primary /97 | Dev clean /44 | Unseen /53 | Unseen robust /53 | Bench /65 | Bench robust /65 | Halluc. | Invalid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A (control) | 26.5 ± 0.7 | 25.5 ± 0.7 | 1.0 ± 0.0 | 0.0 ± 0.0 | 24.0 ± 1.4 | 23.0 ± 1.4 | 7.0 ± 0.0 | 4.5 ± 0.7 |
| B (spec-first) | 28.5 ± 0.7 | 26.5 ± 0.7 | 2.0 ± 0.0 | 1.0 ± 0.0 | 22.5 ± 2.1 | 21.5 ± 2.1 | 8.0 ± 1.4 | 7.5 ± 0.7 |

(mean ± standard deviation over the 2 seeds)

- **Paired differences B − A on the primary metric: +3 (seed 1301), +1 (seed 1302). Mean +2.0,
  95% CI [−10.7, +14.7]** (t interval with 1 degree of freedom).
- **Pre-registered verdict: rejected.** Retaining B needed a mean above the one-sided margin
  (6.31 with two seeds) and no failure category worse by 2 or more points. B's mean (+2.0) is
  below the margin, and the coding category is worse by 4.5 points.
- **What the category rule caught.** The rule counts failures per category within each run.
  Every item behind B's 4.5 extra coding failures is one that **both** arms fail: A returns a wrong
  value (counted as reasoning, which falls by 9.0), while B's wrong code raises an exception:
  `running_max([])` calls `max` on an empty list; `repeat_word('hi', 3)` multiplies by a string;
  `insert_at_front` indexes with a string. So the rule fired on a change in failure type, not on
  lost items. That is a weakness of the rule, recorded here and to be fixed **before** the next
  experiment (count, per category, the items the control passes and the treatment fails). It
  does not change the outcome: on the primary metric alone the verdict is inconclusive, not
  retained.
- **Benchmark (reported, never used for decisions): B 22.5 vs A 24.0 of 65**, robust 21.5 vs
  23.0. Invalid outputs rise from 4.5 to 7.5 per run: in each B run, 4 of them are an unterminated
  docstring (dev `smaller`, `count_greater`, `first_negative`; benchmark `count_vowels`).

### 4.3 What changed, item by item (post hoc)

Not part of the decision rule. Per seed, the clean items B passes and A fails, and the reverse:

| Seed | B passes, A fails | A passes, B fails |
| --- | --- | --- |
| 1301 | dev bug fix `longest_word`, `remove_spaces`; dev write `count_odd`, `remove_spaces`; unseen `keys_above` | dev bug fix `count_greater`; dev write `index_of_min` |
| 1302 | dev bug fix `longest_word`, `remove_spaces`; dev write `remove_spaces`; unseen `keys_above` | dev bug fix `count_greater`, `index_of_min`; dev write `index_of_min` |

- The **same four gains and the same two losses appear in both seeds**. Between the two seeds
  of one arm, only one item flips (A: `count_odd`; B: bug fix `index_of_min`). So the change is
  caused by the treatment and reproducible, not seed noise, but it is small: about +2 items of 97.
- Pooled over seeds, 9 gains vs 5 losses: exact two-sided p = 0.42 (and the pooled pairs are not
  independent: the same items recur). Not evidence of an improvement.

### 4.4 Why restating the task did not fix task interpretation

- **B restates the task correctly, then writes the wrong code anyway.** In B1301, 51 of 53
  unseen answers begin with a docstring that matches the restated prompt exactly; 49 of those 51
  fail. Example: `double_plus_one` is restated correctly, and the body is the neighbour's
  `return 2 * x`. Failure classes after training are unchanged: task interpretation is 30–32 of
  the ~51 unseen failures in every run of both arms.
- **The restated task helps any model when it is given, but B did not learn to use it better.**
  Code-only loss on the reference solutions (tokens of the function body only, seed 1302),
  with each reference rendered bare or with the restating docstring:

| Model | Unseen new: bare reference | Unseen new: restated reference | Dev clean: bare | Dev clean: restated |
| --- | --- | --- | --- | --- |
| A1302 (control) | 1.145 | **0.906** | 0.276 | **0.214** |
| B1302 (spec-first) | 1.641 | 0.998 | 0.756 | 0.244 |

  With the task restated just before the code, the control predicts the correct code better
  than B does (0.906 vs 0.998). Training on this format taught B to write the restatement, not
  to condition its code on it.
- **Interpretation (hypothesis, consistent with the above).** The model retrieves code from the
  nearest trained concept, keyed on the function name and the surface of the prompt. Saying
  the distinguishing word again does not change the key. A fix has to make the distinguishing
  word decide the code during training, for example near-twin concepts trained side by side.

### 4.5 Against the starting checkpoint

The round-54 checkpoint was re-evaluated on this host with the same harness. Its answers are
identical to those from the AMX host on 62 of 62 dev items and 64 of 65 benchmark items, and its
scores are identical too. The answers are in `results/m4/anchor/`.

| | Primary /97 | Dev clean /44 | Unseen /53 | Unseen robust /53 | Dev /62 (robust) | Bench /65 | Bench clean /61 | Bench robust /65 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Round 54 (start) | **29** | **27** | **2** | 1 | 39 (38) | 22 | 19 | 20 |
| A, mean of 2 | 26.5 | 25.5 | 1.0 | 0.0 | 37.5 (37.5) | 24.0 | 21.0 | 23.0 |
| B, mean of 2 | 28.5 | 26.5 | 2.0 | 1.0 | 38.5 (37.5) | 22.5 | 20.5 | 21.5 |

- **Neither arm beats the starting checkpoint on the primary metric.** 24 more steps on the clean
  pipeline (A) cost 1.5 clean dev items and 1 unseen item. B roughly holds its ground (−0.5 and
  0).
- Of the items the starting checkpoint passes, the share a run fails:

  | Arm | Dev | Unseen | Benchmark |
  | --- | --- | --- | --- |
  | A | 9% | 50% (1 of 2) | 2% |
  | B | 3% | 0% | 5% |

- A's benchmark is 2 points above the start (24.0 vs 22). The benchmark is reported only and
  chooses nothing; on the primary metric A is worse.

### 4.6 Costs

| | Per run |
| --- | --- |
| Training tokens | 98,304 (both arms) |
| Training time | 53–61 min (fp32, about 30 tokens/s) |
| Total time (data, training, 3 evaluations, diagnostic) | 88–106 min |
| Peak memory | 11.3 GB training, 7.7 GB evaluation |
| Disk | 7.3 GB per checkpoint (weights plus optimizer state), deleted after scoring |
| Whole experiment | 4 runs in 7 h 47 min of wall clock (03:44 to 11:31 UTC), including the discarded first A1301 attempt (48 min) |

### 4.7 Decision: no V5.1 model release candidate

- **Spec-first answers (B) are not adopted.** The pre-registered verdict is rejected. On the
  primary metric alone it would be inconclusive (+2.0, 95% CI [−10.7, +14.7]). The benchmark,
  which only reports, is 1.5 points lower, and invalid outputs rise.
- **No V5.1 model release candidate.** No trained arm beats the starting checkpoint (section
  4.5), and the release rule (a retained treatment, never a benchmark-selected checkpoint) is
  not met. The round-54 checkpoint (`anchor_sha256` 2538b55d…) stays the best-supported model.
  The runner deletes each run's checkpoint after scoring. Every run can be rebuilt from its
  manifest, seed, the spec and the commit recorded in `results/m4/runs/`.
- **What V5.1 delivers is the evaluation and experiment release.** It contains:
  - versioned, fingerprinted sets (`eval-v5.1-1`);
  - the unseen-concept set (`unseen-v1`, 53 items);
  - independent robust tests (`robust-v1`);
  - leakage checks by name and by behaviour;
  - failure classification;
  - the format-matched loss diagnostic;
  - a resumable, disk-checked experiment runner with matched seeds and a paired-comparison
    verdict.
  
  With these, the next intervention can be measured in about 3.4 hours of compute per seed pair
  at this budget.
- **The headline number for the next round** is the gap the dev set hid: clean dev 25.5–26.5
  of 44, but unseen concepts 1–2 of 53.

### 4.8 Recommended next steps (each needs approval for compute)

1. **Fix the category rule before the next comparison.** Count, per category, the items the
   control passes and the treatment fails, not failure counts per type. Pre-register it with the
   spec, before any run.
2. **Target the retrieval failure directly: near-twin write examples.** For each trained concept,
   generate a verified twin that differs in one word of the task and in the code (another
   operation, unit or position), in the same batch. Then the distinguishing word, not
   the surface of the prompt, decides the code.
   - Every twin must pass `behaviour_leaks` and the unseen-set overlap check, which already
     reject equivalents of held-out items. Several held-out items are exactly such twins of
     trained concepts (`hours_to_seconds`, `middle_of_three`), so the twin generator is the
     part of this plan with the highest leakage risk.
   - Expected cost: one generator, plus one 2-seed comparison (about 7 hours on this host,
     less on an AMX host).
3. **Use a budget that can show an effect.** 24 steps cover about a quarter of one pass over
   the data, and between-seed noise is only one item. So:
   - Run 72 steps, the production budget, where it is affordable (AMX host).
   - Use 3 seeds, so that the one-sided margin falls from 6.3 to 2.9 times the standard error.

## 5. Acceptance criteria

| Criterion | Status |
| --- | --- |
| Verification tests pass | Done, with nothing else running, at the final commit: `tests/test_v51.py` 39 of 39. Full suite: 368 of 370 passed in the staging worktree. The 2 failures are `tests/test_local_provider.py` tests that check the bundled `models/` folder, which the staging worktree does not check out. That file passes 10 of 10 in the main checkout at the same code commit. |
| Weak tests strengthened, historical scores kept | Done. `robust-v1` adds independent assertions for all 150 held-out functions (180 items: 30 functions appear in both the write and the bug-fix suite), plus hand-written edge cases for `first_word`, `fibonacci`, `maximum`, `minimum`, `second_largest` and `is_between`. Official tests and scores are unchanged; every table reports official, clean and robust scores with denominators. |
| Failures ranked | Done (section 2): task interpretation 66%, logic 18%, code generation 7%, edge cases 6%, formatting 3%, evaluation failures 0%. Re-measured on every trained run (section 4.4). |
| Experiment reproducible | Done. Pinned in the repo: the spec and the code commit. Recorded per run: the data manifest (options, seed, token count) and hygiene record, the evaluation-set fingerprints (dev `b4b74aea…`, unseen `6cbdf242…`, benchmark `c41681ec…`), the starting checkpoint's SHA-256 and the trained checkpoint's SHA-256. The same seed and data reproduced the same step-1 loss (section 3). The runner resumes from a mid-run checkpoint and refuses to start with under 8 GB of free disk. |
| Results reported honestly | Every run is reported, including the discarded first attempt (section 3). The pre-registered verdict is applied as written, and its weakness is stated rather than fixed after the fact. Post-hoc analyses are labelled. The benchmark never chose anything. |
| No benchmark or evaluation leakage | Every run's generated data passed the name and behaviour leakage checks with nothing to remove (`hygiene.json` is empty), and the replay filter dropped the replay rows that name a benchmark function (6 for seed 1301, 3 for seed 1302). No checkpoint was selected on any score. |
