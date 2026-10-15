# YCode-LM v5.1 mission: reliable evaluation and unseen-concept generalisation

Measured results are marked **measured**; hypotheses say so. Nothing below comes from a run that
did not happen. Raw data is in `results/`. The experiment is still running; sections 3 and 4 are
filled in as runs finish.

## 1. Evaluation audit (phase 1)

**Measured.** All checks are in `tests/test_v51.py` (38 tests, all passing).

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
  checks per-step time and peak memory.

## 4. Results

Pending.
