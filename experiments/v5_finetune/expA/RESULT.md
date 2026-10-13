# Experiment A: is V5's score reliable?

**Question.** Does evaluating the same checkpoint again give the same score?

**Setup.** The round-54 checkpoint (the current anchor; `model.pt` SHA-256
`2538b55d5aee371899a988799168042af4da27c03cafff3417a3c9925b8e8e98`, tokenizer SHA-256 `b766ca56...`)
evaluated three times on the dev set (62 items) and the benchmark (65 items), same machine, same
code (git `170f7e3`, evaluator file hashes in `manifest_round54.json`), greedy decoding (temperature
0, at most 200 new tokens, no repeated 12-grams), seed 0, bfloat16 weights, 4 CPU threads.

**Result: pass.** All six evaluations finished.

| Set | Runs | Mean | Std dev | Same verdict on every item | Same answer text |
| --- | --- | --- | --- | --- | --- |
| Dev | 39, 39, 39 | 39.0 | 0 | 62/62 | 62/62 |
| Benchmark | 22, 22, 22 | 22.0 | 0 | 65/65 | 65/65 |

The evaluator is deterministic: every answer is identical character for character, and the scores
equal round 54's own logged scores (dev 39; benchmark 7 + 9 + 6 = 22). Differences between rounds
therefore come from training (data seed and weights), never from re-evaluation.

Per category (run 1; all runs identical):

| | Formatting | Instruction following | Hallucination | Coding | Factual | Reasoning |
| --- | --- | --- | --- | --- | --- | --- |
| Dev, points lost | 2 | 11 | 0 | 0 | 0 | 10 |
| Benchmark, points lost | 1 | 23 | 2 | 3 | 3 | 11 |

Groups: dev concept 5/15, dev slot 27/32, dev fix 7/15; benchmark standard 7/30, fresh 9/20, fixed 6/15.

**Rejection rate.** 9 of 53 rounds (17%): rounds 5, 7, 9, 21, 22, 41, 53, 55, 56.

**Round 53** (`round53.txt`). Its dev score was 36; the rule then required at least the anchor
minus 3 (38 - 3 = 35: met) and at least the best ever minus 3 (40 - 3 = 37: not met), so it was
rejected because of round 50's dev score of 40. Re-scoring its 62 saved answers with the same
harness gives 36 again, with the same verdict on every item. The round itself cannot be re-trained:
its starting weights (round 52) were replaced when round 54 was accepted. Rounds now write a
`round{N}.json` record (anchor, data seed, code commit, data mix, settings) before training.
