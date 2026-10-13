# Error analysis of YCode-LM v5 fine-tuning (rounds 8-56)

Every answer from every evaluated round was re-run against each test assertion, and every failure
was given one category. Data: `analysis/failures.jsonl` (3,457 failure records: prompt, expected
reference and per-assertion expected values, actual answer and code, exception, category, reason,
model, decoding settings), `analysis/rounds.json` (category totals per round), `analysis/report.md`
(tables). Rebuild with `failure_history.py`; analyse one run with
`ycode-lm analyze ANSWERS.jsonl --set bench|dev --store failures.jsonl`.

## Which errors cost the most points

| Category | Benchmark, all rounds | Benchmark, per evaluation (rounds 45-54) | Dev set, all rounds | Dev, per evaluation (rounds 47-56) |
| --- | --- | --- | --- | --- |
| Instruction following | 961 (50%) | 20.8 | 895 (58%) | 13.4 |
| Reasoning | 468 (24%) | 11.6 | 511 (33%) | 9.9 |
| Factual accuracy | 195 (10%) | 4.1 | 0 | 0.0 |
| Coding (crashes) | 132 (7%) | 2.2 | 19 (1%) | 0.0 |
| Formatting | 87 (5%) | 0.3 | 79 (5%) | 0.5 |
| Hallucination | 84 (4%) | 2.2 | 26 (2%) | 0.6 |

A recent benchmark evaluation loses about 41 of 65 points; **about 21 of them are instruction
following**, and the same holds on the dev set, which nothing was selected or designed on.

## What the categories mean here

- **Instruction following (half the lost points).** Almost always the model writes, verbatim, the
  body of the most similar function it was trained on, ignoring the word that differs:
  `to_upper` -> `to_lower`'s body, `celsius_to_fahrenheit` -> `celsius_to_kelvin`'s,
  `count_vowels` -> `string_length`'s, `maximum` -> `most_common`'s, `is_empty` -> `has_one_item`'s,
  `second_item` -> `third_item`'s; on the dev set `last_word` -> `second_to_last_word`,
  `longest_word` -> `shortest_word`, `fourth_item` -> `fifth_item`. It is retrieval of a memorised
  neighbour, not misunderstanding of the format. Smaller parts: the bug handed back unchanged,
  a different function name, wrong parameters.
- **Reasoning (a quarter).** The code runs but computes the wrong thing: `flatten` returns
  `[lists]`, `is_palindrome` is always False, `product` returns its input, `weeks_to_days(1)`
  gives 0.01.
- **Factual (benchmark only).** Items that hinge on a fact the model does not know: the factorial,
  the Celsius-to-Fahrenheit formula, which letters are vowels, 60 seconds per minute.
- **Coding, hallucination.** Crashes (`RecursionError` in `factorial`, division by zero in
  `binary_search`) and invented APIs (`a.dict`, `s.toupper()`, calling an undefined `square`).
- **Formatting.** Nearly gone since round 23 (0.3 per evaluation): the def-line fix removed
  `defis_zero(`-style answers.

## What this implies

1. The biggest single lever is the nearest-neighbour copying. Changes worth testing (each as an
   ablation against two baseline seeds): fewer repetitions of the same hand-written concepts
   (they are seen hundreds of times and act as attractors), more prompt pairs that differ in one
   word and need different code, and down-weighting answers identical to a neighbour's.
2. Reasoning and factual errors are mostly missing knowledge; fine-tuning on exercises has not
   moved them in 45 rounds.
3. Formatting is solved; coding and hallucination are small.

The "body of a training function" check uses today's training generators, so a few early-round
failures that copied functions added later are labelled with today's names.
