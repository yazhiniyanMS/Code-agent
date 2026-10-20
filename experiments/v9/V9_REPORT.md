# YCode V9-1: fine-tuning on a wider compositional grammar, with the benchmark

Status: **complete (one seed; the pre-registered rule stops the screen here).** Every number is
measured. The benchmark is reported only and played no part in any choice. The fixed holdout
(unseen-v2) was not evaluated. Round 54 (sha256 `2538b55d...`) and L2 (`fdf3d162...`) are unchanged
(re-hashed after the V8 runs).

## Answer

**No candidate replaces round 54.**

- **The fine-tune (C2, seed 1301):** 28/44 dev clean and 3/53 unseen-v1 clean, so dev + unseen-v1 is
  31, against round 54's 29. That is the highest of the 18 runs since V6, but only one item above
  V6's D (30 on both seeds) and G (30 on seed 1301). Its +2 is short of the V6 rule's mean gain of 3.
- **Benchmark (reported only):**

  | Model | Official | Clean | Robust |
  | --- | --- | --- | --- |
  | Round 54 | 22/65 | 19/61 | 20/65 |
  | C2 | 21/65 | 18/61 | 20/65 |
  | Earlier fine-tunes with a benchmark score (V6) | 18-27/65 | | |

- **Composition reaches held-out items, but only those made of trained parts.**
  - Grammar v2 can express 20 of the clean dev, unseen-v1 and benchmark items as new combinations of
    its parts. Every one was excluded from training by name, skeleton and behaviour.
  - On those 20, round 54 scores 2 and C2 scores 5.
  - On the other 138 held-out items, C2 scores 44 to round 54's 46: churn, with 7 gained and 9 lost.
  - Two of the new unseen-v1 passes are exactly such combinations: `count_capitalized` (words that
    start with an upper-case letter, counted) and `sum_at_odd_positions`.
- **No cross-family transfer.** Held-out families score 1/66 (round 54: 2), and the neighbour set 2/38
  (4).
  - Before training, grammar v2 could express only 5 of the 66 held-out-family tasks.
  - The pre-registered continuation rule (at least 6/66 with dev + unseen-v1 at least 27) is not met,
    so no second seed was run.
- **Grammar v2 is under-learned at 3 examples per program.**
  - C2 reproduces 26/40 of its trained programs (round 54: 0/40) and writes 32/60 held-out
    combinations (round 54: 0/60).
  - V8's COMP managed 57/60 and 55/60 on its single-domain grammar.
  - The errors are part swaps: the right structure with a selection or transform taken from another
    combination.

## 1. What was run

Grammar v2 (`ycode/lm/compositions2.py`) builds `aggregate(transform(selection(input)))` over four
domains:
- an integer list, with new views of it: neighbour differences, pair sums, odd positions, the last
  three, the numbers before the first negative, the running maximum and the numbers after the first;
- the digits of a non-negative integer;
- the characters of a string;
- the words of a text.

Every part names the V8 training family it was taken from, and a test checks that none comes from a
held-out family. Names state what they compute (`count_above_3_...`), which fixes the V8
`count_odd` interference.

- **Hygiene:** 753 combinations with distinct behaviour, of which 693 are kept. The 60 dropped are:
  - 26 that match a dev, unseen or benchmark item (name, skeleton or behaviour);
  - 26 that compute a function round 54 was trained on;
  - 8 that compute, on its own 40 probe inputs, a cross-family, held-out-family, within-family or
    neighbour task.
- **Split** (`ycode/lm/v9_grammar_split.json`), per domain (numbers / digits / chars / words):
  - 350 training programs: 105 / 85 / 55 / 105;
  - 60 held-out combinations (`c2_test`): 15 per domain, each made of parts trained in that domain;
  - 40 trained programs sampled for reproduction (`c2_trained`): 10 per domain.
- **Recorded before training:** grammar v2 can express 5 held-out-family tasks (`digit_product`,
  `largest_jump`, `shortest_word_length`, `sum_of_even_digits`, `sum_of_odd_digits`).
- **Run:** from round 54, 3 examples per program, 50% replay from round 54's own training
  distribution, learning rate 1e-4, seed 1301, Lion on the top 24 layers.
  - 56 steps instead of the pre-registered 52. Grammar v2's longer task texts make the data 14%
    larger (238k tokens), and 56 steps is one epoch, as 52 was for COMP.
  - 229,376 tokens were trained, 2.8 exposures per program (1-5).
  - 166 minutes of training (23.8 tokens/s; this host ran slower than in V8) and 268 minutes in
    total.

## 2. Against round 54 and every earlier run

Produced by `experiments/v9/compare.py`.

| Model | Dev clean | Unseen-v1 clean | Dev + unseen | Bench official | Bench clean | Bench robust |
| --- | --- | --- | --- | --- | --- | --- |
| Round 54 | 27/44 | 2/53 | 29 | 22/65 | 19/61 | 20/65 |
| V6 C 1301 | 23/44 | 1/53 | 24 | 18/65 | 16/61 | 14/65 |
| V6 D 1301 | 28/44 | 2/53 | 30 | 24/65 | 22/61 | 23/65 |
| V6 D 1302 | 28/44 | 2/53 | 30 | 26/65 | 23/61 | 24/65 |
| V6 E 1301 | 24/44 | 1/53 | 25 | 26/65 | 23/61 | 23/65 |
| V6 F 1301 | 26/44 | 2/53 | 28 | 24/65 | 21/61 | 22/65 |
| V6 G 1301 | 27/44 | 3/53 | 30 | 22/65 | 20/61 | 19/65 |
| V6 G 1302 | 26/44 | 2/53 | 28 | 27/65 | 25/61 | 23/65 |
| V7 L1 1301 | 21/44 | 2/53 | 23 | - | - | - |
| V7 L2 1301 | 15/44 | 1/53 | 16 | - | - | - |
| V7 P1 1301 | 23/44 | 0/53 | 23 | - | - | - |
| V8 B50 1301 | 21/44 | 2/53 | 23 | - | - | - |
| V8 C75 1301 | 25/44 | 2/53 | 27 | - | - | - |
| V8 COMP 1301 | 26/44 | 2/53 | 28 | - | - | - |
| V8 COMP 1302 | 24/44 | 1/53 | 25 | - | - | - |
| V8 MORE 1301 | 25/44 | 1/53 | 26 | - | - | - |
| V8 MORE 1302 | 23/44 | 2/53 | 25 | - | - | - |
| V8 T50 1301 | 25/44 | 1/53 | 26 | - | - | - |
| **V9 C2 1301** | 28/44 | 3/53 | 31 | 21/65 | 18/61 | 20/65 |

- **Benchmark scores:** round 54's are its saved V5.1 answers rescored with the current scorer
  (dev 27/44 and unseen-v1 2/53 reproduce its record exactly). V7 and V8 runs did not evaluate the
  benchmark, and their models were not kept (disk), so they have no benchmark score.
- **C2's dev + unseen-v1 of 31** is the highest of these 18 runs, one above V6's D (30, 30) and G
  (30, 28). Its benchmark (21/65) is below round 54 and below every V6 candidate except C.

Task sets (strict: correct on all 40 probe inputs), paired with round 54 by task, exact sign test:

| Set | Round 54 | C2 1301 | Only run / only round 54 (p) |
| --- | --- | --- | --- |
| c2_test | 0/60 | 32/60 | 32 / 0 (p = 0.000) |
| c2_trained | 0/40 | 26/40 | 26 / 0 (p = 0.000) |
| t_families (all) | 2/66 | 1/66 | 0 / 1 (p = 1.000) |
| t_families (expressible in grammar v2) | 0/5 | 0/5 | 0 / 0 (p = 1.000) |
| t_families (not expressible) | 2/61 | 1/61 | 0 / 1 (p = 1.000) |
| neighbours | 4/38 | 2/38 | 1 / 3 (p = 0.625) |

- **Held-out families:** C2 solves none of the 5 tasks grammar v2 can express, and 1 of the other 61
  (round 54: 2).
- **Neighbours:** 2/38 against 4/38.

## 3. Held-out items grammar v2 can express

`idioms.held_out_check` drops every combination that computes a dev, unseen-v1 or benchmark
item. The items they compute are exactly the held-out items grammar v2 can express as a new
combination of its parts. None of them was trained.

Grammar v2 computes: bench:dict_size, bench:digit_sum, bench:first_word, bench:maximum, bench:minimum, bench:reverse_string, bench:second_item, bench:to_upper, bench:total, dev:count_uppercase, dev:double_all, dev:last_word, dev:product, dev:running_max, unseen:count_capitalized, unseen:digits_of, unseen:drop_first, unseen:last_three, unseen:sum_at_odd_positions

Clean items only:

| Set | Items | Round 54 | C2 1301 | Gained / lost (first run) |
| --- | --- | --- | --- | --- |
| dev expressible | 6 | 0 | 1 | bugfix:product / - |
| dev other | 38 | 27 | 27 | bugfix:first_negative, bugfix:index_of_min, write:first_three_chars / bugfix:longest_word, bugfix:remove_spaces, write:remove_spaces |
| unseen expressible | 5 | 0 | 2 | write:count_capitalized, write:sum_at_odd_positions / - |
| unseen other | 48 | 2 | 1 | write:double_plus_one / write:count_words_longer_than, write:keys_above |
| bench expressible | 9 | 2 | 2 | write:to_upper / bugfix:maximum |
| bench other | 52 | 17 | 16 | bugfix:factorial, bugfix:filter_even, write:is_even / bugfix:is_sorted, bugfix:square, write:count_words, write:count_zeros |

- **On the expressible items, C2 gains 4 and loses 1:** 2 -> 5 of 20.
  - **`count_capitalized`** comes from two parts never trained together. The capitalized-words
    selection was trained in 16 programs (`list_lengths_capitalized_words`,
    `ndistinct_capitalized_words`, ...). The count aggregate was trained in only 4 programs, none over
    words (`count_last_three`, `count_first_half_chars`, `count_distinct_digits`,
    `count_nonzero_digits`).
  - **`sum_at_odd_positions`** comes from the odd-positions selection, trained in 12 programs
    (`max_odd_positions`, `prod_odd_positions`, `sum_capped_odd_positions`, ...), and the sum
    aggregate.
  - `bugfix:product` is gained, as by every V8 grammar run.
- **On the other 138 items, C2 is at round 54's level:** 44 against 46, with 7 gained and 9 lost, the
  churn seen in every earlier run.
- These are counts from one seed: small numbers, not yet a replicated effect.

## 4. Learning the grammar

| Set | Domain | Round 54 | C2 1301 |
| --- | --- | --- | --- |
| c2_test | numbers | 0/15 | 12/15 |
| c2_test | digits | 0/15 | 9/15 |
| c2_test | chars | 0/15 | 7/15 |
| c2_test | words | 0/15 | 4/15 |
| c2_trained | numbers | 0/10 | 6/10 |
| c2_trained | digits | 0/10 | 6/10 |
| c2_trained | chars | 0/10 | 6/10 |
| c2_trained | words | 0/10 | 8/10 |

- **Numbers come closest to V8's single-domain grammar:** 12/15 held-out combinations.
- **Words are hardest:** 4/15. Word programs combine string and number parts (lengths), and their
  task texts are the longest.
- **Reproduction of trained programs is 6-8/10 per domain.** The errors keep the program's structure
  and swap one part for another combination's, for example:
  - `prod_doubles_running_max` written over `numbers[1:]`;
  - `first_capped_before_negative` written with a filter instead of the cap.
  At 2.8 exposures, 350 programs over 4 domains and 85 domain-specific parts are not yet separated.
  V8's COMP had 1 domain and 27 parts at 2.9 exposures.

## 5. Decision

- **No second seed.** The pre-registered continuation rule needs held-out families at least 6/66;
  C2 has 1/66.
- **No candidate.** The V6 rule needs every seed to beat 29 with a mean gain of at least 3. C2's
  single seed is +2, the same order as V6's D (+1 on both seeds) and G (+1, -1).
- **The benchmark** is 21/65 against 22/65 and was not used in this decision.

## 6. What this adds, and what to run next

- **New evidence.** Composition training transfers to natural-wording held-out items that are new
  combinations of trained parts. On those items C2 goes from 2/20 to 5/20, including 2 of the 5
  unseen-v1 items grammar v2 can express. It does not transfer to anything else.
- **What remains.** The remaining dev, unseen-v1 and benchmark failures, and the held-out families,
  need parts no grammar here contains.
- **If anything is run next,** the cheapest informative step is a second seed of C2 (about 4.5 hours
  on this host). It would show whether the 2 -> 5 gain on expressible items replicates before any
  grammar is widened. Widening the grammar toward the failing items would mean designing it from
  held-out items, which the protocol forbids.
