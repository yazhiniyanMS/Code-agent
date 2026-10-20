# YCode V8: learning new programs without forgetting, and testing transfer

Status: **complete**. Every number is measured, on development tasks only. The
fixed holdout (unseen-v2) was not used anywhere in V8. Round 54 (sha256 `2538b55d...`) and L2
(`fdf3d162...`) are unchanged; both hashes were re-verified before V8 started.

## Answer

**No candidate replaces round 54.** Round 54 and L2 are unchanged, and the fixed holdout (unseen-v2)
was not evaluated.

- **Replay solves most of the forgetting, if it adds tokens rather than displacing the curriculum.**
  - At L2's fixed budget of 98,400 tokens, replay trades one for the other. Dev rises 15 -> 21 -> 25
    (of round 54's 27) as replay goes from 24% to 54% to 77% of the tokens. Trained programs fall
    35 -> 32 -> 28 /36, and new constants 14 -> 12 -> 9 /24.
  - With 50% replay and about twice the tokens (52-56 steps), dev is 24-26/44 and the curriculum is
    still learned: COMP writes 55-58 of each 60 of its programs.
  - What forgetting remains is mostly the same few items from run to run:
    - `subtract_three_each` (now written as `x + 3`) and `bugfix:remove_spaces`, each lost in 6 of
      the 7 replay runs;
    - `count_odd`, which the grammar runs lose to the grammar's own naming (section 4).
- **Transfer across families did not improve in any run.**
  - Held-out families: 0-2/66, against round 54's 2.
  - Neighbour set: 2-4/38, against 4.
  - Unseen-v1: 1-2/53, against 2.
  - This holds for a balanced 144-behaviour curriculum over 123 families (T50) and for both stage-3
    arms.
- **Learning does generalize, but only inside the families and grammar that were trained:**
  - new constants of trained families: L2 14/24;
  - held-back variants of trained families: T50 7/45, against round 54's 1;
  - above all, new combinations of trained parts: COMP 55 and 54 /60 on two seeds, against round
    54's 1.
- **Diverse, compositionally structured examples beat more examples of fewer programs.** With equal
  curriculum tokens, COMP (352 programs x 3 examples) beats MORE (60 programs x 18) on held-out
  combinations: 55 against 45 at seed 1301 (p = 0.006) and 54 against 45 at seed 1302 (p = 0.022).
  Both arms cost the same 2.8 hours per run. Neither reaches other families, so diversity within one
  grammar is not yet transfer.
- **Next:** test whether composition carries across families when the parts come from the training
  families' own operations (section 5; the first screen is one seed, about 3 hours).

## 0. Baselines and protocol (verified before any run)

- **Hashes:** round 54 and L2's kept model match their records.
- **L2's data reproduces byte for byte.** It was rebuilt at the current commit and again with the V8
  pipeline (tagged rows, same shuffle).
- **Nothing that scores answers has changed since L2.** No code under `ycode/` and no evaluation
  script or set changed since L2's commit, and the dev and unseen-v1 fingerprints match
  (`b4b74aea...`, `6cbdf242...`).
- **Scoring:**
  - Dev and unseen-v1 use their official tests and the V7 rules (clean items: /44, /53).
  - Every task set counts an answer correct only if it is right on all 40 probe inputs of its task
    (`strict`). The inputs are the program's own 40, or for the neighbour set its tests followed by
    typed variations.
- **Same settings for every run:** from round 54, Lion, top 24 layers, 4 x 1024-token rows, greedy
  evaluation in bf16. Seed 1301 everywhere; stage 3 repeats both arms at seed 1302.
- **Time** is wall-clock hours on this 4-core host, evaluation included.
- **Exact token and exposure counts.** Every run logs the tokens it actually trained on by source.
  Each curriculum behaviour's exposures are the answer tokens trained divided by the example's answer
  tokens, summed over its examples. L2's exposures were reconstructed by replaying its sampler on
  its byte-identical data.

## 1. Replay against forgetting (stage 1, `v8_replay.json`)

These runs share L2's curriculum examples (976 write examples of 36 programs), learning rate 1e-4
and the 98,400 trained tokens of 24 steps. Replay is sampled from round 54's own training
distribution: one production round of the v5 mix plus original SFT replay rows, drawn to a token
share (`rounddata.legacy_share`).

| Run | Non-curriculum share of trained tokens | Exposures per behaviour (mean, min-max) | Dev clean /44 | Unseen-v1 /53 | Trained programs /36 | New constants /24 | Neighbours /38 | Train min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Round 54 | - | - | **27** | 2 | 0 | 0 | 4 | - |
| A = L2 | 24% (basics writes + 50 replay rows) | 23.6 (11.5-31.4) | 15 | 1 | **35** | **14** | 1 | 47 |
| B50 | 54% (production mix) | 14.2 (6-24) | 21 | 2 | 32 | 12 | 2 | 58 |
| C75 | 77% | 7.3 (2-15) | 25 | 2 | 28 | 9 | 4 | 55 |

At a fixed budget, replay trades forgetting against learning:
- Each step up in replay restores dev (15, then 21, then 25 of round 54's 27).
- The cost is curriculum exposures, and with them trained programs (35, 32, 28) and new constants
  (14, 12, 9).
- About 14 exposures still learn most programs. About 7 lose whole families (`shift_by`,
  `sum_of_first`).
- Unseen-v1 and the neighbour set stay at round 54's level throughout.

## 2. Transfer across families (stage 2, `v8_transfer.json`)

`experiments/v8/transfer_split.py` splits the 171 curriculum families that have held-out-clean
programs:
- **Held out:** 48 families, stratified by source and level, never trained.
- **Trained:** 123 families with 144 behaviours. By level: 65 single idioms, 47 multi-step, 32
  algorithms. 136 behaviours have aliases and 42 a trained sibling.
- **Within-family test:** every multi-variant family holds one variant back, 45 in all.
- **Cross-family test:** the held-out families' 66 programs, after dropping 6 that share a skeleton
  with a trained program.

T50 trains this curriculum: 4 write examples per name, 50% replay, 56 steps, 229,600 tokens, about
6.8 exposures per behaviour, 120 min of training.

| Model | Dev clean /44 | Unseen-v1 /53 | Trained programs /144 | Held-back variants /45 | **Held-out families /66** | New constants /24 | Neighbours /38 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Round 54 | 27 | 2 | 8 | 1 | **2** | 0 | 4 |
| L2 | 15 | 1 | - | 13* | **2** | 14 | 1 |
| T50 | 25 | 1 | 32 | 7 | **2** | 5 | 4 |

\* 11 of L2's 13 are variants of the constant families it trained in full: trained programs for L2.

- **Replay solved forgetting** at this budget: dev 25/44.
- **The broad curriculum was under-learned.** At about 7 exposures T50 writes 22/65 single-idiom
  programs, 7/47 multi-step and 3/32 algorithms.
- **No cross-family transfer.** On held-out families T50 scores 2/66, the same as round 54 and L2: it
  gains `window_min_2` and loses `keep_big_values`. Mean agreement with the reference over the 40
  inputs rises from 0.14 to 0.20, but no more answers are correct. The neighbour set stays at 4/38.

## 3. Compositional structure against more examples (stage 3, `v8_comp.json`)

A V8 grammar (`ycode/lm/compositions.py`) builds programs of the form
`aggregate(transform(selection(numbers)))` from 11 selections, 7 element transforms and 9
aggregates. The task text is composed the same way, for example "returns the largest of the squares
of the numbers that come after the first two in the list".
- **Candidates:** 461 combinations have distinct behaviour. 412 are kept after dropping 19 that
  match a held-out item and 30 that compute a function round 54 was trained on.
- **Test set:** 60 combinations held out (`c_test`), each made of a selection, transform and
  aggregate that all occur in training.
- **Arms:** both start from round 54 with 50% replay, 52 steps (213,200 trained tokens, one epoch)
  and learning rate 1e-4, and use matched curriculum tokens (99.8k and 102.1k):
  - **COMP:** 352 combinations x 3 examples (2.9 exposures each).
  - **MORE:** 60 of them x 18 examples (17.3 exposures each).

| Model | Dev clean /44 | Unseen-v1 /53 | Programs both arms train /60 | COMP-only programs /60 | **Held-out combinations /60** | Held-out families /66 | Neighbours /38 | Train min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Round 54 | 27 | 2 | 0 | 1 | **1** | 2 | 4 | - |
| COMP, seed 1301 | 26 | 2 | 57 | 55 (trained) | **55** | 1 | 2 | 106 |
| MORE, seed 1301 | 25 | 1 | 60 | 43 (new for MORE) | **45** | 0 | 3 | 111 |
| COMP, seed 1302 | 24 | 1 | 56 | 58 (trained) | **54** | 1 | 2 | 109 |
| MORE, seed 1302 | 23 | 2 | 59 | 40 (new for MORE) | **45** | 1 | 2 | 110 |

- **Compositional structure makes learning cheap.** COMP writes 55-58 of each 60 of its programs
  after about 3 examples each, on both seeds. Each selection, transform and aggregate recurs in
  dozens of programs. By contrast, the constant families needed about 14-24 exposures, and T50's
  assorted families learned 22% at 7.
- **Both arms generalize to new combinations of known parts, and diversity generalizes better on
  both seeds.**
  - On the 60 held-out combinations, round 54 scores 1, COMP 55 and 54, and MORE 45 and 45.
  - Paired by task, seed 1301 has 11 tasks solved only by COMP and 1 only by MORE (exact sign test,
    p = 0.006); seed 1302 has 11 and 2 (p = 0.022).
  - COMP is consistent across seeds: 52 tasks are solved on both seeds and 57 on at least one.
  - MORE also writes 43 and 40 of a 60-program sample of COMP's other programs, which are new
    combinations for it.
- **Neither crosses families.**
  - Held-out families: COMP 1 and 1, MORE 0 and 1, of 66 (round 54: 2).
  - Neighbours: 2-3/38 (4).
  - Unseen-v1: 1-2/53 (2).
  - None of the 66 held-out-family tasks is expressible in this grammar. The grammar takes one
    integer list, and none of the 57 one-list tasks computes any of its 461 combinations. So the
    grammar alone cannot reach them.
- **Replay mostly holds dev:** COMP 26 and 24, MORE 25 and 23, of 44 (round 54: 27). Section 4
  lists the items lost.
- **Hygiene, checked by behaviour as well.** Two neighbour tasks compute grammar combinations:
  `product_of_positives` is `prod_pos` and `squares_total` is `sum_sq`. Neither was trained:
  `prod_pos` is a held-out combination, and `sum_sq` is excluded as a function round 54 was trained
  on. `test_no_trained_combination_computes_a_test_task` now checks that no COMP or MORE training
  program computes, on the 40 probe inputs, what any cross-family, within-family or neighbour task
  asks for.

## 4. Decision

The V6 rule is unchanged. A candidate replaces round 54 only if:
- every seed beats all_clean 29 (dev clean + unseen-v1 clean),
- the mean gain is >= 3 items,
- no failure category loses >= 3 items that round 54 passes.

| Run | Seed | Dev clean /44 | Unseen-v1 /53 | Total | Dev items lost / gained against round 54 |
| --- | --- | --- | --- | --- | --- |
| Round 54 | - | 27 | 2 | **29** | - |
| B50 | 1301 | 21 | 2 | 23 | 9 / 3 |
| C75 | 1301 | 25 | 2 | 27 | 5 / 3 |
| T50 | 1301 | 25 | 1 | 26 | 4 / 2 |
| COMP | 1301 | 26 | 2 | 28 | 4 / 3 |
| COMP | 1302 | 24 | 1 | 25 | 5 / 2 |
| MORE | 1301 | 25 | 1 | 26 | 5 / 3 |
| MORE | 1302 | 23 | 2 | 25 | 7 / 3 |

**No run reaches 29 on any seed, so there is no candidate, and round 54 stays.**
- L2 stays a diagnostic checkpoint.
- The fixed holdout (unseen-v2) was not evaluated, because nothing passed screening.

The dev items show where the remaining forgetting comes from:
- **The replay runs lose largely the same items.** Of the 7 runs:
  - `subtract_three_each` (now written as `x + 3`) and `bugfix:remove_spaces` are lost in 6;
  - `remove_spaces`, `sum_odd` and `bugfix:longest_word` in 4 each.
- **All four grammar runs lose `count_odd` the same way: to the grammar's own naming.** In the
  grammar, `count_<selection>` means "how many of the selection are greater than 3". So for "how
  many numbers in the list are odd" the trained models write the grammar's pattern, for example
  `sum(1 for y in ys if y > 0)`.
  - This is a "similar names, different behaviour" interference created by the curriculum itself.
  - B50 also loses `count_odd`, but differently: it counts odd numbers among the first four only.
- **Not every gain is a grammar effect.** `first_three_chars` is gained by every V8 run and by L2, so
  fine-tuning of any kind brings it. `bugfix:product` is gained only by COMP and MORE, on every seed:
  it is a new combination of the trained `prod` aggregate (the exact item is excluded from training).

## 5. What to run next

The evidence points to one bottleneck. The model generalizes systematically over parts it has
trained in many combinations, at about 3 examples per program, while 50% replay keeps dev. But it
gains nothing on families whose operations it never trained in composition. None of the 66
held-out-family tasks is expressible in the V8 grammar, which covers selection, element transform
and aggregate over one integer list. Those families need positional, windowed, pairwise, digit,
string, dict and grid operations.

**V9-1, a single-seed screen of about 3 hours.** Build grammar v2 from the training families'
own operations, and train COMP-style on it.
- **Primitives:** positions and neighbouring pairs, windows, digits of an integer, characters and
  words of a string, dict entries, with thresholds as parameters. Derive them only from the 123
  training families; do not design them by looking at held-out families, dev or unseen-v1 items.
- **Naming:** names say what is counted (for example `count_above_3_even`), which removes the
  `count_odd` interference.
- **Hygiene:** exclude every combination that computes a held-out item, a cross-family task or a
  neighbour task, by skeleton and by behaviour. `test_no_trained_combination_computes_a_test_task`
  already checks the behaviour half for V8.
- **Run:** 3 examples per combination, 50% replay, 52 steps, learning rate 1e-4, seed 1301.
- **Before training (minutes, no training):** record how many held-out-family tasks grammar v2 can
  express. This is an upper bound on what the screen can show.
- **Scoring:** the same sets as stage 3.
- **Pre-registered continuation rule:** run a second seed (another 3 hours) only if held-out families
  reach at least 6/66 (round 54: 2) with dev + unseen-v1 at least 27. Otherwise stop.
  - If grammar v2 can express many held-out-family tasks yet solves few, then training on
    compositions does not carry over to other families' natural wording.
  - If it can express few, the screen could not test transfer, and those families need their own
    parts.

**Not worth running:**
- more replay-ratio points (stage 1 already maps the trade-off);
- more examples per program (MORE);
- longer training on the T50 curriculum. It was under-learned at 6.8 exposures, but stage 3 shows
  that generalization beyond the trained programs comes from shared structure, not repetition.
