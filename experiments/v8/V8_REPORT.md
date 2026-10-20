# YCode V8: learning new programs without forgetting, and testing transfer

Status: **in progress** (stage 3 running). Every number is measured, on development tasks only. The
fixed holdout (unseen-v2) was not used anywhere in V8. Round 54 (sha256 `2538b55d...`) and L2
(`fdf3d162...`) are unchanged; both hashes were re-verified before V8 started.

## Answer

ANSWER_PENDING

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
- **Same settings for every run:** from round 54, seed 1301, Lion, top 24 layers, 4 x 1024-token
  rows, greedy evaluation in bf16.
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
| COMP, seed 1302 | SEED2 | | | | | | | |
| MORE, seed 1302 | SEED2 | | | | | | | |

- **Compositional structure makes learning cheap.** COMP writes 57/60 and 55/60 of its programs
  after about 3 examples each. Each selection, transform and aggregate recurs in dozens of programs.
  By contrast, the constant families needed about 14-24 exposures, and T50's assorted families
  learned 22% at 7.
- **Both arms generalize to new combinations of known parts, and diversity generalizes better.** On
  the 60 held-out combinations COMP scores 55 and MORE 45, against round 54's 1. Paired by task, 11
  are solved only by COMP and 1 only by MORE (sign test over tasks, p = 0.006). MORE also writes
  43/60 of COMP's other programs, which are new combinations for it.
- **Neither crosses families.** Held-out families: 1/66 and 0/66, against round 54's 2. Neighbours:
  2/38 and 3/38 (4). Unseen-v1: 2/53 and 1/53 (2).
- **Replay holds.** Dev 26/44 and 25/44, against round 54's 27.

## 4. Decision

DECISION_PENDING
