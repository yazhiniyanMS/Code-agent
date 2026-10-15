### Per run (v51-m3-spec-first)

Primary = clean dev + unseen, out of 97. Losses are answer-token cross-entropy.

| Arm | Seed | Primary /97 | Dev /62 | Dev clean /44 | Dev robust /62 | Unseen /53 | Unseen robust /53 | Bench /65 | Bench clean /61 | Bench robust /65 | Loss unseen dev / new | Loss train rows / val rows | Code loss unseen dev / new (own format) | Train loss (last 10 steps) | Halluc. | Invalid | Train tokens | Train min | Total min | Peak MB train / eval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 1301 | 26 | 37 | 25 | 37 | 1 | 0 | 25 | 22 | 24 | 0.2327 / 0.9766 | 0.3289 / 0.009 | None / None | 0.1115 | 7 | 5 | 98304 | 53 | 88 | 11314 / 7659 |
| A | 1302 | 27 | 38 | 26 | 38 | 1 | 0 | 23 | 20 | 22 | 0.2187 / 0.9387 | 0.1955 / 0.0093 | 0.2756 / 1.1448 | 0.0814 | 7 | 4 | 98304 | 57 | 98 | 11314 / 7662 |
| B | 1301 | 29 | 39 | 27 | 38 | 2 | 1 | 24 | 22 | 23 | 0.6219 / 1.3641 | 0.29 / 0.0106 | None / None | 0.1205 | 7 | 8 | 98304 | 61 | 105 | 11315 / 7659 |
| B | 1302 | 28 | 38 | 26 | 37 | 2 | 1 | 21 | 19 | 20 | 0.6001 / 1.3453 | 0.1707 / 0.0142 | 0.2442 / 0.9978 | 0.1002 | 9 | 7 | 98304 | 57 | 106 | 11315 / 7659 |

### Per arm (mean ± standard deviation over seeds)

| Arm | Seeds | Primary /97 | Dev clean /44 | Unseen /53 | Unseen robust | Bench /65 | Bench robust | Loss unseen new | Code loss unseen new (own format) | Val loss | Halluc. | Invalid | Regression vs start (dev / unseen / bench) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 2 | 26.5 ± 0.7 | 25.5 ± 0.7 | 1.0 ± 0.0 | 0.0 ± 0.0 | 24.0 ± 1.4 | 23.0 ± 1.4 | 0.958 ± 0.027 | 1.145 | 0.009 ± 0.000 | 7.0 ± 0.0 | 4.5 ± 0.7 | 0.09 ± 0.02 / 0.50 ± 0.00 / 0.02 ± 0.03 |
| B | 2 | 28.5 ± 0.7 | 26.5 ± 0.7 | 2.0 ± 0.0 | 1.0 ± 0.0 | 22.5 ± 2.1 | 21.5 ± 2.1 | 1.355 ± 0.013 | 0.998 | 0.012 ± 0.003 | 8.0 ± 1.4 | 7.5 ± 0.7 | 0.03 ± 0.00 / 0.00 ± 0.00 / 0.05 ± 0.00 |

Regression vs start: share of the items the starting checkpoint passes that a run fails, among the items the run answered. Starting checkpoint's answers: dev `/home/user/lmwork/v51m3/anchor_eval/dev.jsonl`, unseen `/home/user/lmwork/v51/m2/unseen_anchor.jsonl`, bench `/home/user/lmwork/v51m3/anchor_eval/bench.jsonl`.

### Paired comparison

| Comparison | Paired diffs (primary) | Mean | 95% CI | Category change (dev+unseen, + = more lost) | Verdict |
| --- | --- | --- | --- | --- | --- |
| B vs A | [3, 1] | 2.0 | [-10.71, 14.71] | {'formatting': 1.0, 'hallucination': 1.5, 'coding': 4.5, 'reasoning': -9.0} | **rejected** |

### Item-level transitions (post hoc: not the pre-registered decision rule)

Per matched seed: clean dev and unseen items (the primary metric's 97) that the first arm passes and the second fails (gained), and the reverse (lost). The exact test pools item x seed pairs as if independent: it asks whether these trained checkpoints answer this fixed item set differently, not whether another training run would; only more seeds measure run-to-run variance.

| Comparison | Seed | Gained | Lost | Gained items | Lost items |
| --- | --- | --- | --- | --- | --- |
| B vs A | 1301 | 5 | 2 | dev:bugfix:longest_word, dev:bugfix:remove_spaces, dev:write:count_odd, dev:write:remove_spaces, unseen:write:keys_above | dev:bugfix:count_greater, dev:write:index_of_min |
| B vs A | 1302 | 4 | 3 | dev:bugfix:longest_word, dev:bugfix:remove_spaces, dev:write:remove_spaces, unseen:write:keys_above | dev:bugfix:count_greater, dev:bugfix:index_of_min, dev:write:index_of_min |
| B vs A | pooled | 9 | 5 | exact two-sided p = 0.424 | |

Noise floor: the same transitions between two seeds of one arm (same recipe, different seed).

| Arm | Seeds | Up | Down | Items up | Items down |
| --- | --- | --- | --- | --- | --- |
| A | 1302 vs 1301 | 1 | 0 | dev:write:count_odd | - |
| B | 1302 vs 1301 | 0 | 1 | - | dev:bugfix:index_of_min |

### Unseen-task failure classes (clean), per run

- A 1301: unseen {'incorrect_task_interpretation': 30, 'algorithmic_or_logic_error': 10, 'formatting_or_parsing': 5, 'incorrect_code_generation': 4, 'edge_case_failure': 3}; dev {'incorrect_task_interpretation': 12, 'algorithmic_or_logic_error': 4, 'edge_case_failure': 3}
- A 1302: unseen {'incorrect_task_interpretation': 31, 'algorithmic_or_logic_error': 10, 'incorrect_code_generation': 6, 'edge_case_failure': 3, 'formatting_or_parsing': 2}; dev {'incorrect_task_interpretation': 11, 'algorithmic_or_logic_error': 4, 'edge_case_failure': 3}
- B 1301: unseen {'incorrect_task_interpretation': 32, 'algorithmic_or_logic_error': 9, 'incorrect_code_generation': 7, 'formatting_or_parsing': 2, 'edge_case_failure': 1}; dev {'incorrect_task_interpretation': 12, 'edge_case_failure': 2, 'formatting_or_parsing': 2, 'algorithmic_or_logic_error': 1}
- B 1302: unseen {'incorrect_task_interpretation': 30, 'algorithmic_or_logic_error': 10, 'incorrect_code_generation': 9, 'formatting_or_parsing': 1, 'edge_case_failure': 1}; dev {'incorrect_task_interpretation': 13, 'edge_case_failure': 2, 'formatting_or_parsing': 2, 'algorithmic_or_logic_error': 1}
