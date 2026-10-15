# Experiment v51-m3-spec-first

Decision metric: `unseen_clean` = clean dev (/44) + unseen-concept set (/53), out of 97: concepts the starting checkpoint never saw, in any form.
Benchmark scores are reported only. Mean ± standard deviation over seeds.

| Arm | Seeds | Primary | Dev clean /44 | Unseen /53 | Unseen robust | Dev official /62 | Bench official /65 | Bench clean /61 | Bench robust /65 | Halluc. dev+unseen | Invalid dev+unseen | Train min | Peak RSS MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 2 | 26.5 ± 0.7 | 25.5 ± 0.7 | 1.0 ± 0.0 | 0.0 ± 0.0 | 37.5 ± 0.7 | 24.0 ± 1.4 | 21.0 ± 1.4 | 23.0 ± 1.4 | [4, 4] | [5, 4] | 55 | 11314 |
| B | 2 | 28.5 ± 0.7 | 26.5 ± 0.7 | 2.0 ± 0.0 | 1.0 ± 0.0 | 38.5 ± 0.7 | 22.5 ± 2.1 | 20.5 ± 2.1 | 21.5 ± 2.1 | [5, 6] | [6, 5] | 59 | 11315 |

| Arm | Answer loss: train / held-out (validation) / unseen dev clean / unseen new | Unseen-task failure classes (clean, mean per seed) |
| --- | --- | --- |
| A | 0.2622 / 0.0091 / 0.2257 / 0.9577 | {'algorithmic_or_logic_error': 10.0, 'edge_case_failure': 3.0, 'formatting_or_parsing': 3.5, 'incorrect_code_generation': 5.0, 'incorrect_task_interpretation': 30.5} |
| B | 0.2303 / 0.0124 / 0.611 / 1.3547 | {'algorithmic_or_logic_error': 9.5, 'edge_case_failure': 1.0, 'formatting_or_parsing': 1.5, 'incorrect_code_generation': 8.0, 'incorrect_task_interpretation': 31.0} |

| Comparison | Paired diffs | Mean | 95% CI | One-sided 95% margin | Category change (+ = more points lost) | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| B vs A | [3, 1] | 2.0 | [-10.71, 14.71] | 6.31 | {'formatting': 1.0, 'hallucination': 1.5, 'coding': 4.5, 'reasoning': -9.0} | **rejected** |
