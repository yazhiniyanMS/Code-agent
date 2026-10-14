# Experiment v51-arms

Decision metric: `dev_clean` (dev score without items the starting checkpoint saw in equivalent form).

| Arm | Seeds | Dev | Dev clean | Dev robust | Bench | Bench clean | Bench robust | Halluc. (dev) | Train min | Peak RSS MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 1 | 6.0 | 2.0 | 6.0 | 7.0 | 7.0 | 7.0 | [0] | 7 | 11317 |
| E | 1 | 6.0 | 2.0 | 6.0 | 7.0 | 7.0 | 7.0 | [0] | 3 | 11317 |

| Comparison | Paired diffs | Mean | 95% margin | Category change (dev, + = more points lost) | Verdict |
| --- | --- | --- | --- | --- | --- |
