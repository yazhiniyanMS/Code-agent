# Ablation abl3: targeted data for the largest error class

The error analysis (`../ERROR_ANALYSIS.md`) and abl2 (`ABL2_RESULT.md`) show that most lost points
come from the model writing a neighbouring function's body (`fourth_item` -> `fifth_item`). abl3
tests two data changes aimed at that, each with both baseline seeds, compared seed by seed against
the abl2 baseline runs (same configuration and code; the default data was verified byte-identical
after adding the `NOVEL_DRAWS` setting). With two seeds the noise floor of a mean is 3 / sqrt(2),
about 2.1 dev points.

| Group | Change | Dev per seed (1101, 1102) | Bench per seed | Dev instruction-following lost | Decision |
| --- | --- | --- | --- | --- | --- |
| base | none | 35, 38 | 22, 22 | 14, 14 | baseline |
| novel_x5 | 5x fresh-constant exercises (30 constants, all 8 ordinals; 136 -> 528 examples) | 35, 37 | 22, 21 | 15, 13 | not kept (mean -0.5) |
| contrast_x3 | 3 contrast bug fixes per exercise instead of 1 (588 -> 1,680) | 35, 37 | 25, 24 | 17, 14 | not kept (mean -0.5; instruction following +1.5) |

Neither change reduces instruction-following errors on held-out prompts. contrast_x3 scores
higher on the benchmark (+2.5 mean), but the benchmark is not used to choose, and the dev set,
which decides, shows no gain.

## Conclusion after abl2 and abl3

Fourteen single changes from the round-54 checkpoint (data cleaning, data mix, targeted data,
learning rate, steps, batch size, sequence length, weight averaging) all land within the seed
noise of the baseline on the held-out dev set. Further rounds of this fine-tuning recipe are not
expected to move held-out performance; the plateau since about round 36 is a property of the
model and recipe, not of the training settings.
