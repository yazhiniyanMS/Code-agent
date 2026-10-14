# Ablation abl2: data quality and training settings, one change at a time

Every arm starts from the round-54 checkpoint (`loop/anchor`), trains 72 steps on data built with
seed 1101, and is scored with the same prompts, scoring and decoding (the evaluation is
deterministic, see `../expA/RESULT.md`). Decisions use the dev set only; the benchmark is reported.
Rule: keep a change only if its dev score beats the baseline by more than the seed-to-seed noise
floor and no error category loses 2 or more extra points. Per-arm results: `abl2/<arm>.json`,
full table: `abl2/summary.txt`.

| Arm | Change | Dev /62 | Bench /65 | Decision |
| --- | --- | --- | --- | --- |
| base_s1 | none (seed 1101) | 35 | 22 | baseline |
| base_s2 | none (seed 1102) | 38 | 22 | noise floor: 3 dev points |
| data_dedup | drop exact duplicates and second answers for the same task (4,458 -> 3,785 examples) | 36 | 23 | not kept |
| data_cap_repeats | at most 3 examples per answer body, 3 writes per concept (-> 2,133) | 36 | 24 | not kept |
| data_no_replay | no replay of earlier data | 37 | 24 | not kept |
| data_skills | add the 49 skill functions (the loop's current default) | 34 | 17 | not kept: hallucination +3 |
| ft_average | average the new weights 50/50 with the anchor | 36 | 24 | not kept |
| lr_half | learning rate 7.5e-6 | 37 | 25 | not kept |
| lr_double | learning rate 3e-5 | 35 | 24 | not kept: instruction following +2 |
| steps_double | 144 steps | 36 | 23 | not kept |
| batch_half | batch 2 | 36 | 19 | not kept |
| seq_512 | sequence length 512, batch 8 | 36 | 22 | not kept |

## What it shows

1. **Seed noise is 3 dev points** on 62 items; the benchmark score was 22 under both seeds.
   With one seed per arm, effects smaller than about 3 points cannot be told from noise.
2. **No data-cleaning or training-setting change helps beyond the noise.** All ten changes land
   at 34-37 on dev, between the two baseline seeds. Learning rate, steps, batch size and sequence
   length do not matter at this point.
3. **The model has memorised its fine-tuning data.** The training loss at steps 50/60/70 is the
   same with the learning rate doubled (0.077/0.177/0.181 vs 0.076/0.177/0.181): it follows which
   batches come up, not the setting. Doubling the steps adds nothing.
4. **The skill families hurt.** Adding them raised hallucinations on dev from 1 to 4 (the
   category-damage rule) and cost 5 benchmark points. The round loop includes them by default;
   it should run without them.
5. **The anchor's dev score (39) is partly selection luck.** It was chosen as the best dev score,
   and no fresh run of the same recipe (35-38) or averaging with it (36) reaches it again.
6. **Across the 11 arms, 32 dev items always pass, 12 flip and 18 never pass.** The never-pass
   items are mostly a neighbour's body (`fourth_item` -> `fifth_item`, `first_three_chars` ->
   `first_two_chars`, `last_word` -> `second_to_last_word`, `smaller` -> `larger`) or the wrong
   operation (`product` returns its input, `multiply_three_numbers` adds). That is the target of
   the next ablation, abl3: more fresh-constant exercises and more contrast bug fixes, each run
   with both seeds and compared seed by seed (`abl3.json`).
