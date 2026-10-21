# V10 plan and decision rules (written before any V10 result)

## Starting point

- **C2** is V9-1's run: grammar v2, 350 programs x 3 examples, 50% replay, 56 steps, lr 1e-4 WSD (2
  warmup steps, stable for 80%, cosine to 1e-5), seed 1301, from round 54.
  - Its checkpoint was deleted after evaluation (`keep_models` false, 1.7 GB free).
  - **R0 rebuilds it** with the identical spec (`v10_c2_rebuild.json`) and keeps it on the RAM disk.
  - The rebuild is C2 if its final checkpoint sha256 equals V9-1's record (`9c138498...`). If not, the
    rebuild is evaluated in full and used under its own numbers, and the difference is reported.
- **Round 54** (`/home/user/lmwork/loop/anchor`) is the control and stays unchanged.

## Measurements (every candidate; one script, one set of settings)

- **Selection:**
  - dev clean /44;
  - unseen-v1 clean /53;
  - combined /97;
  - the new-family-member probe `family_new` /24, strict on all 40 inputs.
- **Reported, never used for selection:**
  - the benchmark (official /65, clean /61, robust /65);
  - cross-family neighbours /38 and held-out families /66, strict;
  - training-program reproduction (`c2_trained` /40, plus a set for any new curriculum);
  - held-out combinations (`c2_test` /60).
- **Forgetting:** dev and unseen-v1 items lost and gained against round 54 and against C2.
- **Cost and identity:** training minutes, trained tokens, peak RAM, checkpoint sha256.
- **unseen-v2** is never evaluated. The training-data decontamination that has run since V6 (dropping
  any example whose name, AST skeleton or behaviour matches a dev, unseen, unseen-v2 or benchmark
  item) stays on. It only removes data; no model is ever run on unseen-v2.

## Decision rules

- **Second seed:** a one-seed candidate gets a second seed when both hold:
  - its combined score is at least 32 (C2 + 1);
  - no failure category loses 3 or more of the items round 54 passes.
- **Replacing round 54:** the V6 rule, unchanged. Every seed beats 29, the mean gain is at least 3
  (mean combined at least 32), and no category loses 3 or more items round 54 passes.
- **Better than C2:** a candidate must also beat C2's combined score on the mean of its seeds.
- **What is not evidence of generalization:** reproducing training programs, or new constants of
  trained families, alone.

## Experiments, one change at a time, cheapest first

The order of R2-R4 depends on R1's result and is decided before each launch.

| Run | Change from C2 | Question |
| --- | --- | --- |
| R0 | C2 rebuilt | Does C2 reproduce? (required: the checkpoint is gone) |
| R1 | From C2: more steps on new examples of C2's own 350 programs (data seed 1302), same recipe (as run, fixed in `v10_r1.json` before launch: 2 examples per program, 36 steps = one epoch) | Does consolidation help? C2 reproduces only 26/40 of its programs |
| R2 | From C2: cross-family compositions (grammar v3: training-family operations chained across domains, e.g. word lengths, then neighbour differences, then an aggregate) | B: does combining concepts across families transfer? (Replay already restores dev, so the decision policy puts this first among the coverage changes.) |
| R3 | From C2: minimal pairs (each program next to its sibling that differs in one part, with V7's pair weighting) | C: do contrastive distinctions remove C2's part-swap errors? |
| R4 | R1's recipe at lr 3e-5 | D: a conservative rate against the strong one |

Replay (A) is not rerun as its own experiment. V8 measured it (shares of 24-77%), and every V10 run
keeps C2's 50%.

## Amendment after R1 (written before R5 runs)

- **R1's result:** combined 33 (dev 30, unseen-v1 3). It loses 3 instruction-following dev items round
  54 passes: `write:remove_spaces`, `bugfix:remove_spaces` and `bugfix:longest_word`. C2 loses the
  same 3. Under the gate above, R1 gets no second seed. The gate is unchanged.
- **The diagnosis:** on both bug fixes, R1 returns the buggy code unchanged. Neither grammar curriculum
  contains a find-the-bug example.
- **R5:** R1's recipe plus one verified bug-fix example per program (`v10_r5.json`). It runs after R2.
- **R5's rules:** the gates and rules above, unchanged. Dev items are never used to build training
  data: the bug-fix examples come from the grammar programs, by the generator every curriculum uses.
