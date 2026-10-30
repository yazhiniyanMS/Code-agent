# V13 plan and decision rules (written before any V13 training result)

V13 fine-tunes the champion, S1 (V12), on string operations, keeping what S1 can do.

## 1. Starting point (verified 2026-10-30)

**S1 and everything needed to rebuild it are on disk and verified.** Nothing of S1 is changed by V13.

| S1 seed | Stored | Verified |
| --- | --- | --- |
| 1301, checkpoint of record | `lmwork/v12/ckpt/S1_1301.ypk`: delta pack against R1 (`lmwork/v10/ckpt/R1_1301.ypk`), base round 54 (`lmwork/loop/anchor`) | after the 03:08 restart: all 217 tensors restored, content `5075d097...` |
| 1302, replication | rebuilt from V11-A's pack (`lmwork/v11/ckpt/A_1302.ypk`) with its V12 spec, unchanged (`experiments/v13/specs/v13_baseline_s1_1302.json`, driver `experiments/v13/drivers/rebuild_s1_1302.sh`) | the rebuild's dataset is byte-identical to V12's (35,232 tokens), and its checkpoint is bit-identical: sha256 `3fdbf898...` = the V12 record |

**S1 1302 has no disk copy.** The disk has 0.2 GB free and S1 1302's pack is 1.3 GB. It lives on the RAM disk, which is
wiped by each container restart (five between 2026-10-28 23:05 and 2026-10-30 05:15). The rebuild above makes it reproducible exactly in about
15 minutes.

**S1's scores** (V12 records, `experiments/v12/results/runs/S1/`):

| S1 seed | Combined clean dev + unseen-v1 | Robust-clean | Neighbours /38 | New constants /24 | Held-out families /66 | Held-out combinations /60 | Benchmark official · clean · robust |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1301 | 33 (30 + 3) | 28 | 3 | 4 | 4 | 55 | 22 · 19 · 22 |
| 1302 | 31 (30 + 1) | 30 | 3 | 3 | 2 | 50 | 23 · 20 · 21 |

**S1's string ability before V13,** on the sets of `experiments/v13/sets` (PR #23). Strict means all 40 probe outputs are
right; "tests" means the answer passes the task's tests:

| S1 seed | strings_heldout /32: strict · tests | strings_trained /47: strict · tests |
| --- | --- | --- |
| 1301 | 0 · 0 | 1 · 5 |
| 1302 | 0 · 0 | 0 · 4 |

S1's answers are coherent code for the wrong task. Asked for `periods_to_commas`, it writes a word-reversing function:
V7's "wrong function retrieved", on strings. There is a lot of room, and the failure is the one the curriculum's
minimal pairs and boundary tests target.

## 2. What is trained (all merged into this branch)

- **Data:** the string curriculum (#22), 47 concepts in 8 groups, every example executed against its tests, with
  bug fixes on all 47 (#24).
- **Replay:** uniform replay of round 54's production mix (`legacy_share`), the V8-V12 retention mechanism, and S1's
  own V12 data, the 31 literal bug fixes once each (`literal_bugfix: 1`).
- **Forgetting while training:** each step's loss on the replayed rows against the string rows (`loss_by_source`, #25).
  At the end of each run, every retention set is compared with S1 of the same seed.

## 3. Runs (one variable changed at a time; start small)

**T1** (`experiments/v13/specs/v13_t1.json`):

- **Start and seeds:** from S1 1301, seed 1301, data seed 1302.
- **Data:** `strings_write: 4`, `strings_bugfix: 2`, `literal_bugfix: 1`, `legacy_share: 0.5`.
  - The dry run gave 58,377 tokens: strings 34.7%, round 54 replay 58.8%, S1's literal bug fixes 6.4%.
  - The hygiene check removed nothing.
- **Training:** S1's own stage settings. lr 3e-5 warmup-stable-decay, 1 warmup step, top 24 layers, fp32, Lion, batch
  4 × 1,024, one epoch = 13 steps (55,816 training-split tokens / 4,096). Compact bf16 final checkpoint.
- **Cost:** about 30 min of training, then the screen (section 4), about 30 min.

**What follows T1 depends only on its screen, and each next run changes one variable:**

| T1's screen | Next run |
| --- | --- |
| G1-G3 pass | full evaluation (G4, G5); if those pass, T1 on seed 1302 from S1 1302 (data seed 1303) |
| G1 fails, G2 and G3 pass (strings did not move enough) | **T2** = T1 with lr 1e-4 |
| G1 passes, G2 or G3 fails (S1's abilities slipped) | **T3** = T1 with `legacy_share: 0.75` |
| G1 fails and G2 or G3 fails | **T3**, then T2 only if T3 keeps G2 and G3 |

At most three arms on seed 1301 before reporting. A candidate gets its second seed only after passing every gate on
seed 1301.

## 4. Gates (fixed now, before any V13 training result)

The reference is always S1 of the same seed, so a run is judged by what it changed.

| Gate | Requirement, per seed | Measured on |
| --- | --- | --- |
| **G1: strings improve** | strings_heldout strict ≥ S1's + 3 | screen |
| **G2: S1's clean evaluation kept** | combined clean dev + unseen-v1 ≥ S1's − 1 | screen |
| **G3: no category regresses** | fewer than 3 of S1's clean official passes lost in any failure category (V6's `items_lost` rule, with S1 of the same seed in place of round 54) | screen |
| **G4: robust-clean kept** | robust-clean dev + unseen-v1 ≥ S1's − 1 | full evaluation |
| **G5: unrelated families kept** | neighbours /38, new constants /24, held-out families /66 and held-out combinations /60 each ≥ S1's − 2 | full evaluation |

**Promotion,** which replaces S1 as the checkpoint of record:

- G1-G5 pass on both seeds;
- the mean combined clean score over the two seeds is at least S1's (32.0);
- the audit passes. It is V11's: no training example shares a gained item's name, AST skeleton or behaviour (the
  hygiene check, rerun); no duplicate of a gained item; each gain checked against the robust tests. A gain from leakage
  voids the candidate, and a weak gain is reported as weak.

**Reported, never used to choose:**

- the benchmark (official, clean, robust);
- strings_trained, which is training-program reproduction;
- training-program reproduction c2_trained;
- the failure-mechanism counts and the `loss_by_source` curves.

**unseen-v2 stays sealed** until the final evaluation you authorize.

**A known limit, stated before the results:** G1's threshold (+3 of 32, about 10%) is a judgement, not a measured
noise level. S1 scores 0 on this set, so any real learning clears it on one seed. The second seed is what guards
against a lucky run.

## 5. Storage and when training starts

- **T1 does not start until you decide the disk question.**
  - A V13 candidate packed as a delta against S1 needs about 0.3-0.45 GB. The disk has 0.2 GB.
  - Your instructions: resolve the disk issue before launching further training, and delete nothing without approval.
  - The options are in the development log and the session report.
- **S1 1302's rebuild was not V13 training.** It reran an existing, recorded recipe to restore a checkpoint the RAM disk
  had lost, wrote only records to disk (dataset and logs, under 1 MB), and its result is bit-identical to V12's.
- **Runs are reproducible from their spec and seeds alone,** as the rebuild shows. A run lost to a restart is rerun, not
  resumed.
