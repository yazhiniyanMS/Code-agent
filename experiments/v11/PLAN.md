# V11 plan and decision rules (written before any V11 result)

## Scope and storage

- **Repositories:** the Git repository is `/home/user/Code-agent`, branch
  `claude/affectionate-rubin-7rl1am`, remote `github.com/yazhiniyanMS/Code-agent`.
  `/home/user/lmwork` is the (non-Git) working area for checkpoints, data and run outputs.
- **Disk space** came from the main worktree's working copy of `models/`, about 3.2 GB of published
  v2-v5 model files. It now uses the same sparse checkout as the staging worktree (`'/*' '!/models/'`).
  - All 120 blobs were checked present in the local Git store first, and they are also on the remote.
  - `git sparse-checkout disable` restores them. No checkpoint was moved or deleted.
- **Training runs one at a time** on the RAM disk (`/dev/shm`, 16 GB; training peaks at about 11 GB).
- **Every kept candidate is stored as a verified lossless pack** (`ycode/lm/packckpt.py`) in
  `/home/user/lmwork/v11/ckpt`.
- **Runs are watched actively.** The container restarted twice while idle in V10, killing runs.

## Gates (unchanged from V6/V10; not to be changed after results)

- **Selection metric:** clean dev /44 + clean unseen-v1 /53 = combined /97.
- **Second-seed gate** for a one-seed candidate, both required:
  - combined >= 32;
  - no failure category loses 3 or more of round 54's clean passes (`items_lost`, official passes).
- **Promotion** (replaces round 54 as the checkpoint of record), the V6 rule:
  - every seed beats 29;
  - mean combined >= 32 (a mean gain >= 3);
  - mean `items_lost` < 3 in every category, over the seeds.
- **Always reported, never used to select:**
  - the benchmark (official, clean and robust);
  - the new-family probe /24, neighbours /38 and held-out families /66;
  - reproduction and held-out-combination sets;
  - robust dev and unseen-v1 passes.
- **Audit of every promoted candidate's gained items, before promotion:**
  - no training example shares the item's name, AST skeleton or behaviour (the built-in hygiene
    check, rerun);
  - no duplicate of the item in the data;
  - whether each gain is robust (robust tests).
  - A gain from leakage voids the candidate. A weak gain is reported as such.
- **unseen-v2 and the benchmark are never used to choose.** unseen-v2 is never evaluated. The training
  data's decontamination against it stays on.

## Runs (sequential; each from the verified C2 unless stated; one principal change each)

| Run | Hypothesis | Changed variable | Expected | Forgetting risk | Seeds | Cost | Stop / next |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | R1's +4 over round 54 is not one seed's luck | R1's seeds only | combined about 33, the same 3 inherited losses | same as R1 | train 1302, data 1303 | ~1.3 h train + ~1.6 h eval | always run |
| B | A curriculum that restores what C2 forgot keeps R1's gains without its losses | how the 50% replay is drawn: weighted by C2's forgetting of each example (C2's loss minus round 54's, on a fresh production round; never dev, unseen or benchmark items) | combined >= 32 with the 3 losses restored | lower than R1; risk of narrowing replay | 1301 (data 1302) | +~0.5 h scoring, ~3 h run | second seed if both gates pass; otherwise C |
| C | Correct bug-fix contrast pairs stop the echo (returning the buggy code): pairs whose buggy code differs from the fix by one token, from round 54's own bug-fix generators, with the decisive tokens weighted | B's recipe plus those pairs | the bug-fix echo removed | low | 1301 | ~3 h | only if B leaves the bug-fix losses; second seed if gates pass |
| D | Cross-family compositions (grammar v3) add transfer once retention holds | the best of B/C plus grammar v3 (R6's addition) | unseen-v1 above R1 (R6 reached 5/53) | dev fell in R6 | 1301 | ~4 h | only if B or C passes the gates |

- A second seed of any run that passes the second-seed gate comes before any new run.
- The forgetting investigation runs alongside A. It covers the training targets, the tokenized
  examples and loss masks, the sampling frequencies, per-token losses of the correct answers, and the
  generated outputs for `remove_spaces` and the `longest_word` bug fix. It changes no gate. Dev items
  are inspected, never trained on.
