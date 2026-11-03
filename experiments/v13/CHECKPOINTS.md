# V13 checkpoints: identity, location and verification

T1's spec (`specs/v13_t1.json`) names the checkpoints it needs by identity, not by path. A path such as
`/home/user/lmwork/v12/ckpt/S1_1301.ypk` exists only on the machine that wrote it.

## What T1 needs

| Role | Checkpoint | Identity | Stored as | Its own dependencies |
| --- | --- | --- | --- | --- |
| starting point (`init_checkpoint`) | S1 seed 1301, the V12 champion | content `5075d097a6571cc964907a011b6d4f634100bfb393b61bf5a18b897abb7837f7` | delta pack, 0.42 GB (file sha256 `4b9f4898c35a...`) | parent R1, base round 54 |
| parent of S1 | R1 seed 1301 (V10) | content `9909c511043b8a166b4a713e50ad206d84a47cbc927e9ed17bfc9b0e161c6ec3` | pack against round 54, 1.32 GB (file sha256 `eb7ec5951101...`) | base round 54 |
| base and anchor (`anchor_checkpoint`) | round 54 (V5.1 checkpoint of record) | model.pt sha256 `2538b55d5aee371899a988799168042af4da27c03cafff3417a3c9925b8e8e98`, 5,206,846,811 bytes | training checkpoint directory (model.pt, info.json) | none needed to load it |

- **Content hash:** a hash of the tensors themselves (`packckpt.content_sha256`), not of the file.
- **Records:** `experiments/v12/V12_REPORT.md`, `experiments/v11/artifacts/CHECKPOINTS.json` and
  `experiments/v5_finetune/expA/manifest_round54.json`.

## How they are found and checked

1. **Point the runner at the checkpoints:** put them in one directory, in any layout, and set
   `YCODE_CHECKPOINT_DIR` to it.
   - The spec's `file` entries (`v12/ckpt/S1_1301.ypk`, `loop/anchor`) are only the first place looked.
   - Packs are found by their content hash; a base by its size and SHA-256.
2. **Before any data is built,** the experiment runner (`resolve_checkpoints` in `ycode/lm/experiment.py`)
   locates every checkpoint the spec names and checks it.
   - S1's pack must have S1's content hash.
   - It must have been packed against R1, and R1 must exist with R1's content hash.
   - The chain must end at R1, as the spec says.
   - Round 54's model.pt must have its SHA-256; it is hashed in full.
   - The located chain goes into each run's `result.json` as `init_chain`.
3. **Loading checks every tensor** against the pack's content hash; for S1 this also covers R1.
   - The frozen layers come from the base. Its SHA-256 is checked against `YCODE_BASE_SHA256`, which the
     runner sets for the training and evaluation processes it starts.
4. **Failures are loud.** A missing checkpoint, parent or base raises `CheckpointNotFound`, and one with
   another hash raises `CheckpointMismatch`. Either way, the error names what was looked for and where.
   Nothing is ever used in a checkpoint's place.

Checking a checkpoint directory without running anything:

```bash
export YCODE_CHECKPOINT_DIR=/path/to/checkpoints
python -c "import json; from ycode.lm.experiment import resolve_checkpoints; \
spec = resolve_checkpoints(json.load(open('experiments/v13/specs/v13_t1.json'))); \
print(spec['anchor']); print(*spec['init_chain'], sep='\n')"
```

Without `YCODE_CHECKPOINT_DIR`, specs that give paths (`anchor`, `init_from`) work as before.

## Where to obtain them: not currently obtainable

On 2026-11-03 this work moved to a new container. The three checkpoints existed only on the previous
container's disk (`/home/user/lmwork`), and no copy was ever stored elsewhere: V11 could not attach a release
asset (`403`) or register an LFS upload (`experiments/v11/V11_REPORT.md`).

- **Until they are recovered,** T1 stops at step 2 with `CheckpointNotFound`, naming round 54 first.
- **Recovery is a separate investigation:** can round 54 be rebuilt and checked against
  `2538b55d...`, then R1 and S1 from their recorded recipes?
- **A V13 run needs one of two things:** these exact checkpoints, or a newly selected and pre-registered
  baseline. Another checkpoint must not stand in for S1.

## Not covered: data paths

`tokenizer` (`/home/user/lmwork/data/tokenizer.json`), `replay_dir`, `out`, `work_dir` and `keep_dir` are
still paths. The tokenizer and replay data were on the same lost disk. They need their own recovery and
checks before T1 can run.
