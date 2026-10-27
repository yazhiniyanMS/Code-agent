# YCode V11: replicating R1, finding the forgetting, and publishing

Status: STATUS

## Executive summary

SUMMARY

## 1. Starting state (verified before any training)

- **Repositories:**
  - Git repository: `/home/user/Code-agent`, branch `claude/affectionate-rubin-7rl1am`, remote
    `https://github.com/yazhiniyanMS/Code-agent` (public; the session's account `yazhiniyanMS` has
    admin rights).
  - `/home/user/lmwork` is the non-Git working area for checkpoints, data and runs.
- **Rules:** the selection and promotion rules are V6's and V10's, unchanged. They are written out in
  `experiments/v11/PLAN.md`, committed before any V11 result.
- **Checkpoints and SHA-256 hashes** (full list: `experiments/v11/inventory_sha256.txt`):

  | Checkpoint | Path | File SHA-256 | Tensor content SHA-256 | Size |
  | --- | --- | --- | --- | --- |
  | Round 54 (checkpoint of record) | `lmwork/loop/anchor/model.pt` (fp32) | `2538b55d5aee3718...` | - | 5.21 GB |
  | Round 54 optimizer state (needed to rebuild C2) | `lmwork/loop/anchor/optim.pt` | `bf51203b1f882b80...` | - | 2.13 GB |
  | C2 | `lmwork/v10/ckpt/C2_1301.ypk` | `931ae406e1ff3f34...` | `cc2f2fa07777e69e...` | 1.32 GB |
  | R1 | `lmwork/v10/ckpt/R1_1301.ypk` | `eb7ec5951101986...` | `9909c511043b8a16...` | 1.32 GB |
  | R6 | `lmwork/v10/ckpt/R6_1301.ypk` | `fcc97a345a292d1c...` | `0e8bf2a6a881b174...` | 1.35 GB |
  | R7 | `lmwork/v10/ckpt/R7_1301.ypk` | `cd3d6c78e329ff4b...` | `2d75e24a54747...` | 1.34 GB |

  - The `.ypk` files are lossless packs (`ycode/lm/packckpt.py`): bf16 bits XOR round 54's, compressed.
  - Restoring one needs round 54 at `lmwork/loop/anchor` and the tokenizer at
    `lmwork/data/tokenizer.json` (`7079ff7b...`).
  - The tensor content hash is checked on every unpack. C2's pack was made from the checkpoint whose
    file hash is `9c138498...`.
- **Restoring was verified end to end.** C2 was unpacked and evaluated on dev by the runner's own
  command, and all 62 answers are byte-identical to V9's record.
- **Resources:**
  - 4 CPU cores, no native bf16, 15.7 GB RAM, a 16 GB RAM disk.
  - The disk had **448 MB free** at the start.
  - Space came from the main worktree's copy of `models/` (the published v2-v5 models, 3.2 GB). It now
    uses the same sparse checkout as the staging worktree. All 120 blobs were checked present in Git
    first and are also on GitHub; `git sparse-checkout disable` restores them. No checkpoint was moved
    or deleted.

## 2. V11-A: R1 replicated at another seed

A_SECTION

## 3. Why `remove_spaces` and the `longest_word` bug fix are lost

FORGETTING_SECTION

## 4. Experiments after the replication

EXPERIMENTS_SECTION

## 5. Decision: validated best and best experimental candidate

DECISION_SECTION

## 6. Checkpoints, restoration and disk

ARTIFACTS_SECTION

## 7. Publishing to GitHub

PUBLISH_SECTION

## 8. Next experiment

NEXT_SECTION
