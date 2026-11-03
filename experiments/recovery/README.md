# Checkpoint recovery and reproducibility audit (2026-11-03)

On 2026-11-03 this work moved to a fresh container. `/home/user/lmwork` did not come with it, and no copy of
it exists elsewhere (`experiments/v11/V11_REPORT.md`: the release asset was refused with `403`, and no LFS
upload was ever registered). It held:

- every checkpoint since V5: round 54, C2, R1, R6, R7, V11-A/B/C, S1 and the V6/V7 runs;
- the round loop's state and per-round records;
- the tokenizer file and the replay data.

This audit asks whether round 54 can be rebuilt, and with it the chain round 54 -> C2 -> R1 -> S1. Every
link of that chain has a recorded hash.

**No training was run, and nothing was uploaded.**

## Round 54: dependencies and recovery

Round 54 is the V5 round loop's accepted model after round 54.
- **Source:** `experiments/v5_finetune/until_target2.py`.
- **How a round runs:** it builds data, trains 72 steps from the latest accepted model, scores dev, then
  accepts or rejects the round.
- **What is recorded:** `experiments/v11/artifacts/CHECKPOINTS.json` gives its parent as YCode-LM v5
  (step 60) plus 72 fine-tuning steps.
- **What that does not cover:** the 72 steps are only the last round. The model it started from was the
  latest round accepted before 54, which has its own chain back to v5.

| Dependency | Available now | Recorded identity | Recovery | Status |
| --- | --- | --- | --- | --- |
| Round 54 itself | no | model.pt SHA-256 `2538b55d5aee371899a988799168042af4da27c03cafff3417a3c9925b8e8e98` (40 records), 5,206,846,811 bytes | rebuild, or a copy | **missing** |
| YCode-LM v5 weights (`models/ycode-lm-v5`, bf16 export, 103 shards) | yes, in git | git blob ids (index `7c07fbda...`); no SHA-256 recorded | git | **verified:** files identical to the committed blobs |
| Which v5 file the loop started from | no | the loop's `state2.json` | none | **unknown:** may have been an fp32 training checkpoint, not this bf16 export |
| Embedded tokenizer | yes (in v5) | `b766ca56df3f25bda5df83cee7f69ef8505ad8dbf4ea3e8700ca2238c212d33d` (`expA/manifest_round54.json`) | v5's payload | **verified:** full hash matches |
| `tokenizer.json` (was `lmwork/data`) | regenerated | `7079ff7b8546a23c5416f8ae08adcb636ec69da4c14cd964114f12cee9745341` (109 records) | `BPETokenizer.from_dict(v5 tokenizer).save()` | **verified:** full hash matches |
| v5 SFT replay data (`sft_tokens.bin`, `sft_mask.bin`, `sft_index.npy`) | no | not recorded directly; run manifests since V8 hash the datasets built from it | `prepare_sft_dataset` from its sources: by default the installed Python's stdlib and site-packages, plus instruction data | **not verified:** depends on the exact Python and packages then; checkable only indirectly, by rebuilding a recorded run dataset |
| Loop code (`until_target2.py`, `average.py`, `policy.json`) | yes, in git (2026-10-13/14) | none | git | **partial:** edited while the loop ran; the policy changed at rounds 22, 26, 56 and 57 |
| `build_round.py` as each round ran it | no | copied into each round's record | none | **missing:** the loop read a live copy in `lmwork` |
| Per-round records: data seed (1000 + round), code commit, steps, lr, trained layers, precision, policy, starting round | no | `lmwork/loop` | none | **missing** |
| Accepted-round lineage before 54 | no | `state2.json` | partial inference only (`analysis/failures.jsonl` has answers for rounds 8-56, no training records) | **missing** |
| Precision per round | no | state; chosen per host by `precbench.py` (bf16 on an AMX host) | none | **missing**; bf16 results depend on the CPU (below) |
| Software | yes | torch `2.14.1+cu130`, Python 3.11.15, 4 threads (round 54 manifest and 37 run records) | venv | torch identical; Python 3.11.18 (patch differs); numpy not recorded (2.4.6 now) |
| CPU | yes | "Intel(R) Xeon(R) Processor @ 2.10GHz", AMX bf16 (round 54's evaluation host) | this container | **same model and features.** S1 (V12) was trained on a 2.8 GHz AVX-512 host without bf16 units (V12 report); C2's and R1's hosts were not checked |

**Conclusion: round 54 cannot be rebuilt from what survives.** The missing pieces are records, not compute:
- which rounds were accepted on the way to round 54;
- each round's data-builder source, precision and code commit;
- possibly the v5 file the loop started from.

Without round 54, C2, R1 and S1 cannot be rebuilt either. Their recipes, seeds and dataset hashes are
recorded, but all of them start from round 54 and pack against it. Only a surviving copy of round 54 (or of
the whole `lmwork` tree) recovers the chain.

## Reproducibility experiment (`repro/`)

A small model goes through the real training code: the `small` preset, 4 layers, width 256, 5M parameters.
- **Stage 1:** pretrain 4 steps.
- **Stage 2:** fine-tune the top 2 layers 4 steps (Lion, compact checkpoints, final bf16 save), as the
  V10-V12 runs did.
- **Setup:** every variant runs in a fresh process with seed 1301, on inputs built once.
- **Input check:** building the inputs twice gave identical files (`inputs_sha256_16.txt`), except
  `meta.json`, which records the source folder's path.
- **Environment:** in `environment.json` (torch 2.14.1+cu130, oneDNN 3.12.0, MKL; AMX and avx512_bf16
  present). Raw results are in `results.jsonl`.
- **Code:** run with PR #26's head (5c9cd3a). Rerun from this branch (main's code), the committed scripts
  reproduced the fp32 and bf16 default variants' hashes exactly.

| Variant | Pretrained tensors | Fine-tuned tensors |
| --- | --- | --- |
| fp32, default (twice) | `2cef974c00d4` both | `f00edba3926d` both |
| fp32, `MKL_CBWR=AVX512` | same | same |
| fp32, `ONEDNN_MAX_CPU_ISA=AVX512_CORE` (no AMX, no bf16 instructions) | same | same |
| bf16, default (twice) | `c617573b6f9f` both | `3c151147cadc` both |
| bf16, `ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16` (no AMX) | `ab12758fb0b8`: 58,619 of 3,418,368 values differ, max 0.004 | `a5981013e1f8`: 134,141 of 1,606,912 differ, max 0.0078 |
| bf16, `ONEDNN_MAX_CPU_ISA=AVX512_CORE` (no bf16 hardware) | `2ecf828b64e4`: 61,820 differ, max 0.004 | `15d1462d3f1b`: 141,136 differ, max 0.0078 |

The hashes are content hashes: `packckpt.content_sha256` of the saved tensors.

- **Same settings, same machine:** bitwise identical, in fp32 and bf16.
- **fp32:** unchanged by any instruction-set restriction tried here. This is consistent with fp32 runs
  reproducing on another AVX-512 host, but it was not tested on one: only this CPU was available.
- **bf16:** depends on the CPU's features. Disabling AMX, or bf16 instructions, changes the result.
  - A bf16 round of the loop is therefore reproducible only on a host with the same oneDNN path.
- **File hashes:** identical when the checkpoint stores no path (the pretrained models). A compact
  checkpoint records its base's absolute path, so its file hash changes with the folder even when every
  tensor is identical.
  - Compare checkpoints by content hash, as packs and `ycode.lm.ckptstore` do.
  - Round 54's identity is a file hash, but round 54 is a plain checkpoint, which stores no base path.
- **Not tested:** thread count. The trainer sets torch's threads to the CPU count (4 here, 4 in every record),
  so `OMP_NUM_THREADS` had no effect.

Rerun from the repository root:

```bash
python experiments/recovery/repro/setup_data.py /dev/shm/repro/in
python experiments/recovery/repro/run_variant.py /dev/shm/repro/in /dev/shm/repro/out/fp32-a fp32
ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16 python experiments/recovery/repro/run_variant.py /dev/shm/repro/in /dev/shm/repro/out/bf16-no-amx bf16
```

## Durable checkpoint storage: Hugging Face (investigated, nothing created)

**The account:** `Yazhiniyan`, a free plan, connected through the Hugging Face connector.
- The connector's credential is a short-lived OAuth token (expiring 2026-11-03 08:44 UTC).
- Its scopes include `contribute-repos`, but its tools here only read (list, cat, search). It cannot
  upload, and code in the container cannot use it.

**From the container:** `huggingface.co` is blocked by the environment's network policy
(`CONNECT tunnel failed, response 403`), and no Hugging Face token is configured.

**Limits** (`hub/storage-limits.md`):
- A free account gets 100 GB of private storage; PRO gets 1 TB.
- Files are recommended under 200 GB, with a hard limit of 500 GB; fewer than 100k files per repository and
  10k per folder.
- Repositories are git-versioned, and large files are stored with Xet chunk deduplication.
- The whole chain is about 8.3 GB: round 54's model.pt (5.2 GB), C2 and R1 (1.32 GB each) and S1 (0.42 GB).
  That fits the free tier.

**Integrity:**
- The Hub identifies large files by their SHA-256 (`hf_hub_download`'s cache), and a revision pins a
  commit.
- Downloads should still be checked against our own records: pack content hashes, and the base's file
  SHA-256.

**Delta dependencies:** a pack is useless without its parents and base.
- Store them in the same private repository, with a manifest (name, content hash, file SHA-256, parent hash,
  base hash).
- Download a pinned revision into `YCODE_CHECKPOINT_DIR`, where `ycode.lm.ckptstore` resolves and checks the
  whole chain (PR #26).

**To enable it, each needing your decision:**
1. Allow `huggingface.co` in the environment's network access.
2. Add a fine-grained token, write-scoped to one repository, as an environment secret.
3. Approve creating the private repository.
4. Approve a round-trip test with a small file before any checkpoint is uploaded.

## Blockers

1. **No copy of round 54 or `lmwork` is known to exist.** If one exists anywhere, it recovers the chain:
   `ycode.lm.ckptstore` checks it against the recorded hashes. Otherwise the S1-based V13 plan cannot run
   as registered.
2. **A new baseline would need its own pre-registration.** Candidates are YCode-LM v5 from git, or a new
   round-54-like model trained and stored durably. Another checkpoint must not stand in for S1.
3. **Durable storage** needs the network, token and repository decisions above.
4. **Replay data:** whether the v5 SFT data can be regenerated exactly is unverified; it can be checked
   against a recorded run dataset's hashes.
