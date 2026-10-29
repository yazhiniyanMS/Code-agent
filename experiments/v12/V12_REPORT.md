# YCode V12: breaking the generalization ceiling

Status: **in progress.** P is complete; L was killed by container restarts; S1's first seed passes the
second-seed gate, and its second seed is blocked on storage (section 5). Round 54 remains the validated
checkpoint of record.

## 1. Starting point (verified 2026-10-28)

- **Every checkpoint exists and matches its recorded SHA-256.** Re-hashed after a container
  restart, compared with `experiments/v11/artifacts/CHECKPOINTS.json`:

  | Checkpoint | File SHA-256 | Role |
  | --- | --- | --- |
  | Round 54 `model.pt` | `2538b55d5aee3718...` | validated best, combined 29 |
  | Round 54 `optim.pt` | `bf51203b1f882b80...` | needed to train from round 54 |
  | C2 pack | `931ae406e1ff3f34...` | combined 31; parent of R1 and of every V12 run |
  | R1 pack | `eb7ec59511019869...` | strongest experimental candidate, combined 33, not validated |
  | R6, R7, V11-A, V11-B, V11-C packs | as recorded | - |

- **C2 was restored** to the RAM disk. The unpack checks every tensor against the pack's content
  hash.
- **R1 was restore-checked in V11** in a clean process: tensors bit-identical, all scores equal.
- **Hardware:**
  - 4 Xeon cores (AVX-512, no bf16 units), 15.7 GB RAM, a 16 GB RAM disk.
  - 0.67 GB free disk, so no new 1.3 GB pack fits.
  - Training runs at about 2.2 min per step of 4,096 tokens. Evaluation takes about 95 min.
- **The model is the decoder-only YCode-LM v5** (1.52B parameters), not an encoder-decoder.

## 2. What was tested and why

The choice of runs is in `experiments/v12/PLAN.md`, written before any V12 result, with one
amendment written after P and before L. The gates are V6/V10's, unchanged.

- **The dominant failure:** in 38 of round 54's 68 clean failures, the model writes the trained
  function whose description shares the most words with the prompt (V7).
- **Hypothesis C had never been tested at a learning rate that learns anything.** V7's only test of
  contrast ran at 1.5e-5, where nothing is learned; V7 itself found 1e-4 necessary. It was therefore
  the largest gap in the evidence per CPU hour.
- **Hypotheses A and B are largely settled.** Replay prevents forgetting only of what it contains
  (V8, V11). Diversity helps inside a grammar but not across families (V8).

## 3. Run P: contrastive minimal pairs at lr 1e-4 (hypothesis C)

**Setup.** V10's pre-registered R3 arm, never run before (`experiments/v12/v12_p.json`):

- R1's recipe from C2: the same 350 grammar v2 programs with 2 examples each, data seed 1302, 50%
  uniform replay, 36 steps, lr 1e-4 warmup-stable-decay, top 24 layers, seed 1301.
- Every example is written next to a minimal-pair sibling (exactly one part different, a different
  result), and the answer tokens that differ from the sibling's code weigh 20.
- All 700 curriculum answers were executed on their programs' inputs before training and passed.

**Result** (same pipeline and item sets as every earlier run):

| | Round 54 | C2 | R1 (same seeds) | V11-A | **V12-P** |
| --- | --- | --- | --- | --- | --- |
| Dev clean /44 | 27 | 28 | 30 | 30 | **26** |
| Unseen-v1 clean /53 | 2 | 3 | 3 | 1 | **2** |
| **Combined /97** | 29 | 31 | 33 | 31 | **28** |
| Neighbour probe /38 | 4 | 2 | 2 | 3 | **2** |
| New constants, trained families /24 (all 40 inputs) | 0 | 1 | 4 | 2 | **0** |
| Training-program reproduction /40 | 0 | 26 | 36 | 38 | **38** |
| Held-out combinations /60 | 0 | 32 | 54 | 51 | **50** |
| Held-out families /66 | 2 | 1 | 2 | 3 | **3** |
| Benchmark official · clean · robust (never used to select) | 22 · 19 · 20 | 21 · 18 · 20 | 23 · 20 · 23 | 27 · 24 · 24 | **29 · 25 · 27** |
| Failures: wrong function retrieved (V7 classes) | 38 | 28 | 31 | 33 | **37** |

- **Contrast weighting increased the failure it targets.** It produced 6 more wrong-function
  retrievals than R1 with the same seeds and programs, and 2 more neighbour-probe answers that write
  the trained neighbour (9 against 8).
- **It sharpened reproduction of the trained programs (38/40) and cost everything that measures
  generalization:**
  - new constants fell from 4 to 0;
  - held-out combinations fell from 54 to 50;
  - dev fell from 30 to 26.
- **On dev, P passes nothing R1 fails** and fails four items R1 passes: the `index_of_min` bug fix,
  and `first_negative`, `first_three_chars` and `longest_word` as written functions.
- **Gates:** combined 28 < 32. P fails the second-seed gate and is not a candidate.
- **The benchmark is the highest of any model.** It is reported only, and with dev and unseen-v1
  both lower it is not evidence of a better model.
- **Cost:**
  - 103 min of training. My concurrent test runs slowed it; about 80 min unloaded.
  - 207 min in total, a peak of 11,061 MB of RAM, 147,456 trained tokens.
  - Checkpoint `12b0dcc3a716...`, pack content `8c1cb2d6cf22...`.
  - The pack was kept on the RAM disk only; it is reproducible from the spec.

**Conclusion.** Weighting the tokens where an answer differs from a sibling teaches the model to
reproduce trained programs more exactly, not to read the decisive word. Hypothesis C in this form is
refuted at an effective learning rate.

## 4. Run L: R1 + string-literal bug fixes

**Container restarts and determinism.** The container restarted at about 23:05 while L was training
(step 24 of 42).

- The restart wiped the RAM disk, which held L's working checkpoint and P's pack.
- L was relaunched unchanged. Its dataset was rebuilt byte-identical (181,270 tokens).
- Its per-step training losses and gradient norms equal the first attempt's to every logged digit
  (steps 1-7, for example step 6: loss 0.02959, gradient norm 0.061001).
- **So a V12 run is reproducible from its spec and seeds alone.** That is the fallback while the disk
  (0.59 GB free) cannot hold a 1.3 GB pack.
- P's pack is gone with the RAM disk. P failed its gates, its records and hashes are committed, and
  rebuilding it costs about 80 min.

L_SECTION

## 5. Comparison, decision and next experiment

DECISION_SECTION
