# V12 plan and decision rules (written before any V12 result)

## Starting point (verified 2026-10-28, after a container restart)

- **Every checkpoint V12 depends on exists and matches its recorded SHA-256** (re-hashed, compared
  with `experiments/v11/artifacts/CHECKPOINTS.json`):
  - round 54 (`model.pt` `2538b55d...`, `optim.pt` `bf51203b...`);
  - the C2, R1, R6 and R7 packs;
  - the V11-A, B and C packs;
  - the tokenizer.
- **C2 was restored to the RAM disk.** The unpack checks every tensor against the pack's content
  hash. R1 was restored and re-evaluated in a clean process in V11: tensors bit-identical, all
  scores equal.
- **Validated best: round 54.** Combined clean dev + unseen-v1 = 29/97.
- **Strongest experimental candidate: R1.** Combined 33, not validated: 3 instruction-following
  losses, and its second seed scored 31. **C2, R1's parent,** scores 31.
- **Hardware:**
  - 4 cores (Xeon 2.8 GHz, AVX-512, no bf16 units), 15.7 GB RAM, a 16 GB RAM disk.
  - 0.67 GB free on disk, too little for one more 1.3 GB lossless pack.
  - Measured in V11 on this host type: training about 2.2-2.5 min per step (4 × 1,024 tokens),
    so a 36-step run is about 80 min. The full evaluation (dev, unseen-v1, benchmark, five task
    sets) takes about 95 min.
- **The model** is the decoder-only YCode-LM v5 (1.52B parameters; not an encoder-decoder).

## Choosing the experiment: the V6-V11 evidence on each hypothesis

| Hypothesis | Already tested | What is known | Information still missing |
| --- | --- | --- | --- |
| A. Replay prevents forgetting under stronger fine-tuning | V8 (replay shares), V10-R7 (75%), V11-B (weighted) | Replay prevents most forgetting when it adds tokens; it cannot restore skills it does not contain (V11) | little |
| B. Diverse verified examples beat repeated variations | V8 COMP vs MORE (2 seeds, p < 0.05) | true within a grammar; no cross-family transfer | little |
| C. Contrastive examples make the model respect decisive differences | V7 P1 **only at lr 1.5e-5, where nothing is learned** (V7: "the real bottleneck is learning strength"); V10-R3 built and unit-tested, never run | the dominant failure: 38 of round 54's 68 clean failures write the trained function whose wording is closest | **untested at an effective learning rate** |
| D. Staged learning rate | C2 (1e-4) then R1 (1e-4); R4 (3e-5) | 3e-5 keeps more but learns less | partial |
| E. Compositional training | V8-2b, V9 (grammar v2), V10-R2/R6 (grammar v3) | in-grammar combinations: 54/60; R6 reached the best unseen-v1 (5/53) on one seed | whether R6's unseen-v1 gain replicates |

**First run: P = hypothesis C at an effective learning rate.**

- It targets the dominant failure class directly, and its only previous test could not have worked.
- It costs one run of R1's size, about 3 h with full evaluation.
- It is item-for-item comparable with R1: the same programs, data seed, steps and start; only the
  contrast changes.
- Its arm is V10's pre-registered R3, so the configuration was fixed before any of its results.

## Run P (`experiments/v12/v12_p.json`)

- **Start:** C2 (content hash `cc2f2fa0...`).
- **Seeds:** train 1301, data 1302, the same as R1.
- **Configuration:** R1's 350 grammar v2 programs with 2 examples each, each written next to a
  minimal-pair sibling.
  - The sibling has the same parameters, exactly one part different (max/min, first/last,
    odd/even, ...), and a different result on the program's inputs. It is the example's contrast.
  - The answer tokens that differ from the sibling's code weigh 20.
  - 50% uniform replay of round 54's production mix.
  - 36 steps, lr 1e-4 warmup-stable-decay to 1e-5, top 24 layers, fp32.
- **Example validation:** every curriculum answer is executed on its program's 40 inputs and must
  reproduce the reference outputs before training (`rounddata.validate_curriculum`). All 700 of
  P's pair examples pass.
- **Expected if C holds:**
  - neighbour probe above 4/38, where no run since round 54 has been;
  - fewer "wrong function retrieved" failures on dev + unseen-v1;
  - held-out combinations (c2_test) at or above R1's 54/60.
- **Risk:** weight 20 on the differing tokens can distort other tokens. If it fails, it fails on the
  same items as R1, so the comparison is readable either way.

## Gates (the established V6/V10 rules, unchanged)

- **Selection metric:** clean dev /44 + clean unseen-v1 /53 = combined /97.
- **Second-seed gate:** combined >= 32, and no failure category loses 3 or more of round 54's clean
  official passes (`items_lost`).
- **Promotion** (replaces round 54):
  - every seed beats 29;
  - mean combined >= 32;
  - mean `items_lost` < 3 in every category.
- **Transfer evidence**, reported because P targets transfer, never a substitute for the gates:
  - the neighbour probe /38 (round 54: 4);
  - new constants within trained families, family_new /24, strict on all 40 inputs;
  - held-out families /66;
  - the V7 failure-mechanism counts.
- **Reported, never used to choose:** the benchmark (official, clean, robust), training-program
  reproduction (c2_trained), held-out combinations (c2_test).
- **unseen-v2 (71 tasks)** is not evaluated or read in V12. It stays reserved for the authorized
  final protocol.
- **A known flaw, documented and not changed:** two of R1's three blocking losses are round 54
  weak passes (its own code fails the robust tests). Changing the rule now would promote R1 after
  its results are known, so the rule stays as it is.

## What happens next

1. **If P passes the second-seed gate:** P at seed 1302 (data 1303), then promotion.
2. **If P shows a transfer signal without passing** (neighbours >= 6/38, or at least 5 fewer
   wrong-function failures than R1): keep C and add the missing retention in the next run.
3. **Otherwise C is refuted at an effective learning rate.** The next run is E: R6's recipe at seed
   1302 / data 1303, to test whether cross-family compositions' unseen-v1 gain (5/53) is real.

## Storage

- A candidate stays on the RAM disk during its evaluation. Its lossless pack and content hash are
  made there.
- Training is deterministic from the recorded spec and seeds: C2's rebuild was bit-identical. A
  container restart therefore costs a re-run, not the result.
- A durable on-disk pack needs about 1.4 GB, and the disk has 0.67 GB. It is made only for a
  candidate that passes the second-seed gate, after your approval to free space. No checkpoint is
  deleted without that approval.
