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

## Amendment 1 (after run P's results, before run L starts; gates unchanged)

**P's result** (seed 1301; full tables in the V12 report). It fails, and shows no transfer.

| Measure | P | R1 (same seeds and programs) | Round 54 |
| --- | --- | --- | --- |
| Combined | 28 (dev 26, unseen-v1 2) | 33 | 29 |
| Neighbour probe | 2/38, the trained neighbour written instead 9 times | 2/38, 8 times | 4/38, 9 times |
| New constants within trained families | 0/24 | 4/24 | - |
| Training-program reproduction | 38/40 | 36/40 | - |
| Held-out combinations | 50/60 | 54/60 | - |

- Contrast weighting at lr 1e-4 sharpens reproduction of the trained programs and costs
  generalization. P passes nothing R1 fails and fails four dev items R1 passes.
- **Hypothesis C, in this form, is refuted at an effective learning rate.**

**This is branch 3 of the plan, and the order changes before the next run:**

- **Run L comes before E.**
- **E's precedent cannot pass the gate.** R6 scored dev 26, so even a fully replicated E lands near
  combined 31, under the gate's 32. E measures transfer; it is not a route to a better checkpoint.
- **L can.** Your stated priority is a measurably better checkpoint, and the closest recipe to the
  gates is R1's (33 and 31). It is blocked by 3 instruction-following losses.
  - Two of them are bug fixes that every post-round-54 model echoes.
  - One of those, `remove_spaces` (`'_'` → `''`), belongs to a class absent from round 54's
    training mix: no bug fix there restores a string literal.
- **L, `experiments/v12/v12_l.json`:** R1's recipe exactly, plus `literal_bugfix: 3`.
  - That adds 93 correct string-literal bug fixes from round 54's own training concepts
    (`basics.literal_bugfix_examples`, PR #15).
  - The literal to restore differs by task, so this is also a contrastive (C) intervention at the
    decisive token.
  - 42 steps: one epoch of a 181,270-token dataset, measured by a dry run.
  - Seeds 1301 / data 1302, so it is item-for-item comparable with R1.
- **Expected:** `bugfix:remove_spaces` restored, with R1's gains kept: combined at least 33 and 2
  instruction-following losses.
- **The known risk:** this class was found on dev, so a gain on that one item is weak evidence. It
  needs the second seed (1302 / data 1303) and an unchanged unseen-v1 before any claim, as the gates
  require anyway.
- **Next:**
  - If L passes the second-seed gate: L at seed 1302.
  - Otherwise: E.

## Amendment 2 (after two container restarts killed run L, before run S1; gates unchanged)

**What happened to L.**

- The container restarted at about 23:05 and again at about 23:49 on 2026-10-28, three restarts in
  under seven hours. Each one wiped the RAM disk.
- That killed run L twice, at steps 24 and 12 of 42. The second attempt's losses equalled the
  first's to every digit, so the runs are deterministic.
- A run of R1's size needs about 3 h of uninterrupted uptime.
- A durable resumable checkpoint (trained layers in fp32 plus optimizer state, about 6 GB) or even a
  lossless pack (1.3 GB) does not fit the 0.59 GB of free disk. Making room needs your approval to
  remove old checkpoints, which has not been given.
- **Under the rule against jobs that cannot finish or checkpoint safely, L is not relaunched** until
  storage allows it.

**New run S1: a short staged continuation from R1 (hypothesis D with L's data).**

- **Hypothesis:** R1's 36 steps at lr 1e-4 are the acquisition stage. A short second stage at a low
  learning rate can add the missing bug-fix skill without damaging R1's gains.
  - V10-R4 showed that 3e-5 retains more than 1e-4.
  - V11-B showed one of R1's lost bug fixes coming back from a small replay change.
- **Configuration (`experiments/v12/v12_s1.json`):**
  - from R1 (pack `eb7ec595...`, content `9909c511...`);
  - lr 3e-5 warmup-stable-decay, 7 steps (one epoch, measured by a dry run);
  - data: the 93 string-literal bug fixes (L's) plus uniform replay of round 54's production mix
    (data seed 1302), with no new curriculum.
- **Cost:** about 15 min of training. Short enough to survive the restart pattern, and deterministic
  if lost.
- **Screen first:**
  - dev + unseen-v1 (about 18 min) give the combined score and the category rule.
  - Only a screen that reaches combined >= 32 with no category losing 3 or more gets the remaining
    sets (benchmark, neighbours, new constants, reproduction, held-out sets) and then a second
    seed (1302 / data 1303, from V11-A, R1's second seed).
  - This saves about an hour per failing run and changes no gate: the gate depends on dev and
    unseen-v1 only.
- **Expected:** R1's 33 kept, plus at least one of the three instruction-following items restored.
- **Risk:** 7 steps without curriculum examples may wear away R1's grammar gains, which the dev
  screen measures directly.
