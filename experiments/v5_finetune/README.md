# YCode-LM v5 fine-tuning: loop, experiments and ablations

The scripts that fine-tune YCode-LM v5 on a CPU in rounds, decide which rounds to keep, and test
whether a change helps. They ran from `/home/user/lmwork` (paths are hard-coded there); the
model code they call is the `ycode.lm` package in this repository.

| File | What it does |
| --- | --- |
| `until_target2.py` | The round loop: build data, train 72 steps from the current anchor, evaluate on the dev set (decisions) and the benchmark (report only), analyse errors, accept or reject. Stops when the anchor's benchmark reaches the target (55/65). |
| `build_round.py` | The data mix of one round (run in a fresh process, so generator edits apply without a restart). |
| `policy.json` | Acceptance tolerances, steps per round and round averaging; re-read every round. |
| `average.py` | Blends a finished round's weights with its anchor's (smooths round-to-round flips). |
| `precbench.py` | fp32 vs bf16 speed on the current CPU (the loop picks the faster one per host). |
| `ablate.py` | Ablation runs: several arms from the same checkpoint, one change each, same evaluation. |
| `ablations/abl1.json` | First ablation: skill families and round averaging, against two baseline seeds. |
| `expA/` | Experiment A, a reproducible baseline: one checkpoint evaluated three times, with its hash, tokenizer hash, decoding settings, seed and evaluator version (`run_expA.py`, `report.py`); the reconstruction of round 53's rejection (`round53.py`, `round53.txt`). |

## Test sets

- **Benchmark** (`ycode.lm.evaluate`): 30 standard problems, 20 fresh problems, 15 bug fixes. Never
  trained on and never used to choose a checkpoint; reported for every accepted round.
- **Dev set** (`ycode.lm.devset`): 47 write problems and 15 bug fixes, held out of training by
  concept. Used to accept or reject rounds and to decide ablations.

## Error analysis

`ycode-lm analyze ANSWERS.jsonl --set bench|dev [--compare OLD.jsonl]` runs each failed answer
against every test assertion and puts the failure in one category (formatting, instruction
following, hallucination, coding, factual, reasoning). On round 54's benchmark answers,
instruction following cost 23 of the 43 lost points: the model writes the body of the most
similar training function (`to_upper` -> `to_lower`'s body, `celsius_to_fahrenheit` ->
`celsius_to_kelvin`'s).
