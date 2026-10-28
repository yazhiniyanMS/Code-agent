"""How often each training function body appears in one round of the v5 production mix.

usage: train_frequency.py OUT.json [SEEDS...]
Counts the generated examples (writes and bug fixes; replay excluded) whose answer, renamed to f
and without its docstring, equals a body; averaged over the given round seeds (default: the last
three rounds before round 54, seeds 1052-1054). Round 54's 54 rounds used this mix from round 22 on.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ycode.lm.leakage import normalized_body  # noqa: E402
from ycode.lm.rounddata import DEFAULTS, round_examples  # noqa: E402

out = Path(sys.argv[1])
seeds = [int(s) for s in sys.argv[2:]] or [1052, 1053, 1054]
bodies: Counter = Counter()
kinds: Counter = Counter()
for seed in seeds:
    for e in round_examples(seed, DEFAULTS):
        code = re.search(r"```python\n(.*?)\n```", e.response, re.S)
        if not code:
            continue
        try:
            key = normalized_body(code.group(1))
        except Exception:
            continue
        bodies[key] += 1
        kinds["bugfix" if e.prompt.startswith("This function has a bug") else "write"] += 1
    print("seed", seed, "done", flush=True)
out.write_text(json.dumps({"seeds": seeds, "examples": dict(kinds),
                           "per_round": {k: v / len(seeds) for k, v in bodies.items()}}, indent=0))
