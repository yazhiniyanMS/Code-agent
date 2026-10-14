"""Behavioural leakage audit of one round's fine-tuning data (production mix of build_round.py).

usage: leakage_audit.py SEED [--no-skills]
Builds the round's examples exactly like /home/user/lmwork/build_round.py (same generators, same
replay pool and filter), then runs every answer function against every benchmark and dev item with
the same arity (ycode.lm.leakage). Generated code runs in-process; replayed library code runs in a
subprocess per check.
"""
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import ycode.lm.compose as compose  # noqa: E402
from ycode.lm.basics import basics_examples, write_basics_dataset  # noqa: E402
from ycode.lm.leakage import answer_functions, behaviour_leaks, held_out_items  # noqa: E402
from ycode.lm.tokenizer import ASSISTANT, END, BPETokenizer  # noqa: E402

W = Path("/home/user/lmwork")


def main() -> None:
    seed = int(sys.argv[1])
    compose.INCLUDE_SKILLS = "--no-skills" not in sys.argv
    examples = (basics_examples(seed, write_per_concept=6, bugfix_per_concept=4)
                + compose.compose_examples(seed, per_exercise=3)
                + compose.compose_bugfix_examples(seed, per_exercise=1)
                + compose.contrast_examples(seed) + compose.novel_examples(seed))
    generated = answer_functions(ex.response for ex in examples)
    tok = BPETokenizer.load(W / "data" / "tokenizer.json")
    with tempfile.TemporaryDirectory() as d:  # the replay rows this round would really get
        write_basics_dataset(d, W / "data" / "tokenizer.json", replay_dir=W / "data", replay=200, seed=seed,
                             examples=[], log=lambda *_: None)
        import numpy as np
        tokens = np.fromfile(Path(d) / "sft_tokens.bin", dtype=np.uint16)
        index = np.load(Path(d) / "sft_index.npy")
        replay_answers = []
        for start, length in index:
            text = tok.decode(tokens[start:start + length].tolist())
            replay_answers.append(text.split(f"{ASSISTANT}\n", 1)[-1].rsplit(f"\n{END}", 1)[0])
    replayed = {s: v for s, v in answer_functions(replay_answers).items() if s not in generated}
    items = held_out_items()
    t0 = time.time()
    leaks = behaviour_leaks(generated, items, near=0.95) + behaviour_leaks(replayed, items, isolate=True, near=0.95)
    report = {
        "seed": seed, "skills": compose.INCLUDE_SKILLS, "examples": len(examples),
        "generated_functions": len(generated), "replayed_functions": len(replayed),
        "held_out_items": dict(Counter(i.split for i in items)), "seconds": round(time.time() - t0, 1),
        "equivalent": [{"function": lk.function, "item": lk.item, "split": lk.split, "probes": lk.probes,
                        "source": lk.source} for lk in leaks if lk.agreement == 1.0],
        "near_equivalent": [{"function": lk.function, "item": lk.item, "split": lk.split,
                             "agreement": lk.agreement, "probes": lk.probes, "source": lk.source}
                            for lk in leaks if lk.agreement < 1.0],
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
