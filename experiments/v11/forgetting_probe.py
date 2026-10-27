"""V11: why are `remove_spaces` (written and as a bug fix) and the `longest_word` bug fix forgotten?

usage: forgetting_probe.py OUT.json NAME=MODEL_DIR:DEV_ANSWERS.jsonl [...]

For each of the three dev items round 54 passes and C2/R1 fail, and for each model: the summed and
per-token log-probability of the correct answer, of the model's own (greedy) answer from its dev
record, and for the bug fixes of the unchanged buggy code (the "echo"); and the first token where the
correct answer and the model's answer part, with both tokens' probabilities there. Diagnosis only:
the items are read, never trained on, and nothing here feeds a training set.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FIX = {  # the correct answer for each item (the official tests' intent)
    ("write", "remove_spaces"): "def remove_spaces(s):\n    return s.replace(' ', '')",
    ("bugfix", "remove_spaces"): "def remove_spaces(s):\n    \"\"\"Return s with all spaces removed.\"\"\"\n    return s.replace(' ', '')",
    ("bugfix", "longest_word"): ("def longest_word(text):\n    \"\"\"Return the longest word in text (the first one if there "
                                 "is a tie).\"\"\"\n    best = ''\n    for word in text.split():\n        if len(word) > "
                                 "len(best):\n            best = word\n    return best"),
}


def answer_logprobs(model, tok, prompt: str, code: str) -> tuple[list[str], list[float]]:
    import torch
    import torch.nn.functional as F

    from ycode.lm.data import InstructionExample, encode_example

    ids, mask = encode_example(tok, InstructionExample(prompt, f"```python\n{code}\n```"))
    x, y = torch.tensor([ids[:-1]]), torch.tensor([ids[1:]])
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):  # bf16 weights, as evaluation runs
        logits, _ = model(x, y)
    logp = F.log_softmax(logits[0].float(), -1)
    out_t, out_p = [], []
    for j, (t, m) in enumerate(zip(ids[1:], mask[1:])):
        if m > 0:
            out_t.append(tok.decode([t]))
            out_p.append(float(logp[j, t]))
    return out_t, out_p


def main() -> None:
    import torch

    from ycode.lm.evalsets import eval_items
    from ycode.lm.tokenizer import BPETokenizer
    from ycode.lm.train import load_checkpoint

    out_path, specs = sys.argv[1], sys.argv[2:]
    items = {(i.suite, i.name): i for i in eval_items("dev")}
    tok = BPETokenizer.load("/home/user/lmwork/data/tokenizer.json")
    result = {}
    for spec in specs:
        name, rest = spec.split("=", 1)
        model_dir, answers = rest.split(":", 1)
        given = {(r["suite"], r["name"]): r for r in map(json.loads, Path(answers).read_text().splitlines())}
        model, _, _ = load_checkpoint(Path(model_dir), dtype=torch.bfloat16)
        model.eval()
        result[name] = {}
        for key, fix in FIX.items():
            it = items[key]
            own = given[key].get("code") or ""
            cands = {"correct": fix, "own": own}
            if key[0] == "bugfix":
                cands["echo"] = re.search(r"```python\n(.*?)\n```", it.prompt, re.S).group(1)
            scored = {k: answer_logprobs(model, tok, it.prompt, c) for k, c in cands.items() if c}
            row = {k: round(sum(p), 3) for k, (_, p) in scored.items()}
            row["own_passes"] = bool(given[key]["pass"])
            row["own_code"] = own
            # where the correct answer and the model's own answer part
            ct, cp = scored["correct"]
            ot, op = scored.get("own", ([], []))
            j = next((i for i, (a, b) in enumerate(zip(ct, ot)) if a != b), None)
            if j is not None:
                row["diverge"] = {"after": "".join(ct[:j])[-40:], "correct_token": ct[j], "correct_logp": round(cp[j], 3),
                                  "own_token": ot[j], "own_logp": round(op[j], 3)}
            result[name][f"{key[0]}:{key[1]}"] = row
            print(name, key, {k: v for k, v in row.items() if k not in ("own_code",)}, flush=True)
        del model
    Path(out_path).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
