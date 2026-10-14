"""Generalisation check for a checkpoint: answer loss on training rows, held-out rows, unseen tasks.

Low training loss alone says nothing about memorisation. This compares, for one checkpoint, the
mean per-token loss on the answer tokens (unweighted) of:

* ``train``: rows the run trained on (``SFTData``'s train split of a data directory);
* ``heldout_rows``: that data's validation split: never trained on, but drawn from the same
  generators, so it measures fit to the training distribution;
* ``unseen_tasks``: the dev set's problems with their reference solutions, concepts the
  generators never produce.

A small train/held-out gap and a large held-out/unseen gap points to a model that has learned the
training distribution but does not transfer to new tasks; a large train/held-out gap points to
memorised rows.

usage: python -m ycode.lm.diagnose MODEL_DIR DATA_DIR [--n 64] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def _example_loss(model, ids: list[int], weights: list[int], device: str, autocast) -> tuple[float, int]:
    """Summed token loss over answer tokens (weight > 0, counted as 1) and their number."""
    import torch
    import torch.nn.functional as F

    x = torch.tensor([ids[:-1]], device=device)
    y = torch.tensor([ids[1:]], device=device)
    mask = torch.tensor([[1.0 if w > 0 else 0.0 for w in weights[1:]]], device=device)
    with torch.no_grad(), autocast:
        logits, _ = model(x, y)
    losses = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), reduction="none")
    return float((losses * mask.view(-1)).sum()), int(mask.sum())


def _summary(pairs: list[tuple[float, int]]) -> dict:
    total, tokens = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
    per_example = [s / n for s, n in pairs if n]
    mean = sum(per_example) / max(1, len(per_example))
    sd = math.sqrt(sum((v - mean) ** 2 for v in per_example) / max(1, len(per_example) - 1)) if per_example else 0.0
    return {"examples": len(pairs), "answer_tokens": tokens, "token_loss": round(total / max(1, tokens), 4),
            "example_loss_mean": round(mean, 4), "example_loss_sd": round(sd, 4)}


def unseen_task_rows(tok) -> list[tuple[list[int], list[int]]]:
    """The dev problems, each answered with its reference solution, encoded like training rows."""
    from ycode.lm.data import InstructionExample, encode_example
    from ycode.lm.devset import dev_problems
    from ycode.lm.leakage import held_out_items

    refs = {i.name: i.reference for i in held_out_items() if i.split == "dev"}
    return [encode_example(tok, InstructionExample(p.prompt, f"```python\n{refs[p.name]}\n```"), weighted=True)
            for p in dev_problems() if p.name in refs]


def split_losses(model_dir, data_dir, *, n: int = 64, seed: int = 0, device: str = "cpu",
                 val_fraction: float = 0.05) -> dict:
    import numpy as np

    from ycode.lm.generate import LocalLM

    lm = LocalLM(model_dir, device=device)
    data_dir = Path(data_dir)
    tokens = np.memmap(data_dir / "sft_tokens.bin", dtype=np.uint16, mode="r")
    mask = np.memmap(data_dir / "sft_mask.bin", dtype=np.uint8, mode="r")
    index = np.load(data_dir / "sft_index.npy")
    n_val = max(1, int(len(index) * val_fraction))  # the same split SFTData uses
    rng = random.Random(seed)
    block = lm.model.cfg.block_size

    def rows(idx) -> list[tuple[list[int], list[int]]]:
        chosen = rng.sample(list(idx), min(n, len(idx)))
        return [(tokens[s:s + length].tolist()[:block], mask[s:s + length].tolist()[:block]) for s, length in chosen]

    unseen = [(ids[:block], w[:block]) for ids, w in unseen_task_rows(lm.tokenizer)]
    sets = {"train": rows(index[n_val:]), "heldout_rows": rows(index[:n_val]), "unseen_tasks": unseen}
    out = {"model": str(model_dir), "data": str(data_dir), "seed": seed}
    for name, examples in sets.items():
        out[name] = _summary([_example_loss(lm.model, ids, w, lm.device, lm.precision()) for ids, w in examples])
    out["gap_train_to_heldout"] = round(out["heldout_rows"]["token_loss"] - out["train"]["token_loss"], 4)
    out["gap_heldout_to_unseen"] = round(out["unseen_tasks"]["token_loss"] - out["heldout_rows"]["token_loss"], 4)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("data")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    result = split_losses(args.model, args.data, n=args.n, seed=args.seed, device=args.device)
    text = json.dumps(result, indent=1)
    if args.json:
        args.json.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
