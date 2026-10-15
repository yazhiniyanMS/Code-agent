"""Generalisation check for a checkpoint: answer loss on training rows, held-out rows, unseen tasks.

Low training loss alone says nothing about memorisation. This compares, for one checkpoint, the
mean per-token loss on answer tokens (every answer token counts once: the training weights are
not applied) of:

* ``train``: rows from the train split of a data directory (``SFTData``'s split). They are rows the
  checkpoint trained on **only if** it was trained on this data directory: pass ``trained_on``,
  which is recorded in the output; otherwise they are just more unseen rows of the same
  generators;
* ``heldout_rows``: that data's validation split: never trained on, drawn from the same generators,
  so it measures fit to the training distribution;
* ``unseen_tasks``: every dev problem with its reference solution (concepts the generators never
  produce); ``unseen_dev_clean`` leaves out the dev items v5's checkpoints saw in equivalent form;
  ``unseen_new``: the unseen-concept set (``ycode.lm.unseen``).

All splits are encoded the same way (``encode_example`` for unseen tasks, the stored encoding for
data rows), use the same target shift and answer mask, and report the token-weighted mean
(``token_loss``) with a bootstrap 95% interval over examples, the per-example mean, and the loss
per row type (write, bug fix, explain, docstring, ...), because a split's mix of row types moves
its loss: prose answers (explanations of library code) cost far more than code answers.

usage: python -m ycode.lm.diagnose MODEL_DIR DATA_DIR [--n 64] [--trained-on] [--json OUT]
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


def _token_loss(pairs) -> float:
    tokens = sum(n for _, n in pairs)
    return sum(s for s, _ in pairs) / max(1, tokens)


def _summary(pairs: list[tuple[float, int]], kinds: list[str] | None = None, *, seed: int = 0) -> dict:
    per_example = [s / n for s, n in pairs if n]
    mean = sum(per_example) / max(1, len(per_example))
    sd = math.sqrt(sum((v - mean) ** 2 for v in per_example) / max(1, len(per_example) - 1)) if per_example else 0.0
    rng = random.Random(seed)
    boots = sorted(_token_loss([pairs[rng.randrange(len(pairs))] for _ in pairs]) for _ in range(1000)) if pairs else []
    out = {"examples": len(pairs), "answer_tokens": sum(n for _, n in pairs), "token_loss": round(_token_loss(pairs), 4),
           "token_loss_ci95": [round(boots[25], 4), round(boots[974], 4)] if boots else None,
           "example_loss_mean": round(mean, 4), "example_loss_sd": round(sd, 4)}
    if kinds:
        by_kind: dict[str, list] = {}
        for k, p in zip(kinds, pairs):
            by_kind.setdefault(k, []).append(p)
        out["by_kind"] = {k: {"examples": len(v), "token_loss": round(_token_loss(v), 4)} for k, v in sorted(by_kind.items())}
    return out


def unseen_task_rows(tok, split: str = "dev", *, clean: bool = False) -> list[tuple[list[int], list[int]]]:
    """Held-out problems of ``split`` ("dev" or "unseen"), each answered with its reference
    solution, encoded like training rows. ``clean`` drops the contaminated items."""
    from ycode.lm.data import InstructionExample, encode_example
    from ycode.lm.devset import dev_problems
    from ycode.lm.leakage import CONTAMINATED_ITEMS, held_out_items
    from ycode.lm.unseen import unseen_problems

    refs = {i.name: i.reference for i in held_out_items() if i.split == split}
    problems = dev_problems() if split == "dev" else unseen_problems()
    return [encode_example(tok, InstructionExample(p.prompt, f"```python\n{refs[p.name]}\n```"), weighted=True)
            for p in problems if p.name in refs and not (clean and (split, p.name) in CONTAMINATED_ITEMS)]


def _row_kind(tok, ids: list[int]) -> str:
    from ycode.lm.basics import replay_kind
    from ycode.lm.tokenizer import ASSISTANT, USER

    prompt = tok.decode(ids).split(ASSISTANT, 1)[0].replace(USER, "").strip()
    return replay_kind(prompt)


def split_losses(model_dir, data_dir, *, n: int = 64, seed: int = 0, device: str = "cpu",
                 val_fraction: float = 0.05, trained_on: bool = False) -> dict:
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

    sets = {"train": rows(index[n_val:]), "heldout_rows": rows(index[:n_val])}
    for name, split, clean in (("unseen_tasks", "dev", False), ("unseen_dev_clean", "dev", True),
                               ("unseen_new", "unseen", False)):
        sets[name] = [(ids[:block], w[:block]) for ids, w in unseen_task_rows(lm.tokenizer, split, clean=clean)]
    out = {"model": str(model_dir), "data": str(data_dir), "seed": seed, "trained_on": trained_on,
           "note": None if trained_on else "the checkpoint was not trained on this data: 'train' rows are unseen rows too"}
    for name, examples in sets.items():
        pairs = [_example_loss(lm.model, ids, w, lm.device, lm.precision()) for ids, w in examples]
        out[name] = _summary(pairs, [_row_kind(lm.tokenizer, ids) for ids, _ in examples], seed=seed)
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
    ap.add_argument("--trained-on", action="store_true", help="the checkpoint was trained on DATA's train split")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    result = split_losses(args.model, args.data, n=args.n, seed=args.seed, device=args.device, trained_on=args.trained_on)
    text = json.dumps(result, indent=1)
    if args.json:
        args.json.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
