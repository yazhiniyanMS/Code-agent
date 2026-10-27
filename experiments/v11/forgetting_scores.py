"""V11-B: how much C2 forgot each example of round 54's own training mix (one production round).

usage: forgetting_scores.py SEED OUT.json [--before DIR] [--after DIR] [--spec SPEC --arm ARM] [--limit N]

For the production round that ``rounddata._legacy_sample`` draws replay from at this seed (the same
examples, in the same order), the weighted answer-token loss (the training objective: prompt tokens
weigh 0, a bug fix's changed tokens 8) under the model before the curriculum (round 54) and after it
(C2), one model at a time, bf16, no gradients. forgetting = loss(after) - loss(before). The replay
weight of an example is max(forgetting, 0) plus a floor (5% of the mean positive forgetting), so
every example can still be drawn. Only round 54's training mix is scored: no dev, unseen-v1,
unseen-v2 or benchmark item is read.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def example_losses(model_dir: str, examples: list, tok, log=print) -> list[float]:
    import torch
    import torch.nn.functional as F

    from ycode.lm.data import encode_example
    from ycode.lm.train import load_checkpoint

    model, _, _ = load_checkpoint(Path(model_dir), dtype=torch.bfloat16)
    model.eval()
    block = model.cfg.block_size
    out, t0 = [], time.time()
    with torch.no_grad():
        for n, e in enumerate(examples):
            ids, w = encode_example(tok, e, weighted=True)
            ids, w = ids[:block + 1], w[:block + 1]
            x = torch.tensor([ids[:-1]])
            y = torch.tensor([ids[1:]])
            wt = torch.tensor(w[1:], dtype=torch.float32)
            logits, _ = model(x, y)  # (logits, loss), as in ycode.lm.diagnose._example_loss
            nll = F.cross_entropy(logits[0].float(), y[0], reduction="none")
            out.append(float((nll * wt).sum() / wt.sum().clamp(min=1)))
            if n % 200 == 0:
                log(f"  {model_dir}: {n}/{len(examples)} ({time.time() - t0:.0f}s)")
    del model
    return out


def main() -> None:
    from ycode.lm import rounddata
    from ycode.lm.tokenizer import BPETokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("out")
    ap.add_argument("--before", default="/home/user/lmwork/loop/anchor")
    ap.add_argument("--after", default="/dev/shm/ycode-keep/C2/1301/model_bf16")
    ap.add_argument("--spec", default=str(ROOT / "experiments" / "v10" / "v10_r1.json"))
    ap.add_argument("--arm", default="R1")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N examples (a smoke test)")
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    arm = next(a for a in spec["arms"] if a["name"] == args.arm)
    o = {**rounddata.DEFAULTS, **arm["data"]}
    production = rounddata._legacy_examples(args.seed, {**o, **rounddata.PRODUCTION})
    examples = production[: args.limit] if args.limit else production
    tok = BPETokenizer.load(spec["tokenizer"])
    print(f"seed {args.seed}: {len(production)} production examples, scoring {len(examples)}", flush=True)
    before = example_losses(args.before, examples, tok, log=lambda m: print(m, flush=True))
    after = example_losses(args.after, examples, tok, log=lambda m: print(m, flush=True))
    forgetting = [a - b for a, b in zip(after, before)]
    pos = [f for f in forgetting if f > 0]
    floor = 0.05 * (sum(pos) / len(pos)) if pos else 1.0
    weights = [max(f, 0.0) + floor for f in forgetting]
    if args.limit:  # a partial file cannot drive sampling: rounddata checks n against the round
        weights = None
    Path(args.out).write_text(json.dumps({
        "seed": args.seed, "n": len(production), "scored": len(examples), "before": args.before, "after": args.after,
        "spec": args.spec, "arm": args.arm, "floor": floor, "weights": weights,
        "loss_before": before, "loss_after": after, "forgetting": forgetting,
        "tags": [rounddata._legacy_tag(e) for e in examples],
        "prompts_head": [e.prompt[:120] for e in examples]}))
    print(f"wrote {args.out}: mean forgetting {sum(forgetting) / len(forgetting):+.4f}, "
          f"{len(pos)}/{len(forgetting)} examples forgotten (loss rose)", flush=True)


if __name__ == "__main__":
    main()
