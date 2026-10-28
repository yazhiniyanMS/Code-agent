"""Where does learning stop? Code-only answer loss of one checkpoint on four sets of reference solutions.

usage: probe_learning.py MODEL_DIR OUT.json

* ``trained``: idiom-curriculum v1 programs (the ones a curriculum run trained on: did it learn them?);
* ``family_new``: new members of the same families (other constants, e.g. "after the first seven"),
  which no run trained on: did it learn the family's pattern rather than its instances?
* ``dev_clean``: the clean dev write items (near trained concepts): did fine-tuning forget?
* ``unseen_v1`` / ``unseen_v2``: the held-out unseen-concept references: did anything transfer?

Each item is a write prompt answered with the reference, encoded like training rows; the loss is
on the body tokens only (BODY_WEIGHT), so prompts, fences and names do not count. Lower is better;
comparing two checkpoints on the same items shows what fine-tuning changed.
"""
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ycode.lm import idioms  # noqa: E402
from ycode.lm.data import BODY_WEIGHT, InstructionExample, encode_example  # noqa: E402
from ycode.lm.diagnose import _example_loss, _token_loss  # noqa: E402

NEW_K = (5, 6, 2, 3, 1, 4, 7, 8, 9)  # the first two unused by a family, if the result is not degenerate


def family_new_programs() -> list:
    """Families parameterised by k alone, instantiated at constants none of their variants uses."""
    progs = []
    saved = idioms.FAMILIES
    try:
        for f in saved:
            if f.key not in idioms.FAMILIES_V1:  # fixed set: comparable across curriculum versions
                continue
            if not (f.variants and all(set(v) == {"k", "kw"} for v in f.variants)):
                continue
            used = {v["k"] for v in f.variants}
            for k in [k for k in NEW_K if k not in used][:2]:
                idioms.FAMILIES = (dataclasses.replace(f, variants=({"k": k, "kw": idioms.NUM.get(k, str(k))},)),)
                try:
                    progs += idioms.candidate_programs(seed=99)
                except AssertionError:  # degenerate on the family's domain (e.g. constant): skip
                    pass
    finally:
        idioms.FAMILIES = saved
    first = {}
    for p in progs:  # one name per behaviour
        first.setdefault(p.source(0, "f"), p)
    return list(first.values())


def items(seed: int = 0):
    trained, _ = idioms.held_out_check([p for p in idioms.candidate_programs() if p.family in idioms.FAMILIES_V1])
    seen = {}
    for p in trained:
        seen.setdefault(p.source(0, "f"), p)
    sets = {"trained": [(f"Write a Python function `{p.name}({p.params})` that {p.tasks[0]}", p.source(0))
                        for p in seen.values()],
            "family_new": [(f"Write a Python function `{p.name}({p.params})` that {p.tasks[0]}", p.source(0))
                           for p in family_new_programs()]}
    from ycode.lm.unseen import unseen_problems, unseen_references
    from ycode.lm.unseen2 import unseen2_problems, unseen2_references

    from ycode.lm.devset import dev_problems
    from ycode.lm.leakage import CONTAMINATED_ITEMS, held_out_items

    dev_refs = {i.name: i.reference for i in held_out_items() if i.split == "dev"}
    sets["dev_clean"] = [(p.prompt, dev_refs[p.name]) for p in dev_problems()
                         if p.name in dev_refs and ("dev", p.name) not in CONTAMINATED_ITEMS]
    for name, probs, refs in (("unseen_v1", unseen_problems(), unseen_references()),
                              ("unseen_v2", unseen2_problems(), unseen2_references())):
        sets[name] = [(p.prompt, refs[p.name]) for p in probs]
    return sets


def main() -> None:
    from ycode.lm.generate import LocalLM

    model_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    lm = LocalLM(model_dir, device="cpu", dtype="fp32")
    block = lm.model.cfg.block_size
    result = {"model": str(model_dir)}
    for name, rows in items().items():
        pairs = []
        for prompt, ref in rows:
            ids, w = encode_example(lm.tokenizer, InstructionExample(prompt, f"```python\n{ref}\n```"), weighted=True)
            pairs.append(_example_loss(lm.model, ids[:block], w[:block], lm.device, lm.precision(), only_weight=BODY_WEIGHT))
        per_item = [s / n for s, n in pairs if n]
        result[name] = {"items": len(rows), "code_token_loss": round(_token_loss(pairs), 4),
                        "item_loss_median": round(sorted(per_item)[len(per_item) // 2], 4)}
        print(name, result[name], flush=True)
    out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
