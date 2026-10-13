"""Average a finished round's weights with the anchor it started from (in place, into cur/model.pt).

usage: average.py ANCHOR_DIR CUR_DIR WEIGHT   (new = WEIGHT * cur + (1 - WEIGHT) * anchor)
Only trained (fp32) tensors differ between the two; frozen bf16 tensors are identical and kept.
Averaging consecutive fine-tuning checkpoints smooths round-to-round noise (items that pass in one
round and fail in the next)."""
import json
import sys
from pathlib import Path

import torch

anchor_dir, cur_dir, w = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
cur = torch.load(cur_dir / "model.pt", map_location="cpu", weights_only=True)  # fully in RAM
anchor = torch.load(anchor_dir / "model.pt", map_location="cpu", weights_only=True, mmap=True)
done = 0
with torch.no_grad():
    for k, v in cur["model"].items():
        a = anchor["model"].get(k)
        if a is not None and v.dtype == torch.float32 and a.shape == v.shape:
            v.mul_(w).add_(a.to(torch.float32), alpha=1 - w)
            done += 1
del anchor
(cur_dir / "model.pt").unlink()  # no room for a second copy on disk
torch.save(cur, cur_dir / "model.pt")
info = json.loads((cur_dir / "info.json").read_text())
info["averaged_with_anchor"] = w
(cur_dir / "info.json").write_text(json.dumps(info, indent=2))
print(f"averaged {done} tensors (weight {w} on the new round)")
