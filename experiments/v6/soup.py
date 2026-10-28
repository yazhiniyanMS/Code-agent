"""Average the weights of fine-tuned checkpoints that share a starting point (a "model soup").

usage: soup.py OUT_DIR MODEL_DIR[:WEIGHT] MODEL_DIR[:WEIGHT] ...
Every checkpoint must come from the same base (round 54) with the same architecture; floating-point
tensors are averaged in fp32 with the given weights (default equal) and stored in the first
checkpoint's dtype, everything else (config, tokenizer) is taken from the first. Base-referenced
inputs (ycode.lm.train.export_checkpoint(base=...)) are merged with their base first; if the first
input is base-referenced, so is the result (only tensors that differ from the base are written).
Writes model.pt and info.json (sources, weights, SHA-256 of the result).
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ycode.lm.train import base_state  # noqa: E402

out = Path(sys.argv[1])
parts = [a.rsplit(":", 1) if ":" in a and a.rsplit(":", 1)[1].replace(".", "").isdigit() else (a, None) for a in sys.argv[2:]]
dirs = [Path(d) for d, _ in parts]
weights = [float(w) if w else 1.0 for _, w in parts]
weights = [w / sum(weights) for w in weights]
payloads = [torch.load(d / "model.pt", map_location="cpu", weights_only=True, mmap=True) for d in dirs]
for p in payloads:
    if p.get("base"):
        p["model"] = {**base_state(p), **p["model"]}
base = payloads[0]
reference = base.get("base")
keys = base["model"].keys()
assert all(p["model"].keys() == keys and p["config"] == base["config"] for p in payloads[1:]), "checkpoints differ in shape"
averaged = {}
with torch.no_grad():
    for k in keys:
        v = base["model"][k]
        if v.is_floating_point():
            acc = sum(w * p["model"][k].to(torch.float32) for w, p in zip(weights, payloads))
            averaged[k] = acc.to(v.dtype)
        else:
            averaged[k] = v
if reference:  # write only what differs from the base, like the inputs
    frozen = base_state({"base": reference})
    averaged = {k: v for k, v in averaged.items()
                if not (k in frozen and frozen[k].shape == v.shape and torch.equal(frozen[k].to(v.dtype), v))}
payload = {**base, "model": averaged, "soup": {"sources": [str(d) for d in dirs], "weights": weights}}
out.mkdir(parents=True, exist_ok=True)
torch.save(payload, out / "model.pt")
h = hashlib.sha256()
with (out / "model.pt").open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
info = json.loads((dirs[0] / "info.json").read_text())
info.update({"soup": {"sources": [str(d) for d in dirs], "weights": weights}, "sha256": h.hexdigest()})
(out / "info.json").write_text(json.dumps(info, indent=2))
print(f"wrote {out}: {len(dirs)} checkpoints, weights {weights}, sha256 {h.hexdigest()[:16]}")
