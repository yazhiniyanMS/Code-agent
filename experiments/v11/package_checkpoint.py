"""V11: package a kept checkpoint for publication, standalone (no base model needed), with a manifest.

usage: package_checkpoint.py MODEL_DIR OUT_DIR --name NAME [--lineage TEXT] [--record result.json] [--shard-mb 1900]

* MODEL_DIR: a checkpoint dir (compact fine-tunes reference round 54; their base is folded in here).
* OUT_DIR/NAME: a bf16 inference checkpoint split into shards under --shard-mb MB each
  (ycode.lm.train.export_checkpoint: model.pt names the shards; tied weights stored once), loadable
  with ycode.lm.train.load_checkpoint / `ycode --local OUT_DIR/NAME`.
* OUT_DIR/NAME/MANIFEST.json: sha256 and size of every file, the tensor content sha256 of the full
  bf16 state, the lineage, and the evaluation summary from the run record if given.
The packaged checkpoint is then loaded back in a separate process (--verify-only) and its tensor
content hash compared with the manifest's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_sha256(model_dir: Path) -> str:
    """Tensor content hash of a checkpoint as loaded in bf16 (shards and base resolved)."""
    import torch

    from ycode.lm.packckpt import content_sha256
    from ycode.lm.train import load_checkpoint

    model, _, _ = load_checkpoint(model_dir, dtype=torch.bfloat16)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    return content_sha256(state)


def main() -> None:
    from ycode.lm.train import export_checkpoint

    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--name", required=True)
    ap.add_argument("--lineage", default="")
    ap.add_argument("--record", default=None, help="the run's result.json, for the evaluation summary")
    ap.add_argument("--shard-mb", type=float, default=1900)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    dst = Path(args.out_dir) / args.name
    if args.verify_only:
        manifest = json.loads((dst / "MANIFEST.json").read_text())
        for f in manifest["files"]:
            got = file_sha256(dst / f["name"])
            assert got == f["sha256"], f"{f['name']}: {got} != {f['sha256']}"
        got = state_sha256(dst)
        assert got == manifest["state_sha256"], f"state {got} != {manifest['state_sha256']}"
        print(f"verified {dst}: {len(manifest['files'])} files and the loaded state match the manifest")
        return
    src_state = state_sha256(Path(args.model_dir))
    export_checkpoint(Path(args.model_dir), dst, dtype="bf16", max_shard_mb=args.shard_mb)
    files = sorted(p for p in dst.iterdir() if p.is_file() and p.name != "MANIFEST.json")
    manifest = {"name": args.name, "source": str(Path(args.model_dir).resolve()), "dtype": "bf16",
                "state_sha256": src_state, "lineage": args.lineage,
                "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in files],
                "load": f"python -c \"from ycode.lm.train import load_checkpoint; load_checkpoint('{args.name}')\""}
    if args.record:
        r = json.loads(Path(args.record).read_text())
        manifest["evaluation"] = {w: {k: r[w].get(k) for k in ("points", "of", "clean", "clean_of", "robust", "robust_of")}
                                  for w in ("dev", "unseen", "bench") if w in r}
        manifest["evaluation"]["sets"] = {s: {k: v for k, v in x["scores"]["original"].items() if k != "strict_names"}
                                          for s, x in r.get("sets", {}).items()}
        manifest["training"] = {k: r["training"].get(k) for k in ("steps", "tokens", "minutes", "peak_rss_mb", "checkpoint_sha256")}
        manifest["config"] = {"data_options": r.get("data_options"), "train": r.get("train"), "seed": r.get("seed"),
                              "init_from": r.get("init_from")}
    (dst / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("name", "state_sha256")}), sum(f["bytes"] for f in manifest["files"]))


if __name__ == "__main__":
    main()
