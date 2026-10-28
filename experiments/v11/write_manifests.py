"""V11: the checkpoint manifest (experiments/v11/artifacts/CHECKPOINTS.json) and the two named artifacts.

usage: write_manifests.py --best-experimental NAME [--validated-best NAME]

Every checkpoint V11 relies on or produced: where it is, how it is stored, its file and tensor-content
SHA-256, its parent, configuration and seeds, its evaluation summary (from the run record), its gate
status, and the commands that restore and evaluate it. File hashes are recomputed here (not copied), so
the manifest states what is on disk now.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LM = Path("/home/user/lmwork")
RES = ROOT / "experiments"
UNPACK = "python -m ycode.lm.packckpt unpack {pack} OUT_DIR   # every tensor checked against the content sha256"
EVAL = "python -m ycode.lm.cli eval --model OUT_DIR --problems dev --device cpu --answers-out dev.jsonl"
NEEDS = ["/home/user/lmwork/loop/anchor (round 54: model.pt, info.json; the packs store XOR against it)",
         "/home/user/lmwork/data/tokenizer.json"]

CHECKPOINTS = [
    {"name": "Round 54", "role": "checkpoint of record (V5.1); the base every pack references",
     "files": [LM / "loop/anchor/model.pt", LM / "loop/anchor/optim.pt", LM / "loop/anchor/info.json",
               LM / "loop/anchor/sampler.json"], "storage": "fp32 training checkpoint (trained layers fp32, frozen bf16)",
     "parent": "YCode-LM v5 (models/ycode-lm-v5, step 60) + 72 fine-tuning steps (the V5 round loop)",
     "record": RES / "v51/results/m4/anchor", "restore": "python -c \"from pathlib import Path; from ycode.lm.train import "
     "load_checkpoint; load_checkpoint(Path('/home/user/lmwork/loop/anchor'))\""},
    {"name": "C2", "pack": LM / "v10/ckpt/C2_1301.ypk", "parent": "Round 54",
     "config": "experiments/v10/v10_c2_rebuild.json (= V9-1's C2)", "seeds": {"train": 1301, "data": 1301},
     "record": RES / "v9/results/runs/C2/1301"},
    {"name": "R1", "pack": LM / "v10/ckpt/R1_1301.ypk", "parent": "C2", "config": "experiments/v10/v10_r1.json",
     "seeds": {"train": 1301, "data": 1302}, "record": RES / "v10/results/runs/R1/1301"},
    {"name": "R6", "pack": LM / "v10/ckpt/R6_1301.ypk", "parent": "C2", "config": "experiments/v10/v10_r6.json",
     "seeds": {"train": 1301, "data": 1302}, "record": RES / "v10/results/runs/R6/1301"},
    {"name": "R7", "pack": LM / "v10/ckpt/R7_1301.ypk", "parent": "C2", "config": "experiments/v10/v10_r7.json",
     "seeds": {"train": 1301, "data": 1302}, "record": RES / "v10/results/runs/R7/1301"},
    {"name": "V11-A", "pack": LM / "v11/ckpt/A_1302.ypk", "parent": "C2", "config": "experiments/v11/v11_a.json",
     "seeds": {"train": 1302, "data": 1303}, "record": RES / "v11/results/runs/A/1302"},
    {"name": "V11-B", "pack": LM / "v11/ckpt/B_1301.ypk", "parent": "C2", "config": "experiments/v11/v11_b.json",
     "seeds": {"train": 1301, "data": 1302}, "record": RES / "v11/results/runs/B/1301"},
    {"name": "V11-W", "pack": None, "parent": "0.5 x R1 + 0.5 x Round 54 (no training)",
     "config": "python experiments/v6/soup.py OUT_DIR R1_DIR:0.5 /home/user/lmwork/loop/anchor:0.5 (R1_DIR: R1 unpacked)",
     "seeds": {"train": 1301, "data": 1302}, "record": RES / "v11/results/runs/W/1301",
     "storage": "not kept (deterministic from R1 and round 54; rebuild and compare the file sha256 below)"},
    {"name": "V11-C", "pack": LM / "v11/ckpt/C_1301.ypk", "parent": "Round 54", "config": "experiments/v11/v11_c.json",
     "seeds": {"train": 1301, "data": 1302}, "record": RES / "v11/results/runs/C/1301"},
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summary(record_dir: Path) -> dict:
    sys.path.insert(0, str(ROOT))
    from ycode.lm.experiment import scores

    rec_path = record_dir / "result.json"
    rec = json.loads(rec_path.read_text()) if rec_path.is_file() else {
        w: scores(record_dir / f"{w}.jsonl", w, classify=False) for w in ("dev", "unseen", "bench")}
    out = {w: {k: rec[w].get(k) for k in ("points", "of", "clean", "clean_of", "robust", "robust_of", "robust_clean")}
           for w in ("dev", "unseen", "bench") if w in rec}
    if "dev" in out and "unseen" in out:
        out["combined_clean"] = out["dev"]["clean"] + out["unseen"]["clean"]
    t = rec.get("training") or {}
    out["training"] = {k: t.get(k) for k in ("steps", "tokens", "minutes", "peak_rss_mb", "checkpoint_sha256") if k in t}
    if rec.get("minutes_total") is not None:
        out["minutes_total"] = rec["minutes_total"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-experimental", required=True)
    ap.add_argument("--validated-best", default="Round 54")
    ap.add_argument("--gate", type=Path, default=ROOT / "experiments/v11/results/gates.json",
                    help="gate outcome per checkpoint name (written from the comparison tables)")
    args = ap.parse_args()
    gates = json.loads(args.gate.read_text()) if args.gate.is_file() else {}
    out = []
    for c in CHECKPOINTS:
        e = {"name": c["name"], "parent": c["parent"], "config": c.get("config"), "seeds": c.get("seeds"),
             "gate": gates.get(c["name"]), "role": c.get("role")}
        if c.get("files"):
            e["storage"] = c["storage"]
            e["files"] = [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in c["files"]]
            e["restore"] = c["restore"]
        elif c.get("pack"):
            p = c["pack"]
            if not p.is_file():
                continue
            ok = Path(str(p) + ".ok")
            e["storage"] = "lossless pack (ycode/lm/packckpt.py): bf16 bits XOR round 54's, byte planes, zlib"
            e["files"] = [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)}]
            e["content_sha256"] = ok.read_text().strip() if ok.is_file() else None
            e["restore"] = UNPACK.format(pack=p)
            e["needs"] = NEEDS
        else:
            e["storage"] = c["storage"]
        e["evaluate"] = EVAL
        if (c["record"]).exists():
            e["evaluation"] = summary(c["record"])
            e["record"] = str(c["record"].relative_to(ROOT))
        out.append(e)
    art = ROOT / "experiments/v11/artifacts"
    art.mkdir(parents=True, exist_ok=True)
    (art / "CHECKPOINTS.json").write_text(json.dumps(out, indent=1))
    names = {e["name"]: e for e in out}
    for label, name in (("validated_best", args.validated_best), ("best_experimental_candidate", args.best_experimental)):
        m = {**names[name], "label": label.replace("_", " ")}
        (art / f"{label}.json").write_text(json.dumps(m, indent=1))
        print(label, name, m.get("evaluation", {}).get("combined_clean"))
    print(f"wrote {art / 'CHECKPOINTS.json'}: {len(out)} checkpoints")


if __name__ == "__main__":
    main()
