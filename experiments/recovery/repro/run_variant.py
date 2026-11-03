"""One variant: pretrain the 'small' preset 4 steps, then fine-tune its top 2 layers 4 steps (Lion, compact,
final bf16), as the V10-V12 runs did. Writes the checkpoints' content hashes and file hashes as JSON."""
import hashlib, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the repository root
import torch
from ycode.lm.train import TrainConfig, train
from ycode.lm.packckpt import content_sha256

root, out, precision = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
common = dict(device="cpu", precision=precision, seed=1301, eval_iters=0, eval_interval=1000, log_interval=1000)
t = time.time()
train(root / "data", out / "base", TrainConfig(stage="pretrain", preset="small", max_steps=4, batch_size=4,
      warmup_steps=1, lr=1e-3, optimizer="lion", **common), log=lambda *_: None)
train(root / "sft", out / "ft", TrainConfig(stage="sft", init_from=str(out / "base"), max_steps=4, batch_size=4,
      warmup_steps=1, lr=3e-4, optimizer="lion", train_layers=2, compact_checkpoints=True, final_bf16=True,
      **common), log=lambda *_: None)
res = {"variant": out.name, "precision": precision, "threads": torch.get_num_threads(), "seconds": round(time.time() - t, 1),
       "env": {k: os.environ[k] for k in ("ONEDNN_MAX_CPU_ISA", "MKL_CBWR", "OMP_NUM_THREADS") if k in os.environ}}
for stage in ("base", "ft"):
    payload = torch.load(out / stage / "model.pt", map_location="cpu", weights_only=True)
    res[stage] = {"content": content_sha256(payload["model"]), "file": hashlib.sha256((out / stage / "model.pt").read_bytes()).hexdigest(),
                  "tensors": len(payload["model"])}
    torch.save(payload["model"], out / f"{stage}_state.pt")
print(json.dumps(res))
