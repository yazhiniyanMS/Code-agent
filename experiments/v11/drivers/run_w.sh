#!/bin/bash
# V11-W (PLAN.md amendment 2): interpolate one trained seed of the R1 recipe halfway with round 54 and evaluate
# it with the runner's evaluation (experiments/v11/eval_only.py). Restart-safe: every step is skipped if done.
# usage: run_w.sh SRC_PACK SRC_NAME SEED      e.g. run_w.sh /home/user/lmwork/v10/ckpt/R1_1301.ypk R1 1301
set -u
pack=$1 name=$2 seed=$3
cd /home/user/Code-agent
L=/home/user/lmwork/v11/driver.log
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%S)] $*" >> $L; }
S=/dev/shm/ycode-w/src/$name/$seed/model_bf16
W=/dev/shm/ycode-w/W/$seed/model_bf16
R=/home/user/lmwork/v11/runs/W/$seed
mkdir -p $R
log "W $seed start (from $name $seed: $pack)"
if [ ! -f $W/model.pt ]; then
  if [ ! -f $S/model.pt ]; then
    python3 -m ycode.lm.packckpt unpack $pack $S >> $L 2>&1 && log "W $seed: $name $seed restored from its pack (content verified)" \
      || { log "W $seed: cannot restore $name $seed"; exit 1; }
  fi
  python3 - "$S" "$W" "$R/build.json" >> $L 2>&1 <<'EOF' || { log "W $seed: interpolation failed"; exit 1; }
import json, sys
from pathlib import Path
from ycode.lm.experiment import run_measured
src, out, rec = sys.argv[1], sys.argv[2], Path(sys.argv[3])
anchor = "/home/user/lmwork/loop/anchor"
cmd = [sys.executable, "experiments/v6/soup.py", out, f"{src}:0.5", f"{anchor}:0.5"]
r = run_measured(cmd, rec.with_suffix(".log"))
if r["exit"] != 0:
    raise SystemExit(f"soup.py failed: {r}")
info = json.loads((Path(out) / "info.json").read_text())
rec.write_text(json.dumps({"method": "experiments/v6/soup.py (fp32 average, stored bf16, base-referenced)",
                           "sources": info["soup"]["sources"], "weights": info["soup"]["weights"],
                           "sha256": info["sha256"], **r}, indent=1))
print("interpolated", info["sha256"], r)
EOF
  log "W $seed: interpolated ($(python3 -c "import json;print(json.load(open('$R/build.json'))['sha256'])"))"
fi
rm -rf /dev/shm/ycode-w/src/$name/$seed
if [ ! -f $R/result.json ]; then
  python3 experiments/v11/eval_only.py experiments/v11/v11_b.json $W $R --arm W --seed $seed \
    --init-from /home/user/lmwork/v11/$name/$seed/model_bf16 --build-json $R/build.json \
    --note "0.5 x $name $seed + 0.5 x round 54 (PLAN.md amendment 2)" >> /home/user/lmwork/v11/runs_W_$seed.log 2>&1 \
    && log "W $seed: evaluated" || { log "W $seed: evaluation failed"; exit 1; }
fi
log "W $seed driver done"
