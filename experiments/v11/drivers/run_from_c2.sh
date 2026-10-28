#!/bin/bash
# V11: one run from C2: restore C2 on the RAM disk if needed (verified by its pack), launch the spec's arm, drop
# C2's RAM copy once this run's trainer has loaded it, then pack the kept model to disk and verify the pack.
# usage: run_from_c2.sh SPEC ARM SEED
set -u
spec=$1 arm=$2 seed=$3
cd /home/user/Code-agent
L=/home/user/lmwork/v11/driver.log
K=/dev/shm/ycode-keep
C=/home/user/lmwork/v11/ckpt
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%S)] $*" >> $L; }
if [ ! -f $K/C2/1301/model_bf16/model.pt ]; then
  python3 -m ycode.lm.packckpt unpack /home/user/lmwork/v10/ckpt/C2_1301.ypk $K/C2/1301/model_bf16 >> $L 2>&1 \
    && log "C2 restored to the RAM disk from its pack (content verified on unpack)" || { log "$arm $seed: cannot restore C2"; exit 1; }
fi
tl=/home/user/lmwork/v11/runs/$arm/$seed/train.log
before=$(grep -c "^Loaded " $tl 2>/dev/null || echo 0)
log "$arm $seed start ($spec)"
python3 -m ycode.lm.experiment $spec --arms $arm >> /home/user/lmwork/v11/runs_${arm}_$seed.log 2>&1 &
pid=$!
until [ "$(grep -c "^Loaded " $tl 2>/dev/null || echo 0)" -gt "$before" ] || ! kill -0 $pid 2>/dev/null; do sleep 15; done
sleep 30
if kill -0 $pid 2>/dev/null && [ -f /home/user/lmwork/v10/ckpt/C2_1301.ypk.ok ]; then
  rm -rf $K/C2/1301/model_bf16 && log "$arm $seed: trainer loaded C2; C2's RAM copy dropped (its pack is verified)"
fi
wait $pid
log "$arm $seed finished, exit $?"
src=$K/$arm/$seed/model_bf16 dst=$C/${arm}_$seed.ypk
if [ -f $src/model.pt ]; then
  python3 -m ycode.lm.packckpt pack $src $dst >> $L 2>&1 && python3 -m ycode.lm.packckpt verify $src $dst > $dst.ok 2>> $L \
    && cp $src/info.json $C/${arm}_$seed.model_info.json \
    && log "$arm $seed: packed to $dst ($(stat -c %s $dst) bytes), content sha256 $(cat $dst.ok)" || { rm -f $dst.ok; log "$arm $seed: pack failed"; }
else
  log "$arm $seed: no kept model at $src"
fi
log "$arm $seed driver done"
