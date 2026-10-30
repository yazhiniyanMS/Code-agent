#!/bin/bash
# Rebuild S1 1302 (V12 recipe, unchanged) and evaluate it on the V13 string sets; V11-A's RAM copy is dropped once
# the trainer has loaded it (its pack stays on disk), as the V12 driver did.
set -u
cd /home/user/Code-agent-v12
L=/home/user/lmwork/v13/rebuild_s1_1302.log
A=/dev/shm/ycode-keep/A/1302/model_bf16
python3 -m ycode.lm.packckpt unpack /home/user/lmwork/v11/ckpt/A_1302.ypk $A >> $L 2>&1 || { echo "unpack failed" >> $L; exit 1; }
tl=/home/user/lmwork/v13/runs/S1/1302/train.log
cnt() { local n; n=$(grep -c "^Loaded " "$tl" 2>/dev/null); echo "${n:-0}"; }  # train.log accumulates over attempts
before=$(cnt)
python3 -m ycode.lm.experiment /home/user/lmwork/v13/v13_baseline_s1_1302.json --arms S1 >> $L 2>&1 &
pid=$!
until [ "$(cnt)" -gt "$before" ] || ! kill -0 $pid 2>/dev/null; do sleep 15; done
sleep 30
kill -0 $pid 2>/dev/null && rm -rf $A && echo "[$(date -u +%H:%M:%S)] trainer loaded A; its RAM copy dropped" >> $L
wait $pid
echo "[$(date -u +%H:%M:%S)] exit $?" >> $L
