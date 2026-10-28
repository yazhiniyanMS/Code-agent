#!/bin/bash
# V11: one run of a spec's arm, made to survive a container restart (which wipes the RAM disk):
# * if the arm starts from C2: restore C2 on the RAM disk if needed (content verified by its pack), and drop
#   that copy once this run's trainer has loaded it;
# * as soon as training has written the final model (the runner starts evaluating dev), pack it to disk and
#   verify the pack, before the ~1.5 h of evaluation. After a restart: unpack it to the run's work dir
#   and rerun this script; the runner finds its data (on disk) and the model, trains nothing, evaluates;
# * at the end, check the kept model against that pack (bit for bit) and record it.
# usage: run_v11.sh SPEC ARM SEED
set -u
spec=$1 arm=$2 seed=$3
cd /home/user/Code-agent
L=/home/user/lmwork/v11/driver.log
K=/dev/shm/ycode-keep
C=/home/user/lmwork/v11/ckpt
R=/home/user/lmwork/v11/runs_${arm}_$seed.log
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%S)] $*" >> $L; }
cnt() { local n; n=$(grep -c "$1" "$2" 2>/dev/null); echo "${n:-0}"; }  # matching lines; 0 if no file
work=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['work_dir'])" $spec)/$arm/$seed/model
from_c2=$(python3 -c "import json,sys; s=json.load(open(sys.argv[1])); print(int('C2' in (s.get('init_from') or '')))" $spec)
if [ "$from_c2" = 1 ] && [ ! -f $K/C2/1301/model_bf16/model.pt ] && [ ! -f $work/model.pt ]; then
  python3 -m ycode.lm.packckpt unpack /home/user/lmwork/v10/ckpt/C2_1301.ypk $K/C2/1301/model_bf16 >> $L 2>&1 \
    && log "C2 restored to the RAM disk from its pack (content verified on unpack)" || { log "$arm $seed: cannot restore C2"; exit 1; }
fi
tl=/home/user/lmwork/v11/runs/$arm/$seed/train.log
before=$(cnt "^Loaded " $tl)
evals_before=$(cnt "evaluating dev" $R)
log "$arm $seed start ($spec)"
python3 -m ycode.lm.experiment $spec --arms $arm >> $R 2>&1 &
pid=$!
if [ "$from_c2" = 1 ]; then
  until [ "$(cnt "^Loaded " $tl)" -gt "$before" ] || ! kill -0 $pid 2>/dev/null; do sleep 15; done
  sleep 30
  if kill -0 $pid 2>/dev/null && [ -f /home/user/lmwork/v10/ckpt/C2_1301.ypk.ok ] && [ -d $K/C2/1301/model_bf16 ]; then
    rm -rf $K/C2/1301/model_bf16 && log "$arm $seed: trainer loaded C2; C2's RAM copy dropped (its pack is verified)"
  fi
fi
early=$C/${arm}_$seed.ypk
until [ "$(cnt "evaluating dev" $R)" -gt "$evals_before" ] || ! kill -0 $pid 2>/dev/null; do sleep 20; done
if kill -0 $pid 2>/dev/null && [ -f $work/model.pt ] && [ ! -f $early.ok ]; then
  nice -n 5 python3 -m ycode.lm.packckpt pack $work $early >> $L 2>&1 \
    && nice -n 5 python3 -m ycode.lm.packckpt verify $work $early > $early.ok 2>> $L \
    && cp $work/info.json $C/${arm}_$seed.model_info.json \
    && log "$arm $seed: trained model packed before evaluation: $early ($(stat -c %s $early) bytes), content sha256 $(cat $early.ok)" \
    || { rm -f $early.ok; log "$arm $seed: early pack failed"; }
fi
wait $pid
log "$arm $seed finished, exit $?"
src=$K/$arm/$seed/model_bf16
if [ -f $src/model.pt ]; then
  if [ -f $early.ok ] && got=$(python3 -m ycode.lm.packckpt verify $src $early 2>> $L) && [ "$got" = "$(cat $early.ok)" ]; then
    log "$arm $seed: kept model $src matches the pack $early (content sha256 $got)"
  else
    python3 -m ycode.lm.packckpt pack $src $early >> $L 2>&1 && python3 -m ycode.lm.packckpt verify $src $early > $early.ok 2>> $L \
      && cp $src/info.json $C/${arm}_$seed.model_info.json \
      && log "$arm $seed: packed to $early ($(stat -c %s $early) bytes), content sha256 $(cat $early.ok)" || { rm -f $early.ok; log "$arm $seed: pack failed"; }
  fi
else
  log "$arm $seed: no kept model at $src"
fi
log "$arm $seed driver done"
