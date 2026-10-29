#!/bin/bash
# V12 (from V11's run_v11.sh): one run of a spec's arm. In V12 the pack stays on the RAM disk (no room on disk), so a
# container restart means a re-run of a deterministic recipe. Steps:
# * if the arm starts from C2: restore C2 on the RAM disk if needed (content verified by its pack), and drop
#   that copy once this run's trainer has loaded it;
# * as soon as training has written the final model (the runner starts evaluating dev), pack it to disk and
#   verify the pack, before the ~1.5 h of evaluation. After a restart: unpack it to the run's work dir
#   and rerun this script; the runner finds its data (on disk) and the model, trains nothing, evaluates;
# * at the end, check the kept model against that pack (bit for bit) and record it.
# usage: run_v12.sh SPEC ARM SEED
set -u
mkdir -p /dev/shm/ycode-v12-packs
spec=$1 arm=$2 seed=$3
cd /home/user/Code-agent-v12  # a worktree fixed at the run branch: other work cannot change the code mid-run
L=/home/user/lmwork/v12/driver.log
K=/dev/shm/ycode-keep
C=/dev/shm/ycode-v12-packs  # RAM disk: the disk has 0.67 GB and a pack needs 1.4 (V12 PLAN.md, Storage)
R=/home/user/lmwork/v12/runs_${arm}_$seed.log
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%S)] $*" >> $L; }
cnt() { local n; n=$(grep -c "$1" "$2" 2>/dev/null); echo "${n:-0}"; }  # matching lines; 0 if no file
work=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['work_dir'])" $spec)/$arm/$seed/model
init=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('init_from') or '')" $spec)
from_ram=0
# a starting checkpoint on the RAM disk (/dev/shm/ycode-keep/NAME/1301/model_bf16) is restored from its verified pack
declare -A PACKS=([C2]=/home/user/lmwork/v10/ckpt/C2_1301.ypk [R1]=/home/user/lmwork/v10/ckpt/R1_1301.ypk [A]=/home/user/lmwork/v11/ckpt/A_1302.ypk)
init_name=$(echo "$init" | sed -n 's#^/dev/shm/ycode-keep/\([^/]*\)/.*#\1#p')
if [ -n "$init_name" ]; then
  from_ram=1  # starts from a RAM-disk copy, dropped once the trainer has loaded it
  if [ ! -f $init/model.pt ] && [ ! -f $work/model.pt ]; then
    python3 -m ycode.lm.packckpt unpack ${PACKS[$init_name]} $init >> $L 2>&1 \
      && log "$init_name restored to the RAM disk from its pack (content verified on unpack)" || { log "$arm $seed: cannot restore $init_name"; exit 1; }
  fi
fi
tl=/home/user/lmwork/v12/runs/$arm/$seed/train.log
before=$(cnt "^Loaded " $tl)
evals_before=$(cnt "evaluating dev" $R)
log "$arm $seed start ($spec)"
python3 -m ycode.lm.experiment $spec --arms $arm >> $R 2>&1 &
pid=$!
if [ "$from_ram" = 1 ]; then
  until [ "$(cnt "^Loaded " $tl)" -gt "$before" ] || ! kill -0 $pid 2>/dev/null; do sleep 15; done
  sleep 30
  if kill -0 $pid 2>/dev/null && [ -f ${PACKS[$init_name]}.ok ] && [ -d $init ]; then
    rm -rf $init && log "$arm $seed: trainer loaded $init_name; its RAM copy dropped (its pack is verified)"
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
