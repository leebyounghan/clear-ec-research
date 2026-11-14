#!/bin/bash
# D-CDTRAIN-003 (coordinator 11-12): fold 4 of the two CD-only recipe A s46 arms through the backbone lane's GPU launch guard
# (experiments/lane_backbone_20261108/source/gpu_launch_20261111.sh: flock per GPU, HOLD files, free memory >= 40000 MiB).
# usage: auto_cdonly_f4_20261112.sh GPU ENC_LR NAME [WAIT_EXIT_JSON]   -- optional: first wait for that exit.json (the job now on the card).
G=$1; LR=$2; NAME=$3; WAITF=${4:-}; cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; LOG=$L/logs/auto_${NAME}.log
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
say "pid $$, GPU $G, enc LR $LR, wait for ${WAITF:-nothing}"
if [ -n "$WAITF" ]; then t0=$(date +%s); while [ ! -f "$WAITF" ]; do [ $(( $(date +%s) - t0 )) -gt 36000 ] && { say "gave up waiting"; exit 1; }; sleep 120; done; say "found $WAITF"; sleep 60; fi
CMD="TARGET_W=1,0,0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /workspace/venv/bin/python $L/source/bb_runner_cdonly_20261111.py --family cpdino_b8 --fold 4 --seed 46 --root $L/arms/$NAME --epochs 20 --micro 4 --tta-final --workers 4 --enc-lr $LR > $L/logs/$NAME.log 2>&1"
say "handing to the launch guard"
bash experiments/lane_backbone_20261108/source/gpu_launch_20261111.sh $G 40000 "$CMD" >> $LOG 2>&1
say "end rc $?"
