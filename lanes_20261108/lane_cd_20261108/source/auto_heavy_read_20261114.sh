#!/bin/bash
# CPU pod (07:55 EST 11-14): heavy-member final reads (cd_heavy_final_20261114.py, self-test passed: every swap of the incumbent members
# reproduces tier p 6.1737 / p2 6.1723). Stage 1 (OUTTAG _v46): when ViT-L s46 2e-5 folds 1-3 have final TTA (folds 0 / 4 exist) and the
# DINOv3 2e-5 fold TTAs exist, strict then cv10. Stage 2 (OUTTAG _vboth): when ViT-L s47 2e-5 folds 0-4 have final TTA, strict then cv10.
# Each read starts only while fewer than 2 lane read processes run on this pod. Stops after 20 h.
cd /workspace/clear-ec-research/experiments/lane_cd_20261108/source || exit 1
PY=/workspace/venv/bin/python; A=../arms; T=../tta; say() { echo "$(date -u +%FT%TZ) $*" >> ../logs/auto_heavy_read.log; }
nreads() { ps -eo args= | grep -cE "^[^ ]*python[^ ]* cd_[a-z0-9_]+\.py"; }
v46() { for f in 1 2 3; do [ -f $A/vitlfu_elr2e5_f${f}_s46/raw/runs/fold$f/dinov3_l16t/predictions_final_tta.npz ] || return 1; done; for f in 1 2 3 4; do [ -f $T/fu2_elr2e5_f${f}_tta4_e16_20.npz ] || return 1; done; }
v47() { for f in 0 1 2 3 4; do [ -f $A/vitl47fu_elr2e5_f${f}_s47/raw/runs/fold$f/dinov3_l16t/predictions_final_tta.npz ] || return 1; done; }
run() {  # $1 tag, $2 mode
  until [ $(nreads) -lt 2 ]; do sleep 60; done
  (OUTTAG=$1 NR=20 OMP_NUM_THREADS=2 nice -n 19 timeout 21600 $PY cd_heavy_final_20261114.py $2 > ../logs/heavy_final_$2$1.log 2>&1; echo "$(date -u +%FT%TZ) $2$1 rc $?" >> ../logs/auto_heavy_read.log) &
  say "$2$1 started"; sleep 30
}
say "start pid $$"; t0=$(date +%s); s1=0; s2=0
while [ $(( $(date +%s) - t0 )) -lt 72000 ] && { [ $s1 = 0 ] || [ $s2 = 0 ]; }; do
  if [ $s1 = 0 ] && v46; then run _v46 strict; run _v46 cv10; s1=1; fi
  if [ $s2 = 0 ] && v47; then run _vboth strict; run _vboth cv10; s2=1; fi
  sleep 120
done
wait; say "exit"
