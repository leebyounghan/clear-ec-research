#!/bin/bash
# CPU pod (09:25 EST 11-14, coordinator 09:16): write the p3 / p3b CD deploy constants right after the heavy read of each stage.
# p3: after reads/heavy_final_cv10_v46_20261114.json exists, and after the ViT-L s46 2e-5 full-data final.pt exists (at most 45 min extra wait,
# then written anyway with exists_at_write_time=false). p3b: the same with the _vboth read and the s47 full-data final.pt. Dry run 09:18 passed
# (t1d rerun = stored p2 constants exactly). Stops after 20 h.
cd /workspace/clear-ec-research || exit 1
L=experiments/lane_cd_20261108; PY=/workspace/venv/bin/python; say() { echo "$(date -u +%FT%TZ) $*" >> $L/logs/auto_p3.log; }
nreads() { ps -eo args= | grep -cE "^[^ ]*python[^ ]* [^ ]*(cd_|build_cd_)[a-z0-9_]+\.py"; }
build() {  # $1 variant, $2 heavy tag, $3 full-data final.pt
  until [ -f $L/reads/heavy_final_cv10_$2_20261114.json ]; do [ $(( $(date +%s) - t0 )) -gt 72000 ] && return 1; sleep 60; done
  w=0; until [ -f $3 ] || [ $w -ge 45 ]; do sleep 60; w=$((w + 1)); done
  until [ $(nreads) -lt 2 ]; do sleep 60; done
  VARIANT=$1 OMP_NUM_THREADS=2 nice -n 19 timeout 3600 $PY $L/source/build_cd_p3_20261114.py > $L/logs/build_cd_$1.log 2>&1; say "$1 rc $? (full-data final.pt present: $([ -f $3 ] && echo yes || echo no))"
}
say "start pid $$"; t0=$(date +%s)
build p3 v46 $L/arms/vitlfu_elr2e5_f99_s46/raw/runs/fold-1/dinov3_l16t/final.pt
build p3b vboth $L/arms/vitl47fu_elr2e5_f99_s47/raw/runs/fold-1/dinov3_l16t/final.pt
say "exit"
