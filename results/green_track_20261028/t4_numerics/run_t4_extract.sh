#!/bin/bash
# Two processes, one per GPU (the lane01 runner gives CUDA_VISIBLE_DEVICES=0,1); waits for both, then analyzes on CPU.
cd /workspace/clear-ec-research
PY=/workspace/venv/bin/python
SRC=experiments/green_track_20261028/source/green_t4_numerics_20261101.py
OUT=experiments/green_track_20261028/t4_numerics
run_list () {   # $1 = gpu, rest = "family:fold" items
  gpu=$1; shift
  for item in "$@"; do
    fam=${item%%:*}; f=${item##*:}
    kind=sg2; [ "$fam" = "s8_s46" ] && kind=s8
    if [ "$kind" = "s8" ]; then tgt=$OUT/fp16/$fam/fold$f/fold${f}_maps.npz; else tgt=$OUT/fp16/$fam/fold$f/fold${f}_sg2_peaks.npz; fi
    [ -f "$tgt" ] && continue
    CUDA_VISIBLE_DEVICES=$gpu $PY -u $SRC extract --family $fam --fold $f --dtype fp16 >> $OUT/extract_gpu$gpu.log 2>&1
  done
}
mkdir -p $OUT
A=""; B=""
for fam in s8_s46 sg2_s46 sg2_s47 sg2_s48; do for f in 4 0 1 2 3; do A="$A $fam:$f"; done; done
for fam in sg2_s49 sg2_s50 sg2pre_s46; do for f in 4 0 1 2 3; do B="$B $fam:$f"; done; done
# balance: move sg2_s48 folds 2,3 to GPU 1 (20 vs 15 items -> 18 vs 17)
A=$(echo $A | sed 's/ sg2_s48:2 sg2_s48:3//'); B="$B sg2_s48:2 sg2_s48:3"
run_list 0 $A &
P0=$!
run_list 1 $B &
P1=$!
wait $P0; wait $P1
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 $PY -u $SRC analyze > $OUT/analyze.log 2>&1
echo "t4 numerics done $(date -u)" >> $OUT/analyze.log
