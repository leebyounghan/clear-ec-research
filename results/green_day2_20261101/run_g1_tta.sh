#!/bin/bash
# G1: four-flip TTA peaks of the three pre-only detectors, fold-clean, on GPUs 0 and 1 (lane01 job).
cd /workspace/clear-ec-research
PY=/workspace/venv/bin/python; S=experiments/green_day2_20261101/source/green_day2_fullframe_20261101.py
LOG=experiments/green_day2_20261101/preonly_tta; mkdir -p $LOG
CUDA_VISIBLE_DEVICES=0 $PY -u $S infer-pre-tta --family s46 --fold 0 --limit 8 --tag smoke > $LOG/smoke.log 2>&1 || { echo "smoke failed"; tail -5 $LOG/smoke.log; exit 1; }
run_list() { gpu=$1; shift; for item in "$@"; do fam=${item%%:*}; f=${item##*:}; if [ -f $LOG/$fam/fold$f/fold${f}_sg2_peaks.npz ]; then continue; fi; CUDA_VISIBLE_DEVICES=$gpu $PY -u $S infer-pre-tta --family $fam --fold $f >> $LOG/extract_gpu$gpu.log 2>&1 || echo "FAILED $item" >> $LOG/extract_gpu$gpu.log; done; }
run_list 0 s46:0 s46:2 s46:4 s47:1 s47:3 s48:0 s48:2 s48:4 &
run_list 1 s46:1 s46:3 s47:0 s47:2 s47:4 s48:1 s48:3 &
wait
ls $LOG/*/fold*/fold*_sg2_peaks.npz | wc -l
echo "g1 done $(date -u)"
