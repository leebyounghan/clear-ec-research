#!/bin/bash
# HEX lane (2026-11-08): wait for seg_run.py (pid arg 1) to exit, then merge its chunks, build mask graphs, and append the two
# mask-graph network jobs to jobs_gpu1.tsv (new file + mv, as the runner expects). Do not edit while it runs.
LANE=/workspace/clear-ec-research/experiments/lane_hex_20261108; cd $LANE; PY=/workspace/venv/bin/python; P=$1
log() { echo "$(date -u +%FT%TZ) $*" >> $LANE/logs/post_seg.log; }
log "waiting for seg pid $P"
while [ -d /proc/$P ]; do sleep 60; done
log "seg exited; chunks $(ls seg_loose | grep -c '^c[0-9]*.npz$')"
OMP_NUM_THREADS=2 $PY source/seg_merge.py seg_loose seg_all >> logs/post_seg.log 2>&1
OMP_NUM_THREADS=1 $PY source/mask_graph.py seg_loose g_mask.npz 3 >> logs/post_seg.log 2>&1 || { log "mask_graph failed"; exit 1; }
cp jobs_gpu1.tsv jobs_gpu1.tsv.new
printf "mask_hex_s5\tsource/gnn_hex.py mask_hex_s5 --graphs $LANE/feats/g_mask.npz --lw 0,0,1 --seeds 5 --threads 2\n" >> jobs_gpu1.tsv.new
printf "mask_all3_s5\tsource/gnn_hex.py mask_all3_s5 --graphs $LANE/feats/g_mask.npz --lw 1,1,1 --seeds 5 --threads 2\n" >> jobs_gpu1.tsv.new
mv jobs_gpu1.tsv.new jobs_gpu1.tsv; log "mask jobs appended"
