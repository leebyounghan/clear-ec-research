#!/bin/bash
# usage: run_scale_views.sh GPU fold:root ...   Sequential per lane, skips finished folds.
cd /workspace/clear-ec-research; X=experiments/meta_ceiling_20261103; G=$1; shift
for item in "$@"; do f=${item%%:*}; r=${item#*:}; [ -f $X/zoom/views_f$f.npz ] || CUDA_VISIBLE_DEVICES=$G /workspace/venv/bin/python $X/source/scale_views_20261104.py $r $X/zoom/views_f$f.npz > $X/zoom/views_f$f.log 2>&1; done
