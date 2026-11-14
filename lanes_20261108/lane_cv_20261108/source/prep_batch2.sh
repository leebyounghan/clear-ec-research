#!/bin/bash
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
PY=/workspace/venv/bin/python; export PREP_WORKERS=2
F=/workspace/clear-ec-research/experiments/green_track_20261028/external_srep
for fam in "" $F/s47_preonly $F/s48_preonly; do
  sfx=$( [ -z "$fam" ] && echo "" || echo "_$(basename $fam | cut -c1-3 | sed s/s/p/)" )
  date; $PY source/graph_prep_sp_20261108.py k5_t035$sfx.npz "$fam" 5 380 300 0.35 0
  date; $PY source/graph_prep_sp_20261108.py box2_t035$sfx.npz "$fam" 2 538 408 0.35 0
done
date; echo ALLDONE
