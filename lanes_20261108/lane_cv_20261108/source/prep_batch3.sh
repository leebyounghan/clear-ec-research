#!/bin/bash
# graphs from the in-domain fine-tuned sg2 specialists (seeds 46-50), standard nodes t0.35, 3 windows 380x300
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
PY=/workspace/venv/bin/python; export PREP_WORKERS=2
until grep -q ALLDONE logs/prep_batch2.log; do sleep 20; done
G=/workspace/clear-ec-research/experiments/green_track_20261028
for s in "" _s47 _s48 _s49 _s50; do
  date; $PY source/graph_prep_sp_20261108.py ft${s:-_s46}_t035.npz $G/sg2_specialist$s 3 380 300 0.35 0
done
date; echo ALLDONE
