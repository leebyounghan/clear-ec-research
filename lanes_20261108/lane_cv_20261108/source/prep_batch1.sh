#!/bin/bash
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
PY=/workspace/venv/bin/python
F=/workspace/clear-ec-research/experiments/green_track_20261028/external_srep
date; $PY source/graph_prep_sp_20261108.py repro_t035.npz "" 3 380 300 0.35 0
date; $PY source/graph_prep_sp_20261108.py sp07_t02.npz "" 3 380 300 0.2 0.7
date; $PY source/graph_prep_sp_20261108.py sp07_t02_p47.npz $F/s47_preonly 3 380 300 0.2 0.7
date; $PY source/graph_prep_sp_20261108.py sp07_t02_p48.npz $F/s48_preonly 3 380 300 0.2 0.7
date; $PY source/graph_prep_sp_20261108.py sp06_t05f07.npz "" 3 380 300 0.5 0.6 0.2 0.7
date; $PY source/graph_prep_sp_20261108.py sp06_t05f07_p47.npz $F/s47_preonly 3 380 300 0.5 0.6 0.2 0.7
date; $PY source/graph_prep_sp_20261108.py sp06_t05f07_p48.npz $F/s48_preonly 3 380 300 0.5 0.6 0.2 0.7
date; echo ALLDONE
