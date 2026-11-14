#!/bin/bash
# Usage: add_arm.sh <name> <fold> <ssl flags...>: SSL, whole-frame features of its last checkpoint, frozen read against SSL5 of that fold. Appends one queue line.
name=$1; fold=$2; shift 2
L=experiments/lane_gpu1_20261106; PY=/workspace/venv/bin/python
printf '%s\t%s\n' "$name" "$PY -u $L/source/ssl_views_20261106.py $name --fold $fold $* && $PY -u $L/source/frozen_features_20261106.py $name $L/ssl/$name/run/last.pt && CUDA_VISIBLE_DEVICES= $PY -u $L/source/frozen_read_20261106.py $name $fold --ref ssl5_f$fold" >> /workspace/clear-ec-research/$L/queue.tsv
