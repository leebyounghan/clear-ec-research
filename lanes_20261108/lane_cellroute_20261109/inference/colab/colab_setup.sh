#!/bin/bash
# deploy lane 2026-11-12 (D-PKG-006): lay out the staged t1 package on the PI's Colab T4 exactly as Dockerfile.freeze does (TIER=t1):
# code tree at /workspace/clear-ec-research, model archive at /opt/ml/model, one link per model_files.txt entry, tier manifest -> deploy.json.
# The staged pack_hooks.py is kept as pack_hooks_staged.py (the container default); the jobs driver swaps hooks per mode.
set -e
C=/content/t1/context; M=/content/t1/model; W=/workspace/clear-ec-research
mkdir -p /workspace /opt/ml /content/t4test/out /content/t4test/jobs
[ -e $W ] || ln -s $C/code $W
[ -e /opt/ml/model ] || ln -s $M /opt/ml/model
n=0; miss=0
while read -r rel; do mkdir -p "$(dirname "$W/$rel")"; ln -sfn "/opt/ml/model/$rel" "$W/$rel"; n=$((n+1)); [ -e "/opt/ml/model/$rel" ] || miss=$((miss+1)); done < $C/model_files.txt
I=$W/experiments/lane_cellroute_20261109/inference
cp $I/manifests/deploy_t1.json $I/manifests/deploy.json; cp $I/params/deploy_t1.json $I/params/deploy.json
[ -f $I/pack_hooks_staged.py ] || cp $I/pack_hooks.py $I/pack_hooks_staged.py
echo "links $n missing_targets $miss tier $(cat $C/MANIFEST.json | head -c 300)"
sha256sum $I/pack_hooks_staged.py | cut -c1-16
[ "$miss" = 0 ] || { echo "MODEL_FILES_MISSING $miss"; exit 1; }
