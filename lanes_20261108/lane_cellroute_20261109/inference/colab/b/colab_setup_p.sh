#!/bin/bash
# coordinator 2026-11-13 (D-PKG-007): lay out the staged tier-p package on the Colab T4 exactly as Dockerfile.freeze does with TIER=p
# (after the deploy lane's colab_setup.sh for t1): code tree at /workspace/clear-ec-research, model at /opt/ml/model, one link per
# model_files.txt entry, tier manifest and params copied to deploy.json.
set -e
C=/content/p/context; M=/content/p/model; W=/workspace/clear-ec-research
mkdir -p /workspace /opt/ml /content/t4test/out /content/t4test/jobs
[ -e $W ] || ln -s $C/code $W
[ -e /opt/ml/model ] || ln -s $M /opt/ml/model
n=0; miss=0
while read -r rel; do mkdir -p "$(dirname "$W/$rel")"; ln -sfn "/opt/ml/model/$rel" "$W/$rel"; n=$((n+1)); [ -e "/opt/ml/model/$rel" ] || miss=$((miss+1)); done < $C/model_files.txt
I=$W/experiments/lane_cellroute_20261109/inference
cp $I/manifests/deploy_p.json $I/manifests/deploy.json; cp $I/params/deploy_p.json $I/params/deploy.json
echo "links $n missing_targets $miss"; head -c 120 $C/MANIFEST.json; echo
sha256sum $I/pack_hooks.py $I/docker/gc_entry.py | cut -c1-16
[ "$miss" = 0 ] || { echo "MODEL_FILES_MISSING $miss"; exit 1; }
