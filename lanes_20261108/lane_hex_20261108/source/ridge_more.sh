#!/bin/bash
# build ridge evidence for the remaining lane graph files, one at a time, after the frz46 build ends (HEX lane, 2026-11-08)
L=/workspace/clear-ec-research/experiments/lane_hex_20261108; cd $L
while [ ! -f feats/ridge_frz46.npz ]; do sleep 30; done
for t in box2 k1 sg2s46 sg2s49 sig5 fprefrz; do
  NW=2 OMP_NUM_THREADS=1 python source/feat_ridge.py feats/g_$t.npz $t > logs/feat_ridge_$t.log 2>&1
  echo "$(date -u +%FT%TZ) ridge $t exit $?" >> logs/ridge_more.log
done
echo "$(date -u +%FT%TZ) all done" >> logs/ridge_more.log
