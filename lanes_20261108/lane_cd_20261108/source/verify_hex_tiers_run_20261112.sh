#!/bin/bash
# Verifier driver (CPU pod, <= 4 workers): rerun the HEX strict ladder per fold into the CD lane's reads, the L5 parity runs per tier, then assemble.
cd /workspace/clear-ec-research/experiments/lane_cd_20261108 || exit 1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2; PY=/workspace/venv/bin/python; H=/workspace/clear-ec-research/experiments/lane_hex_20261108/source
O=reads/hex_ladder_verify_20261112; mkdir -p $O
for f in 0 1 2 3; do timeout 3000 $PY $H/strict_ladder.py $f $PWD/$O > logs/verify_hex_ladder_f${f}_20261112.log 2>&1 & done; wait
timeout 3000 $PY $H/strict_ladder.py 4 $PWD/$O > logs/verify_hex_ladder_f4_20261112.log 2>&1 &
JOINT_TAG=t2 JOINT_DROP=mtcnn5 timeout 3000 $PY $H/joint_deploy.py parity > logs/verify_hex_parity_t2_20261112.log 2>&1 &
JOINT_TAG=t3 JOINT_DROP=mtcnn5,attnp5,cvw3_mean,ridge_fall16,gnn46 timeout 3000 $PY $H/joint_deploy.py parity > logs/verify_hex_parity_t3_20261112.log 2>&1 &
JOINT_TAG=t3core JOINT_DROP=mtcnn5,attnp5,cvw3_mean timeout 3000 $PY $H/joint_deploy.py parity > logs/verify_hex_parity_t3core_20261112.log 2>&1 &
wait
cd source && timeout 900 $PY verify_hex_tiers_20261112.py > ../logs/verify_hex_tiers_20261112.log 2>&1
echo "ALLDONE $(date -u +%FT%TZ)" >> ../logs/verify_hex_tiers_20261112.log
