#!/bin/bash
# deploy lane 2026-11-12 (D-PKG-004): 1-frame smoke tests of the emu and mix precision variants on GPU 2 (shared with synthft_ra_f0_s47), 2 CPU threads each
cd /workspace/clear-ec-research
L=experiments/lane_cellroute_20261109; G2=experiments/lane_backbone_20261108/source/gpu_launch_20261111.sh; PY=/workspace/venv/bin/python
MODS=deploy_hexkonan_image,hex6_correction,green_point_features_20261027,green_sg2_specialist_20261028,green_s8_specialist_seed_20261028,green_selftrain_20261028
RAMP=20 bash $G2 2 12000 "CLEAR_EC_TIER=t1 CLEAR_EC_PREC=emu CLEAR_EC_NO_TF32=1 THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout -k 10 1800 $PY $L/inference/test_t4.py 1 t4_smoke_emu > $L/logs/t4_smoke_emu.log 2>&1 < /dev/null" &
sleep 30
RAMP=20 bash $G2 2 12000 "CLEAR_EC_TIER=t1 CLEAR_EC_PREC=mix CLEAR_EC_FP32_MODULES=$MODS CLEAR_EC_NO_TF32=1 THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout -k 10 1800 $PY $L/inference/test_t4.py 1 t4_smoke_mix > $L/logs/t4_smoke_mix.log 2>&1 < /dev/null" &
wait
echo "done smoke $(date -u)"
