#!/bin/bash
# deploy lane 2026-11-12 (D-PKG-004): tier t1 precision variants on the 20 deploy frames, GPU 2 (shared with the backbone lane's synthft_ra_f0_s47).
# Four streams of one job each, 1 CPU thread per job (<= 4 on the main pod). Parity only; wall ratios come from a separate interleaved timing pass.
#   ref   original fp32 weight files, bf16 autocast, TF32 default   (the reference: how the lanes' numbers were computed)
#   pkg   packaged (fp16 copies, base weights skipped), bf16          (the deploy path on a bf16 GPU)
#   fp16  every autocast fp16, CPSAM float32, TF32 off              (the current T4 path, O-DEPLOY-016)
#   mix   fp16, detector / hex6 modules fp32, CPSAM float32, TF32 off
#   fp32  every autocast region fp32, CPSAM float32, TF32 off
#   emu   bf16 emulated in fp32 arithmetic (no bf16 matmul / conv / attention kernels), CPSAM bf16 emulated, TF32 off
#   kern  native bf16, TF32 off only                                (the part every T4 path carries)
#   flash native bf16, FlashAttention-2 instead of cuDNN attention  (the attention kernel an A10G would run)
#   flops 1 frame, bf16, FLOPs per stage
cd /workspace/clear-ec-research
L=experiments/lane_cellroute_20261109; G2=experiments/lane_backbone_20261108/source/gpu_launch_20261111.sh; PY=/workspace/venv/bin/python
MODS=deploy_hexkonan_image,hex6_correction,green_point_features_20261027,green_sg2_specialist_20261028,green_s8_specialist_seed_20261028,green_selftrain_20261028
T="CLEAR_EC_TIER=t1 THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
run() {   # name, env, frames
  RAMP=60 bash $G2 2 12000 "$T $2 timeout -k 10 14400 $PY $L/inference/test_t4.py ${3:-20} t4_$1_${3:-20} > $L/logs/t4_$1_${3:-20}.log 2>&1 < /dev/null"
  echo "done $1 $(date -u)"
}
( run ref "REF32=1 CLEAR_EC_PREC=bf16"; run fp32 "CLEAR_EC_PREC=fp32 CLEAR_EC_NO_TF32=1" ) &
( run emu "CLEAR_EC_PREC=emu CLEAR_EC_NO_TF32=1"; run kern "CLEAR_EC_PREC=bf16 CLEAR_EC_NO_TF32=1" ) &
( run mix "CLEAR_EC_PREC=mix CLEAR_EC_FP32_MODULES=$MODS CLEAR_EC_NO_TF32=1"; run flash "CLEAR_EC_PREC=bf16 SDPA_BACKEND=flash" ) &
( run flops "CLEAR_EC_PREC=bf16 FLOPS=1" 1; run fp16 "CLEAR_EC_PREC=fp16 CLEAR_EC_NO_TF32=1"; run pkg "CLEAR_EC_PREC=bf16" ) &
wait
echo "done t4 chain $(date -u)"
