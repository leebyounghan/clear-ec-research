#!/bin/bash
# Stage the final9000 checkpoints under the container's expected filenames, verify, build v4, export.
# Run only after all four final9000_{s2,at,cl,xHEX}.pt exist.
set -e
cd /home/qudgks/workspace/CLEAR-EC
ROOT=datasets/clear_ec_official
DST=submission_model_final
mkdir -p $DST
cp -r cpsam_models/cpsam_v2 $DST/ 2>/dev/null || cp -r submission_model/cpsam_v2 $DST/
cp $ROOT/final9000_s2.pt   $DST/regress_ft_best_clahe_s2.pt
cp $ROOT/final9000_at.pt   $DST/regress_ft_best_attn_clahe.pt
cp $ROOT/final9000_cl.pt   $DST/regress_ft_best_clahe_long.pt
cp $ROOT/final9000_xHEX.pt $DST/regress_ft_best_spec_hex.pt
chmod -R o+rX $DST
echo "staged final checkpoints -> $DST"
ls -lh $DST | awk '{print $5,$9}'
# NOTE: final9000 models are trained on ALL data incl. the internal val/test, so verify_container_infer
# on those rows would be optimistic (in-sample). We only smoke-test for correctness, not accuracy.
SMK=/tmp/claude-1001/-home-qudgks-workspace-CLEAR-EC/dc08fee9-8389-4438-be27-f99c7aaeeb70/scratchpad/smoke
cp submission/inference.py clear-ec-algo/inference.py
rm -rf clear-ec-algo/model && mkdir -p clear-ec-algo/model
cp -r $DST/* clear-ec-algo/model/
docker build -t clear_ec_algo:v4 clear-ec-algo/
chmod 777 $SMK/output 2>/dev/null || true
docker run --rm --network none -e CUDA_VISIBLE_DEVICES="" -v $SMK/input:/input:ro -v $SMK/output:/output clear_ec_algo:v4
echo "--- smoke outputs ---"; for f in $SMK/output/*.json; do echo "$(basename $f): $(cat $f)"; done
docker save clear_ec_algo:v4 | pigz -c > clear-ec-algo-image-v4.tar.gz
ls -lh clear-ec-algo-image-v4.tar.gz
echo "V4 READY -> clear-ec-algo-image-v4.tar.gz"
