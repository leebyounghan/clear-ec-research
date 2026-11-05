#!/bin/bash
# Train-fit + SWA inference on completed recipe A s46 fold roots; one process per physical GPU, folds listed per GPU.
# usage: run_trainfit.sh "GPU:FOLD[,FOLD..]" ["GPU:FOLD.."]   e.g. run_trainfit.sh 0:0,1 1:2
cd /workspace/clear-ec-research
PY=/workspace/venv/bin/python; SRC=experiments/meta_ceiling_20261103/source; OUT=experiments/meta_ceiling_20261103/trainfit
declare -A RT=([0]=experiments/phase2_pooling_replication_20261026/f0_s46_std [1]=experiments/oof_recipeA_20261027/fold1/regression_std_s46 \
  [2]=experiments/oof_recipeA_20261027/fold2/regression_std_s46_v2 [3]=experiments/oof_recipeA_20261027/fold3/regression_std_s46 \
  [4]=experiments/phase2_pooling_20261020/arms/std)
mkdir -p $OUT
for spec in "$@"; do
  G=${spec%%:*}; FL=${spec#*:}
  ( for F in ${FL//,/ }; do
      EXTRA=""; [ "$F" = 0 ] || [ "$F" = 4 ] && EXTRA="--epochs 5 10 15"
      CUDA_VISIBLE_DEVICES=$G $PY -u $SRC/trainfit_swa_inference_20261103.py ${RT[$F]} --out $OUT $EXTRA >> $OUT/fold${F}.log 2>&1 || echo "FAILED fold $F" >> $OUT/fold${F}.log
    done ) &
done
wait
grep -l FAILED $OUT/fold*.log && exit 1 || exit 0
