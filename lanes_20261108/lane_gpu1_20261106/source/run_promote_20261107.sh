#!/bin/bash
# run_promote_20261107.sh SSL_ARM FOLD SEED: recipe A end to end from this lane's SSL recipe on folds 1, 2, 3, or 99 for the all-data fit. Steps as run_e2e_v3. Root e2e/<arm>/f<fold>_s<seed>.
set -u
cd /workspace/clear-ec-research
L=experiments/lane_gpu1_20261106
ARM=$1; F=$2; S=${3:-46}; STOP=${4:-all}
case $F in
  0) SPLIT=experiments/ssl_screen_20260911/preflight/fold0.npz;;
  1) SPLIT=experiments/ssl_screen_20260911/preflight/fold1.npz;;
  2) SPLIT=experiments/overnight_target_20260912/preflight/fold2.npz;;
  3) SPLIT=experiments/overnight_target_20260912/preflight/fold3.npz;;
  4) SPLIT=experiments/target_fold4_20260912/preflight/fold4.npz;;
  99) SPLIT=experiments/fulldata_recipeA_20261028/fulldata_regression_split.npz;;
  *) echo "fold $F not wired"; exit 2;;
esac
ROOT=$L/e2e/$ARM/f${F}_s${S}; W=recipeA_ssl_promote_20261107.py; PY=/workspace/venv/bin/python
SSL=$L/ssl/$ARM/regression_input
A="--root $ROOT --split $SPLIT --fold $F --pooling std --seed $S --ssl-root $SSL --head-lr 0.001 --clip 1 --epochs 20 --schedule-epochs 10 --deadline-utc 2026-11-14T04:00:00Z"
RT=/workspace/clear-ec-research/experiments/phase2_foundation_20261002/runtime
PP=$L/source:experiments/meta_ceiling_20261103/source:experiments/phase2_pooling_20261020/source:$RT
GPU=${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p $ROOT
step() { echo "### $(date -u +%FT%TZ) $1"; }
fail() { step "FAILED at $1 rc=$2"; echo "{\"arm\":\"$ARM\",\"fold\":$F,\"seed\":$S,\"status\":\"failed\",\"step\":\"$1\",\"rc\":$2}" > $ROOT/exit.json; exit $2; }
step input
[ -f $L/ssl/$ARM/run/complete.json ] || fail ssl-not-complete 3
if [ -f $SSL/run/complete.json ] && [ $SSL/run/complete.json -nt $L/ssl/$ARM/run/complete.json ]; then echo "regression_input exists and is newer than the SSL, not rewritten"; else CUDA_VISIBLE_DEVICES= PYTHONPATH=$PP $PY $L/source/$W make_input $ARM || fail input $?; fi
step prepare
if [ -f $ROOT/manifest.json ]; then echo "manifest exists, not prepared again"; else CUDA_VISIBLE_DEVICES= PYTHONPATH=$PP $PY $L/source/$W prepare $A || fail prepare $?; fi
FZ=$ROOT/source/$W
step cpu-tests
if [ -f $ROOT/cpu_tests.json ]; then echo "cpu tests already passed for this manifest"; else CUDA_VISIBLE_DEVICES= PYTHONPATH=$RT $PY $FZ cpu-tests $A || fail cpu-tests $?; fi
step smoke
if [ -f $ROOT/smoke.json ]; then echo "smoke already passed for this manifest"; else CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$RT $PY $FZ smoke $A || fail smoke $?; fi
[ "$STOP" = smoke ] && { step "stopped after smoke"; exit 0; }
step train
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$RT $PY $FZ train $A || fail train $?
step audit
CUDA_VISIBLE_DEVICES= PYTHONPATH=$RT $PY $FZ audit $A || fail audit $?
step done
echo "{\"arm\":\"$ARM\",\"fold\":$F,\"seed\":$S,\"status\":\"completed\"}" > $ROOT/exit.json
