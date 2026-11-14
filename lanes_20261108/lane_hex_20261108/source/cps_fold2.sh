#!/bin/bash
# HEX lane (2026-11-10): fold 2 of the no-SSL multi-target CPSAM 512 failed at start (two HEX-zero validation rows hit the recipe's
# assertion; fixed in family_recipe_copy.py as in the CD lane). Wait for cps_queue2.sh (pid arg 1, running fold 3) to end, move the
# failed fold-2 attempt to archive/, then run fold 2 with the same arguments on GPU 1. Do not edit while it runs.
L=/workspace/clear-ec-research/experiments/lane_hex_20261108; Q=$L/logs/cps_queue2.log; cd /workspace/clear-ec-research
P=$1; echo "$(date -u +%FT%TZ) fold-2 rerun: waiting for queue pid $P" >> $Q; while [ -d /proc/$P ]; do sleep 30; done
[ -f $L/STOP_CPS_QUEUE2 ] && { echo "$(date -u +%FT%TZ) STOP_CPS_QUEUE2 found, fold-2 rerun not started" >> $Q; exit 0; }
mv $L/runs/mtcps_nossl_c512_f2 $L/archive/mtcps_nossl_c512_f2_failed_assert_20261110 2>/dev/null; mv $L/logs/mtcps_nossl_f2.log $L/archive/mtcps_nossl_f2_failed_assert_20261110.log 2>/dev/null
echo "$(date -u +%FT%TZ) start multi-target no-SSL fold 2 (rerun)" >> $Q
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 /workspace/venv/bin/python $L/source/mtcps.py all --family cpsam_c512 --fold 2 \
  --root experiments/lane_hex_20261108/runs/mtcps_nossl_c512_f2 --ssl-epochs 0 --epochs 12 --schedule-epochs 10 > $L/logs/mtcps_nossl_f2.log 2>&1
echo "$(date -u +%FT%TZ) fold 2 exit $?" >> $Q
