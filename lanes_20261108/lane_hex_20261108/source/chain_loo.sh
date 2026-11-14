#!/bin/bash
# Leave-one-out over a chain (coordinator 2026-11-11): for each step, the chain with that single step removed and every other step
# forced kept; 4 parallel workers on the CPU pod. usage: chain_loo.sh TAG STEP1 STEP2 ...   -> logs/loo_TAG_<i>.log, logs/loo_TAG.summary
cd /workspace/clear-ec-research/experiments/lane_hex_20261108; TAG=$1; shift; ST=("$@"); n=${#ST[@]}
for i in $(seq 0 $((n - 1))); do
  rest=(); for j in $(seq 0 $((n - 1))); do [ $j -ne $i ] && rest+=("${ST[$j]}"); done
  ( CHAIN_FORCE=1 CHAIN_NORESULTS=1 CHAIN_OUT=chain_loo_${TAG}_$i OMP_NUM_THREADS=1 /workspace/venv/bin/python source/chain.py "${rest[@]}" > logs/loo_${TAG}_$i.log 2>&1; echo "removed ${ST[$i]}: $(grep -E '^\{"step"' logs/loo_${TAG}_$i.log | tail -1 | cut -c1-120)" >> logs/loo_${TAG}.summary ) &
  while [ $(jobs -rp | wc -l) -ge 4 ]; do sleep 5; done
done
wait; echo "done" >> logs/loo_${TAG}.summary
