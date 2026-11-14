#!/bin/bash
L=/workspace/clear-ec-research/experiments/lane_gpu1_20261106; cd $L
# Move the three short arms to the second stream; the main queue keeps the running fold 0 arms and gets the end-to-end runs. Queue files are re-read on each cycle, so an atomic replace is safe.
grep -P '^(cropopt_both_s1004_f4|cropoptlr3_both_f4|cropopt20_both_f4)\t' queue.tsv > queue_short.tsv
grep -vP '^(cropopt_both_s1004_f4|cropoptlr3_both_f4|cropopt20_both_f4)\t' queue.tsv > queue.tmp
printf 'e2e_cropopt_both_f4\tbash %s/source/run_e2e.sh cropopt_both_f4 4 46\n' $L >> queue.tmp
printf 'e2e_cropopt_both_f0\tbash %s/source/run_e2e.sh cropopt_both_f0 0 46\n' $L >> queue.tmp
mv queue.tmp queue.tsv
cd /workspace/clear-ec-research
(setsid nohup /workspace/venv/bin/python -u $L/source/queue_short_20261107.py > $L/logs/queue_short.log 2>&1 < /dev/null &)
sleep 2
cut -c1-70 $L/queue.tsv; echo ---; cut -c1-60 $L/queue_short.tsv
pgrep -af 'queue_gpu1|queue_short' | cut -c1-110
sed -n 1,12p $L/../lane_gpu0_20261106/source/recipeA_measured_aug2_20261106.py | cut -c1-500
sed -n 23,110p $L/../lane_gpu0_20261106/source/recipeA_measured_aug2_20261106.py | cut -c1-170
