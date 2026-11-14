#!/bin/bash
# Reorder the queue as the parent session asked: measured augmentation, then label fractions, then LR x3. Then hand over to the v2 runner.
cd /workspace/clear-ec-research; L=experiments/lane_gpu0_20261106
cp $L/queue.txt $L/logs/queue_before_reorder.txt
grep -v 'measured_v1_lr3' $L/logs/queue_before_reorder.txt > $L/queue.tmp
grep 'measured_v1_lr3' $L/logs/queue_before_reorder.txt >> $L/queue.tmp
mv $L/queue.tmp $L/queue.txt
old=$(ps -eo pid,cmd | grep 'source/queue_gpu0.sh' | grep -v grep | awk '{print $1}')
job=$(ps -eo pid,cmd | grep 'source/run_arm.sh measured_v1' | grep -v grep | awk '{print $1}')
echo "old runner $old, running job $job: $(tr '\0' ' ' < /proc/$job/cmdline)"
kill $old
setsid nohup bash $L/source/queue_gpu0_v2.sh $job > $L/logs/queue_runner_v2.log 2>&1 < /dev/null &
sleep 2
ps -eo pid,ppid,etime,cmd | grep -E 'queue_gpu0|run_arm' | grep -v grep | cut -c1-130
echo ---; cut -c1-125 $L/queue.txt; echo ---; tail -2 $L/logs/queue.log | cut -c1-120
