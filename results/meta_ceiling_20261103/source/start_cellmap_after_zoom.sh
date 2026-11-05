#!/bin/bash
# When the zoom run has finished, stops the 2,3 lane runner (pid 93, cmdline checked) so it cannot wait on the sweep's GPU 3, and starts the cell-map run on GPUs 2,3, sharing GPU 3 with the sweep queue.
A=/workspace/clear-ec-research/experiments/meta_ceiling_20261103
until [ -f $A/zoom/f0_s46/audit.json ] || grep -q ^FAILED $A/zoom/f0_s46/driver.log; do sleep 30; done
if tr '\0' ' ' < /proc/93/cmdline | grep -q lane23_jobs_1104; then kill 93; fi
sleep 5
cd /workspace/clear-ec-research; CUDA_VISIBLE_DEVICES=2,3 bash $A/run_cellmap.sh 0
