#!/bin/bash
L=/workspace/clear-ec-research/experiments/lane_gpu1_20261106
A="bash $L/source/add_arm.sh"
$A cropopt_both_f0 0 --crop 544 --micro 16 --optics measured --pool both
$A cropopt_mean_f0 0 --crop 544 --micro 16 --optics measured --pool mean
$A cropopt_both_s1004_f4 4 --crop 544 --micro 16 --optics measured --pool both --seed 1004
$A cropoptlr3_both_f4 4 --crop 544 --micro 16 --optics measured --pool both --encoder-lr 3e-5
$A cropopt20_both_f4 4 --crop 544 --micro 16 --optics measured --pool both --epochs 20
cut -c1-40 $L/queue.tsv
