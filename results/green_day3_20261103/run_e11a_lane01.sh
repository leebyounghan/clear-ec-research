#!/bin/bash
# E11a noisy student (lane01): teacher peaks on folds 0-3 split over GPUs 0/1, student training on GPU 0, score25.
cd /workspace/clear-ec-research
PY=/workspace/venv/bin/python; S=experiments/green_day3_20261103/source/green_day3_e11_nolabel_20261103.py; LOG=experiments/green_day3_20261103/e11_nolabel; mkdir -p $LOG
[ -f $LOG/teacher_peaks_00000.npz ] || CUDA_VISIBLE_DEVICES=0 $PY -u $S pseudo --rows-from 0 --rows-to 4477 > $LOG/pseudo_gpu0.log 2>&1 &
[ -f $LOG/teacher_peaks_04477.npz ] || CUDA_VISIBLE_DEVICES=1 $PY -u $S pseudo --rows-from 4477 --rows-to 8954 > $LOG/pseudo_gpu1.log 2>&1 &
wait
[ -f $LOG/detector_student.pt ] || CUDA_VISIBLE_DEVICES=0 $PY -u $S train-student --updates 4000 > $LOG/train_student.log 2>&1 || { echo "FAILED train-student" >> $LOG/driver_a.log; exit 1; }
CUDA_VISIBLE_DEVICES=0 $PY -u $S score25 --ckpt $LOG/detector_student.pt --tag student > $LOG/score25_student.log 2>&1 || echo "FAILED score25 student" >> $LOG/driver_a.log
echo "e11a done $(date -u)" >> $LOG/driver_a.log
