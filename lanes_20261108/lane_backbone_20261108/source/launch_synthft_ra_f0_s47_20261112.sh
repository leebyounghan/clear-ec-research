#!/bin/bash
# D-SYN-007 (coordinator, 2026-11-12 18:10 EST): synthetic-pretrained recipe A fold 0, seed 47, on GPU 2.
# Same recipe, init and data as runs/synthft_ra_f0_s46: init = runs/synthpre_ra_f0_s46/pretrain.pt (the fold-0 synthetic pretraining
# from phase2_crossfold_20261015/fold0/ssl5, 50k synth_v3, one pass, seed 46; shared like the SSL5 is shared by plain recipe A s46/s47),
# then unchanged recipe A (cpdino_b8, 20 epochs, flips, per-epoch predictions e1-20, e20 four-flip TTA), 4 loader workers. Only --seed changes.
# Through the GPU-general guard (NEED 40000 MiB). Never edit this file while it runs.
set -u
cd /workspace/clear-ec-research
LANE=experiments/lane_backbone_20261108
echo $$ > $LANE/runs/synthft_ra_f0_s47.launcher.pid
echo "$(date -u +%FT%TZ) start synthft_ra_f0_s47 (launcher pid $$)" >> $LANE/logs/launch_synthft_ra_f0_s47.log
bash $LANE/source/gpu_launch_20261111.sh 2 40000 "/workspace/venv/bin/python $LANE/source/bb_runner_20261108.py --family cpdino_b8 --fold 0 --seed 47 --root $LANE/runs/synthft_ra_f0_s47 --ssl $LANE/runs/synthpre_ra_f0_s46/pretrain.pt --epochs 20 --micro 4 --tta-final --workers 4" > $LANE/logs/synthft_ra_f0_s47.log 2>&1
echo "$(date -u +%FT%TZ) end synthft_ra_f0_s47 rc=$?" >> $LANE/logs/launch_synthft_ra_f0_s47.log
