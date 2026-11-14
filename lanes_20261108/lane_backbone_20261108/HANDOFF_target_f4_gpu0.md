# Hand-off: target-denoising fold 4 on GPU 0 (backbone lane → CD lane, 2026-11-10 02:00 EST)

The coordinator offered GPU 0 from about 06:30 EST 11-10. The backbone lane chose GPU 0 for target fold 4 (both versions) so it runs alongside fold 0 on GPU 3. These jobs are removed from the backbone lane's GPU 3 queues; nothing else will start them. Run each once, from `/workspace/clear-ec-research`; do not edit the script. The two can run at the same time (about 36 GiB each at micro 4) or one after the other. They take about 5 h each alone, or about 10 h together.

```
CUDA_VISIBLE_DEVICES=0 /workspace/venv/bin/python experiments/lane_backbone_20261108/source/bb_runner_20261108.py --fold 4 --seed 46 --root experiments/lane_backbone_20261108/runs/target_soft05_f4_s46 --epochs 12 --teacher experiments/lane_backbone_20261108/data/teacher_nestedLF_f4_20261109.npz --teacher-a 0.5 --ema 0.998,0.999 --swa-start 6 --tta-final --workers 2
```

```
CUDA_VISIBLE_DEVICES=0 /workspace/venv/bin/python experiments/lane_backbone_20261108/source/bb_runner_20261108.py --fold 4 --seed 46 --root experiments/lane_backbone_20261108/runs/target_rowweight_f4_s46 --epochs 12 --row-weight experiments/lane_backbone_20261108/data/rowweight_nestedLF_f4_20261109.npz --ema 0.998,0.999 --swa-start 6 --tta-final --workers 2
```

- Same recipe as fold 0 (on GPU 3 at micro 2; micro only splits the effective batch of 16 into chunks).
- Teacher: strictly label-free nested table from the cell-route lane, fit outside {4, g} for a row in fold g.
- Outputs: `runs/target_*_f4_s46/{raw,ema998,ema999,swa6}/runs/fold4/cpdino_b8/`. The root gets `exit.json` when a run ends. Please tell the backbone lane at start and end.
