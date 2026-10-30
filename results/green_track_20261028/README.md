# green_track_20261028

GPU 3 green-track work (points + ROI boxes from 25 expert-annotated images). One directory per
experiment with protocol.json written before running, frozen source in source/, result.json after.
Ledger: ledger.json. Queue: gpu3_jobs.json (runner: experiments/queue_runner_20261027/gpu_queue_runner_v2.py),
logs in queue_gpu3/. Baseline for every claim: recipe A seed-46 four-flip TTA + LOFO scalars, OOF 8.985
(8.9757 on the 8934 rows without leaked point labels). Shared frozen modules copied from
experiments/green_point_features_20261027/source.
