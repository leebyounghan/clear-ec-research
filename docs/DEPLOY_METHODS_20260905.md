# Confirmed-method deployment fits — September 5, 2026

## Active execution

Started September5 **14:09:46 UTC /23:09:46 KST** on the existing one-node eight-H200 pod. Both one-update GPU smoke jobs passed; full fits started14:09:56 UTC on GPU0/1. The remaining GPUs are intentionally unassigned because the deployable candidate needs exactly these two frozen recipes. No pod lifecycle or challenge upload/submission operation is part of this queue.

## Frozen inputs and selection

The completed five-fold screen confirmed both methods. The all-data jobs use the separately reconciled9,000-image cache, excluding the same two nonpositive-HEX rows, for8,998 eligible training rows. Existing three all-data models and their staged hashes remain untouched.

| Recipe | Five OOF best epochs | Frozen median OOF update | Approx full-data epochs touched |
|---|---|---:|---:|
| metric35 | 26,34,26,35,30 | 4,470 | 23.78 |
| predbag2_35 | 26,29,21,29,20 | 3,874 | 20.61 |

The optimizer-update count, seed42, original CPSAM initialization, label processing, optimizer and cosine LR update-index trajectory are fixed before full-data fitting. Full-data batch size is48 and the partial last batch is included. There is no all-data validation score; training loss does not select a checkpoint. The metric model uses the mixed standardized-log/MAPE loss. The prediction-bag model averages two independent crop head outputs before log-L1 loss. Checkpoints are written separately under `experiments/deploy_methods_20260905`; the completed September4 candidate is not overwritten.

## Automatic postflight

After both fits finish, the same pod controller stages:

- candidate4: existing three seeds plus metric35;
- candidate5: existing three seeds plus metric35 and predbag2_35.

Each configuration receives a global scalar calibration fitted to its historically reused five-fold OOF predictions. The real MHA/three-JSON entrypoint then runs candidate4 and candidate5 in BF16 and FP16, recording cold-process wall time and finite-positive outputs. Strict state loading checks that both k1 and k2-trained checkpoints share the validated inference architecture.

Successful postflight means native staging only. It does not mean container-built, network-isolated, target-device certified, uploaded or submitted. Candidate5's five-fold raw/internal cross-fit values9.035524/9.004594 motivated this step, but neither is a leaderboard forecast or an all-data model score.

## Timing

The prior all-data k1 jobs measured about1.4345s/update;4,470 updates project to1h47m before checkpoint overhead. The k2 fit is expected to dominate at roughly3–3.5h. Provisional total completion is September5 17:30–18:00 UTC / September6 02:30–03:00 KST, followed by a few minutes of native inference testing. These are planning estimates and will be replaced by measured progress.

Implementation: `code/train_deploy_methods.py`, `code/run_deploy_methods.py`, `code/package_deploy_methods.py`, `submission_candidate_20260905/inference.py`. Three focused CPU/interface tests and two live GPU smoke jobs passed before full training. The OpenAI Docs-guided 30-minute task monitor was reactivated and only reports meaningful changes.
