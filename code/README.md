# code/

Result folders named below were removed on 2026-11-06; tracked versions remain in git history up to commit `f8bc49b`.

Archive and canonical copy of the CLEAR-EC research code. Experiments never run from this tree: every
Kubernetes job ran from a flat copy under `/workspace/clear-ec-research/experiments/<exp>/source/`, and the
copies that actually ran are kept under `results/<exp>/**/source/`. The layout below dates from 2026-11-03.
Every moved file is byte-identical to its previous version (sha256 in
[`MOVED_20261103.tsv`](MOVED_20261103.tsv)).

## Layout

| Directory | Files | Contents |
|---|---:|---|
| `code/` (root) | 18 | Phase-1 release modules pinned by `training_reproduction/` and the Phase-1 packages, plus the unit tests of three of them. Do not move (see "Pinned files"). |
| `recipeA/` | 36 | Current Phase-2 pipeline: the frozen CPDINO runner chain, recipe A member runners and wrappers, TTA inference, OOF supervisors and aggregation, the GPU queue, T4 and submission-2 checks. |
| `eval/` | 74 | Evaluation harness (`eval_oof_20261031.py`) and the analyses and pre-registered comparisons built on it, plus the recipe A gates (TTA, calibration, pooling bootstrap, seed ensemble, label-noise floor). |
| `green/` | 15 | Point-detector ("green") track, 2026-10-20 to 10-28: contracts, frozen point probes, point features and specialists, the auxiliary-loss A/B and its supervisors, stacking, HEX boundary screen. |
| `reporting/` | 4 | Leaderboard, research catalog and timeline builders (unchanged). |
| `archive/phase1_2026-09/` | 172 | Legacy. Phase 1, 2026-08-30 to 09-23: CPSAM/Cellpose fine-tuning, DINO, region/SSL/target/statres/specialize screens, ensembles, packaging, transfer and submission sweeps, their tests and shell launchers. |
| `archive/phase2_explore_2026-10/` | 53 | Closed Phase-2 lines, 2026-09-28 to 11-02: metric35 SSL, full-native probes, backbone and CPSAM-scale screens, LeJEPA pilots and sweeps, crossfold ablations and audits, full-9000 runners, nested-OOF calibration, single-model and pooling readbacks, unbounded pooling, masked pooling and the style specialist (both rejected). |

All file names are unique across `code/`, so any set of directories can be copied into one flat directory
without collisions.

## Current pipeline (recipe A), as far as the code shows

Recipe A is a whole-frame CellposeDINO (CPDINO ViT-B) regressor with a LeJEPA SSL5 encoder and std pooling.
All entry points below are in `recipeA/` unless another directory is named.

- Frozen chain. `phase2_foundation_preflight_20261002` <- `phase2_foundation_train_20261002` <-
  `phase2_lejepa_regression_20261003`, `phase2_pooling_20261019`; runners `phase2_pooling_regression_20261019`
  (produced the fold-4 seed-46 member) and `phase2_backbone_ssl_regression_20261020` (OOF and full-data members);
  SSL5 runner `phase2_crossfold_ssl_20261015` (imports `phase2_lejepa_20261003`).
- OOF members. `oof_recipeA_20261027_supervisor.py` runs per fold the fold-train-only SSL5
  (`phase2_crossfold_ssl_20261015.py`, or `phase2_crossfold_ssl_zero_policy_20261027.py` on the fold whose
  validation has zero targets) and the std seed-46 regression (`phase2_backbone_ssl_regression_20261020.py`, or
  `recipeA_regression_zero_policy_20261027.py`). `oof_recipeA_20261027_seeds_supervisor.py` adds seeds 47/48;
  `oof_aggregate_v2.py` re-aggregates with the fold-2 v2 root. `phase2_pooling_replication_20261026_supervisor.py`
  is the std-vs-mean replication sweep that adopted std pooling (analysed by `eval/pooling_bootstrap_20261027.py`).
- Full-data members (submission). `phase2_crossfold_ssl_alldata_20261027.py` (all-data SSL5), then
  `recipeA_fulldata_regression_20261028.py` with seeds 46/47/48 (the hybrid package README maps these to
  `member_s46/47/48.pt`).
- Inference. `tta_inference_20261027.py` (four-flip TTA for one completed regression root, writes
  `predictions_tta4.npz`); `phase2_pooling_inference_20261020.py` (single view); `t4_dry_run_20261027.py`
  (T4 memory/precision dry run).
- Gates. `eval/recipeA_analysis_20261028.py` (TTA gate, LOFO scalar calibration),
  `eval/ensemble_vs_historical_20261028.py`, `eval/ssl_seed_gate_20261031.py`, `eval/eval_oof_20261031.py`.
- Hybrid (submission 1). The hybrid OOF script `hybrid_oof_20261028.py` is not in `code/`; it exists only as
  `results/recipeA_analysis_20261028/source/hybrid_oof_20261028.py`. Deployed inference is
  `submission_hybrid_20261030/hybrid_inference_20261030.py`; its `recipea_src/` holds byte-identical copies of
  `recipeA/phase2_pooling_20261019.py`, `phase2_foundation_train_20261002.py` and
  `phase2_foundation_preflight_20261002.py`.
- hex6 (submission 2). Point detectors come from the green track (`green/green_point_features_20261027.py`,
  `green/green_point_specialist*_20261027.py`). The Hex6Corrector (`hex6/deploy/hex6_correction.py`) is not in
  this repository; its T4 wrapper is `submission_hybrid_hex6_20261101/hex6_t4_20261101.py`. Checks:
  `hex6_t4_parity_20261101.py` (wrapper parity), `fold4_shipped_base_20261031.py` (fold-4 shipped composition),
  `test_sub2_20261101.py` (pod integration test of hybrid + hex6).
- Scheduling. `gpu_queue_runner_v2.py` (sequential GPU queue with per-GPU locks and hot-reloaded job specs).

## Added after the move (2026-11-03)

These are not in `MOVED_20261103.tsv`.

- `eval/`: `hybrid_oof_20261028.py` (from `results/recipeA_analysis_20261028/source/`), and the ceiling analyses `ceiling_meta_cpu_20261103.py`, `ceiling_meta_followup_20261103.py`, `stacker_deployable_20261103.py`, `stretch_nested_20261103.py`.
- `recipeA/`: `hybrid_inference_20261030.py`, `hybrid_guard_20261031.py`, `hex6_t4_20261101.py` (copies; the package directories stay canonical for builds), `recipeA_label_fraction_20261103.py`, `recipeA_bridge_capacity_20261103.py`, `trainfit_swa_inference_20261103.py`.
- `green/`: `hex6_correction.py` (from pod `experiments/green_track_20261028/deploy/`).
- `synth/`: procedural endothelium renderer, calibration and dataset generator (route B).

The paragraphs above that call `hybrid_oof_20261028.py` and the hex6 corrector missing predate these additions.

## Bare-name import rule

Modules import each other by bare name (`import phase2_pooling_20261019 as pooling`) and assume they share one
directory, which is how they ran on the pod. A module and everything it imports by bare name therefore belong in
one directory. Where a directory needs another one, the search path is fixed:

| Directory | Search path (in order) |
|---|---|
| `code/`, `recipeA/`, `eval/`, `reporting/` | itself only (closed) |
| `green/` | `green/`, `recipeA/` (6 chain modules) |
| `archive/phase1_2026-09/` | itself, `code/` (10 pinned modules) |
| `archive/phase2_explore_2026-10/` | itself, `recipeA/` (8 chain modules), `code/` (4 pinned modules) |

To rerun anything, stage a flat source directory as the pod did, for example
`cp code/recipeA/*.py code/green/*.py <exp>/source/`. For a local import check,
`PYTHONPATH=code/green:code/recipeA` (or the matching search path) is equivalent. Nothing outside `archive/`
imports from `archive/`, and `recipeA/` imports no other `code/` directory (its imports of package or pod
modules such as `hybrid_inference_20261030` are listed under "Current pipeline"). The per-file cross-directory imports are in the
`cross_dir_imports` column of `MOVED_20261103.tsv`.

Some archived files locate siblings or the repository by their own path and need a flat staging directory (or
the old depth) to rerun; each is marked in the `note` column of the TSV:

- Depth-sensitive (`Path(__file__).parents[1]` or `.parent.parent` as the repository root):
  `fetch_submission_ranges.py`, `fetch_submission_package.py`, `export_submission_package.py`,
  `assemble_ready_submission_model.py`, `fetch_raw_release_20260913.py`, `fetch_ssl_release_20260914.py`,
  `start_package_proxy_server.py`, `audit_deploy_methods_fold.py`, `package_deploy_review.py`,
  `run_deploy_review.py`, `test_deploy_methods.py`, `test_final_candidate.py` (all in `archive/phase1_2026-09/`).
- Same-directory pins (assert or default that pinned sources sit next to them):
  `phase2_full9000_ssl_20261016.py`, `phase2_full9000_readback_20261016.py`,
  `phase2_pooling_readback_20261021.py`, `phase2_single_model_readback_20261015.py`,
  `phase2_crossfold_loss_pair_audit_20261016.py`, `recipeA_masked_pool_regression_20261031.py`,
  `recipeA_masked_pool_backbone_20261031.py` (all in `archive/phase2_explore_2026-10/`).

## Pinned files (kept in `code/`)

| File | Pinned by |
|---|---|
| `next_candidate_20260913.py`, `train_specialize_screen.py`, `ssl_native_core.py`, `train_deploy_region.py` | `training_reproduction/train_ssl_raw.py` (puts `code/` on `sys.path` and checks each module's directory) |
| `final_candidate_core.py`, `method_training_core.py`, `native_region_views.py`, `fullframe_views.py` | `training_reproduction/train_ssl_raw.py` (hashes `code/<name>`); `final_candidate_core.py` also `submission_candidate_20260905/Dockerfile` |
| `audit_submission_ensembles.py` | import closure of `next_candidate_20260913.py` |
| `ssl_native_core.py` | also `docker_review_20260914/scripts/build_technical_report.py` (GitHub link in the released PDF) and `docs/legacy_phase1/SSL_RELEASE_20260914.md` |
| `cache_native.py`, `prepare_fullframe_cache.py`, `audit_raw_id_aliases.py`, `prepare_final9000_candidate.py`, `train_final_candidate.py`, `train_deploy_methods.py` | literal `code/<name>` in `training_reproduction/workspace.py`, `training_reproduction/README.md` and `review_submission_20260910/TRAINING.md`. These resolve against the package's own `training/code/` copy inside a new workspace; kept here conservatively. |
| `test_fullframe_import.py`, `test_ssl_native_core.py`, `test_next_candidate_20260913.py` | unit tests of pinned modules (the first runs `prepare_fullframe_cache.py` via `Path(__file__).with_name`) |

## Old path to new path

The full mapping (296 rows: old path, new path, group, how it moved, sha256, cross-directory imports, notes) is
[`MOVED_20261103.tsv`](MOVED_20261103.tsv). Summary:

- `code/recipeA_20261027/*` was split: runners, supervisors, inference and submission checks went to
  `recipeA/`; `eval_oof_20261031` and the analyses and comparisons to `eval/`; `green_*` to `green/`;
  `recipeA_masked_pool_*`, `masked_pool_probe_20261031` and `recipeA_style_specialist_20261102` to
  `archive/phase2_explore_2026-10/`. The directory no longer exists.
- From the flat `code/`: the nine chain modules above went to `recipeA/`; the seven 2026-10-20/23 green files
  to `green/`; every other `phase2_*`, `analyze_phase2_*`, `nested_oof_*` and 2026-09-28-or-later file to
  `archive/phase2_explore_2026-10/`; everything else that is not pinned to `archive/phase1_2026-09/`.
- Untracked files were moved with `mv` (git had no history for them); tracked and staged files with `git mv`.

## Added 2026-11-04 to 11-05

- `recipeA/`: `recipeA_zoom`, `_invariance`, `_cotwin`, `_canonical`, `_cellmap` wrappers on the frozen runner. Only `_canonical` passed its fold-0 gate.
- `eval/`: `legibility_roi_probe`, `legible_window_*`, `window_*` (image finds the region); `cell_graph_prep`, `cell_graph_sweep`, `graph_ensemble`, `graph_leak_check` (the cell-graph network, current best); `sweep_*` (combination sweep); `member_compare` (gate read-out on fold 0); `id_sequence_probe`, `retrieval_*`, `cotwin_probe`, `abs_*`, `freq_*`, `scale_*`, `perturb_*`, `duplicate_audit`, `swap_scan`, `technician_emulator`, `selection_bias`, `hex6_recompose`, `assumption_audit*` (probes, closed, results in the report).
