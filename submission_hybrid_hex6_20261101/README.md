# Submission-1 hybrid package (draft, 2026-10-30)

Equal-weight average of (a) the validated Phase-1 six-member CPSAM package (unchanged app, weights, config and
calibration) and (b) three whole-frame CellposeDINO "recipe A" members trained on all 8,952 positive-label rows
(all-data LeJEPA SSL5 encoder, std pooling, 20 epochs, seeds 46/47/48), each with four-flip TTA and per-target
scalars fitted on the five-fold OOF (CD 0.9908, CV 0.9869, HEX 0.9930).

Evidence: five-fold OOF 8.877 vs 8.951 (recipe A alone) and 9.002 (Phase-1 five); fold4 matched rows 8.798 vs
8.888 (Phase-1 six). See results/recipeA_analysis_20261028/hybrid_oof.json and hybrid_vs_historical_fold4.json.

Layout (image build context):
- Dockerfile, requirements.txt, constraints.txt  — Phase-1 base image (PyTorch 2.11.0 + CUDA 12.6) plus timm 1.0.20
- phase1_app/      — the Phase-1 application files, byte-identical to review_submission_20260910/app
- recipea_src/     — frozen research sources used to build the CPDINO regressor (preflight, pooling, train module)
- runtime/dinov3   — vendored DINOv3 package from the research runtime (copy from the pod:
                     experiments/phase2_foundation_20261002/runtime/dinov3, 3.3 MB) — NOT in git
- hybrid_inference_20261030.py, hybrid_entrypoint.py — the hybrid

Model mount /opt/ml/model (not in git): Phase-1 layout (cpsam_v2, six *.pt, candidate6_full_quality.json) plus
recipea/member_s46.pt, member_s47.pt, member_s48.pt (= experiments/fulldata_recipeA_20261028/regression_std_s*/runs/fold99/cpdino_vitb/epoch_20.pt)
and the CPDINO backbone weights expected by recipea_src (experiments/phase2_foundation_20261002/weights/cellpose/cpdino-vitb).

Status: integration test on the research pod (experiments/submission1_test_20261030) — recipe A path reproduces
stored predictions to 0.07%, peak 2.64 GiB under a 15 GiB cap, ~12 s/image on an H200 (Phase-1 40 views x 6
members dominate). Container build and T4-image test: pending.

Note: the frozen recipe A source resolves the CPDINO backbone at the research-tree relative path
`experiments/phase2_foundation_20261002/weights/cellpose/cpdino-vitb`; the image symlinks that path to
`/opt/ml/model/cpdino/cpdino-vitb` and the entrypoint runs from /opt/app, so the frozen sources stay byte-identical.
