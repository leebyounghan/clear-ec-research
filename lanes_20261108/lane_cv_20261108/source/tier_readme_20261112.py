"""D-PKG-002: write README_tier.json into each CV tier deploy folder (model files, dino term, strict value, L5)."""
import json
from pathlib import Path
L = Path('/workspace/clear-ec-research/experiments/lane_cv_20261108')
common = {
 "shared_with_CD_HEX": ["recipe A s46+s47 epoch 20 TTA4: experiments/fulldata_recipeA_20261028/regression_std_s{46,47}/runs/fold99/cpdino_vitb/epoch_20.pt (A_cv; the s46 file is also the token encoder of the frozen heads)",
                        "P1 five members + CPSAM backbone (h_cv), as in the D-CD-001 manifest",
                        "pre-only detectors experiments/green_track_20261028/deploy/detectors/sg2pre_s{46,47,48}.pt, 4 flips each = 12 passes (CV GRAPH, quartets, simulation-embedding ridge; HEX uses the same detectors)"],
 "cv_only": ["experiments/lane_cv_20261108/graph_models/gfull_{noctx_s0..s9,pre47_s0..s2,pre48_s0..s2}_s<k>_e{21..25}.pt (80 files, 40.5 MB): GRAPH base",
             "experiments/meta_ceiling_20261103/sweep/sim_pred/sim2_l8_s{0,1,2}.pt (3 files, 2.3 MB): simulation embedding for the ridge",
             "experiments/lane_cv_20261108/frozen_heads_full/{std,spectrum,cov_sqrt,autocorr}_s{0,1,2}_e{16..20}.pt (60 files, 100.5 MB): frozen heads"]}
quart = ["experiments/lane_cv_20261108/graph_models/full_tta_{s46,s47,s48,seeds3}_tta4_s{0,1,2}_e{21..25}.pt (60 files, 30.5 MB): plain graph quartet",
         "experiments/lane_cv_20261108/graph_models/gord_{s46,s47,s48,seeds3}_tta4_full_s<k>_s<k>_e{21..25}.pt (60 files, 30.4 MB): order-loss graph quartet"]
s46b = "experiments/lane_cd_20261108/arms/dinov3_b16_f99_s46b/runs/fold99/dinov3_b16/epoch_{e}.pt"
T = {
 "deploy_t1": dict(tier="t1", stack="out/cv_tier_t1_20261112.npz", strict_cv=10.1115,
   dino_term="dino_logcv = mean over epochs 16-20 of log CV, single view (no flip), DINOv3-B s46b only (the CD lane's f99 model; the identity flip of its D TTA4 forwards)",
   dino_files=[s46b.format(e=e) for e in range(16, 21)], removed_vs_freeze="lane_cv_20261108/arms/dinov3_b16_f99_s{47,49}/runs/fold99/dinov3_b16/epoch_{16..20}.pt (10 files, 3.43 GB, about 5.5 s per frame)"),
 "deploy_t2": dict(tier="t2 (= t1, identical parameter files)", stack="out/cv_tier_t1_20261112.npz", strict_cv=10.1115,
   dino_term="as deploy_t1", dino_files=[s46b.format(e=e) for e in range(16, 21)], removed_vs_freeze="as deploy_t1",
   note="no cheap middle step: with the CD lane's D loaded (CD t1/t2), t1 costs CV nothing extra; s46b + the CD lane's s47 (also in D) reads 10.1167 strict, not better"),
 "deploy_t3": dict(tier="t3 (pick)", stack="out/cv_tier_t3_e20_20261112.npz", strict_cv=10.1138,
   dino_term="dino_logcv = log CV of DINOv3-B s46b epoch 20 only, single view (one checkpoint, one forward); for a CD tier without D",
   dino_files=[s46b.format(e=20)], removed_vs_freeze="all DINOv3-B files except s46b epoch_20.pt (14 of 15 files, about 4.8 GB)"),
 "deploy_t3_alt": dict(tier="t3 alternative", stack="out/cv_tier_t3_nodino_20261112.npz", strict_cv=10.1238,
   dino_term="none: no DINOv3-B term (params components have no dino entry)", dino_files=[], removed_vs_freeze="every DINOv3-B file (15 files, about 5.15 GB)")}
for d, t in T.items():
    t.update(decision="D-PKG-002", incumbent_same_run=10.1160, convention="strict five folds (cv_search procedure), incumbent D-CV-001 recomputed by the same code",
             model_files=dict(dino=t["dino_files"], quartets=quart, **common),
             entry="source/cv_apply_20261110.py apply(P, members) with P = this folder; member keys as in the freeze (dino_logcv only when components has dino)",
             l5="nested procedure with no outer fold reproduces this folder: weights from the stacker and the strict objective both equal params weights; stretch max abs diff <= 1.1e-16; cv_apply on the per-fold inputs reproduces the strict OOF to <= 3.3e-16 (out/l5_" + t["stack"][4:-4] + ".json, logs/tier_deploy_*.log)")
    json.dump(t, open(L / d / "README_tier.json", "w"), indent=1)
print("ok")
