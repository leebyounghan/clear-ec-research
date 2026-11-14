"""200 labelled rows for the D-GPU-001 score-impact runs (deploy lane, 2026-11-12): fixed-seed random draw from the rows with all three labels > 0
(best_pipeline label table, same order as the CD lane's OOF files and test_t4's zc). Writes inference/rows200_20261112.npy."""
import sys, numpy as np
R = '/workspace/clear-ec-research/experiments/'; L = R + 'lane_cellroute_20261109/'
sys.path[:0] = [R + 'lane_hex_20261108/source', R + 'meta_ceiling_20261103/source', L + 'source', R + 'lane_cv_20261108/source']
import best_pipeline_20261106 as BP
zc = np.load(R + 'lane_cd_20261108/reads/freeze_cd_oof_20261111.npz'); assert (zc['idx'].astype(int) == BP.idx.astype(int)).all()
ok = np.where((np.asarray(BP.y) > 0).all(1))[0]
rows = np.sort(np.random.default_rng(20261112).choice(ok, 200, replace=False)); np.save(L + 'inference/rows200_20261112.npy', rows)
print(len(ok), rows[:10], np.bincount(np.asarray(BP.fold)[rows].astype(int)))
