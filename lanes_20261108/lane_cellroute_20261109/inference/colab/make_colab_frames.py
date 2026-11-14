"""5 of the 20 deploy frames (positions 0, 4, 9, 14, 19 of every budget / packaged / test_t4 run) as Grand Challenge .mha inputs for the real-T4
check (deploy lane, 2026-11-12, D-PKG-006). Written as test_gc_jobs.py writes them (SimpleITK, raw uint8 972 x 1296)."""
import os, numpy as np, SimpleITK as sitk
ROOT = '/workspace/clear-ec-research/'; O = ROOT + 'experiments/lane_cellroute_20261109/inference/colab/frames/'; os.makedirs(O, exist_ok=True)
zc = np.load(ROOT + 'experiments/lane_cd_20261108/reads/freeze_cd_oof_20261111.npz')
MM = np.memmap(ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
rows = np.linspace(0, len(zc['idx']) - 1, 20).astype(int)
for i in (0, 4, 9, 14, 19):
    d = int(zc['idx'][rows[i]]); sitk.WriteImage(sitk.GetImageFromArray(np.asarray(MM[d])), O + f'{i:02d}_{d}.mha'); print(i, d)
