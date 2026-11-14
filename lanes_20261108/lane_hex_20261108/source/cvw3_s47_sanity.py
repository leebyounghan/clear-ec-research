"""Sanity of the cvw3 seed-47 full-data build (HEX lane, 2026-11-11): 64 random training frames, family_infer over the 5 s47
checkpoints and over all 10 checkpoints (s46 + s47, the deploy input dino_cvw3), against the OOF members (s47 alone, seed mean)."""
import sys, json
import numpy as np
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
sys.path.insert(0, LANE + '/source')
import family_infer as FI
import chain as C
rows_all = C.idx.astype(int); rng = np.random.default_rng(0); pick = np.sort(rng.choice(len(rows_all), 64, replace=False)); rows = rows_all[pick]
ck = lambda s: [f'{LANE}/deploy/cvw3_f99_s{s}/runs/fold99/dinov3_b16/epoch_{e:02d}.pt' for e in range(8, 13)]
raw = 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat'
P47 = FI.predict(ck(47), raw, rows, 8954)[:, 2]; P10 = FI.predict(ck(46) + ck(47), raw, rows, 8954)[:, 2]
o47 = np.log(C.cnn5_hex('lane_hex_20261108/runs/cvw3_f{f}_s47', 'dinov3_b16'))[pick]
om = np.mean([np.log(C.cnn5_hex(r, 'dinov3_b16')) for r in ('lane_cv_20261108/arms/dinov3_cvw3_f{f}_s46', 'lane_hex_20261108/runs/cvw3_f{f}_s47')], 0)[pick]
rep = dict(s47_full_vs_oof_r=float(np.corrcoef(P47, o47)[0, 1]), s47_mean_log_ratio=float(np.mean(P47 - o47)),
           ten_ckpt_vs_oof_seedmean_r=float(np.corrcoef(P10, om)[0, 1]), ten_ckpt_mean_log_ratio=float(np.mean(P10 - om)))
print(json.dumps(rep), flush=True); json.dump(rep, open(LANE + '/logs/cvw3_s47_sanity.json', 'w'))
