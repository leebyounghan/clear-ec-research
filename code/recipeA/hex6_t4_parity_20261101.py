"""Parity of the T4 hex6 wrapper (2026-11-01): original Hex6Corrector (bf16, default SDPA) vs wrapper bf16
(memory-efficient SDPA) vs wrapper fp16 on the 25-image regression fixture; green tolerances HEX 0.5 % rel,
feature 0.02 abs; timing per image for fp16."""
import json, sys, time
import numpy as np, torch
DEPLOY = '/workspace/clear-ec-research/experiments/green_track_20261028/deploy'
sys.path.insert(0, DEPLOY); sys.path.insert(0, sys.argv[1])
import hex6_correction as H6
import hex6_t4_20261101 as X6
z = np.load(DEPLOY + '/regression_fixture_25_v2.npz', allow_pickle=True)
cors = dict(ref=H6.Hex6Corrector(DEPLOY), bf16=X6.Hex6CorrectorT4(DEPLOY, dtype=torch.bfloat16), fp16=X6.Hex6CorrectorT4(DEPLOY, dtype=torch.float16))
res = {k: dict(hex=[], feat=[], per=[]) for k in cors}
times = []
for i in range(len(z['ids'])):
    img, base = z['images'][i], z['base_pred'][i]
    for k, c in cors.items():
        t0 = time.time(); out, info = c(img, base); torch.cuda.synchronize(); dt = time.time() - t0
        if k == 'fp16': times.append(dt)
        res[k]['hex'].append(out[2]); res[k]['feat'].append(np.nan if info['feature'] is None else info['feature']); res[k]['per'].append(info['per_detector'])
exp_hex, exp_feat = z['expected_corrected_hex'], z['expected_feature']
summary = {}
for k in cors:
    h, f = np.array(res[k]['hex']), np.array(res[k]['feat'])
    summary[k] = dict(max_rel_hex_vs_expected=float(np.nanmax(np.abs(h - exp_hex) / exp_hex)), max_abs_feat_vs_expected=float(np.nanmax(np.abs(f - exp_feat))),
                      max_rel_hex_vs_ref=float(np.nanmax(np.abs(h - np.array(res['ref']['hex'])) / np.array(res['ref']['hex']))),
                      n_detector_values_differing_from_ref=int(sum(sum(1 for d in res[k]['per'][i] if not (np.isnan(res[k]['per'][i][d]) and np.isnan(res['ref']['per'][i][d])) and res[k]['per'][i][d] != res['ref']['per'][i][d]) for i in range(len(z['ids'])))))
summary['fp16_within_green_tolerance'] = bool(summary['fp16']['max_rel_hex_vs_expected'] <= 0.005 and summary['fp16']['max_abs_feat_vs_expected'] <= 0.02)
summary['fp16_seconds_per_image_h200'] = round(float(np.median(times[1:])), 3)
summary['peak_gib'] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
json.dump(summary, open(sys.argv[2], 'w'), indent=1)
print(json.dumps(summary, indent=1))
