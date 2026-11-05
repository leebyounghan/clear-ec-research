"""Pod integration test for submission 2 (hybrid B + hex6), 2026-11-01. One GPU, 15 GiB allocator cap (T4 bound).
Same 60 fold-4 MHAs as the submission-1 n60 test (rng 20261030). Per image: guarded hybrid (fp16) must reproduce
the submission-1 n60 hybrid; hex6 (T4 wrapper, fp16, memory-efficient SDPA, ten deployed detectors resident)
must leave CD/CV unchanged and move HEX; records time and peak memory for the whole submission-2 path."""
import json, sys, time
from pathlib import Path
import numpy as np, SimpleITK as sitk, torch
T = Path(sys.argv[1]); sys.path.insert(0, str(T / 'source'))
sys.path.insert(0, 'experiments/submission1_test_20261030/source')
sys.path.insert(0, 'experiments/green_track_20261028/deploy')
import hybrid_inference_20261030 as H
import hybrid_guard_20261031 as G
import hex6_t4_20261101 as X6
E = Path('experiments'); MHA = Path('datasets/clear_ec_fullframe_20260904/train_mha')
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(15 * 2**30 / total)
H.assert_memory_efficient_sdpa()
t0 = time.perf_counter()
corr = X6.Hex6CorrectorT4('experiments/green_track_20261028/deploy', dtype=torch.float16)
load_s = time.perf_counter() - t0
r = np.load(E / 'target_fold4_20260912/preflight/reference.npz'); ids, y = r['ids'].astype(str), r['target'].astype(np.float64)
pick = np.sort(np.random.default_rng(20261030).choice(len(ids), 60, replace=False))
n60 = {row['id']: row for row in json.load(open(E / 'submission1_test_20261030/n60/result.json'))['rows']}
phase1_dir = E / 'ssl_release_20260914/model'
members = [str(E / f'fulldata_recipeA_20261028/regression_std_s{s}/runs/fold99/cpdino_vitb/epoch_20.pt') for s in (46, 47, 48)]
rows = []
for k in pick:
    image = sitk.GetArrayFromImage(sitk.ReadImage(str(MHA / f'{ids[k]}.mha')))
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
    v, parts = G.predict_guarded(image, phase1_dir, 'candidate6_full_quality.json', members, torch.float16)
    torch.cuda.synchronize(); t_h = time.perf_counter() - t0
    c, info = corr(G.coerce_image(image)[0], v)
    torch.cuda.synchronize(); t_all = time.perf_counter() - t0
    ref = np.array(n60[ids[k]]['hybrid'])
    rows.append(dict(id=ids[k], mode=parts['mode'], errors=parts['errors'], hybrid=v.tolist(), corrected=c.tolist(), target=y[k].tolist(),
                     rel_vs_sub1_n60=float(np.max(np.abs(v - ref) / ref)), cdcv_unchanged=bool(c[0] == v[0] and c[1] == v[1]),
                     hex_change_pct=float((c[2] / v[2] - 1) * 100), hex6_fallback=bool(info['fallback']), feature=info['feature'],
                     seconds_hybrid=t_h, seconds_total=t_all, peak_gib=torch.cuda.max_memory_allocated() / 2**30))
a = lambda k: np.array([x[k] for x in rows])
tgt = np.array([x['target'] for x in rows]); hy = np.array([x['hybrid'] for x in rows]); co = np.array([x['corrected'] for x in rows])
ok = tgt[:, 2] > 0
res = dict(n=len(rows), corrector_load_seconds=round(load_s, 1), all_mode_hybrid=bool(all(x['mode'] == 'hybrid' and not x['errors'] for x in rows)),
           max_rel_vs_sub1_n60=float(a('rel_vs_sub1_n60').max()), cdcv_unchanged_all=bool(a('cdcv_unchanged').all()), hex6_fallbacks=int(a('hex6_fallback').sum()),
           hex_change_pct=dict(median_abs=float(np.median(np.abs(a('hex_change_pct')))), max_abs=float(np.abs(a('hex_change_pct')).max())),
           seconds_median=dict(hybrid=float(np.median(a('seconds_hybrid')[1:])), total=float(np.median(a('seconds_total')[1:]))),
           peak_gib_max=float(a('peak_gib').max()),
           sample_hex_mape=dict(hybrid=float(np.mean(np.abs(hy[ok, 2] - tgt[ok, 2]) / tgt[ok, 2]) * 100), corrected=float(np.mean(np.abs(co[ok, 2] - tgt[ok, 2]) / tgt[ok, 2]) * 100)),
           note='fold-4 images are training images of the full-data members and of the deployed detectors; sample MAPE is a sanity number, not an estimate',
           rows=rows)
json.dump(res, open(T / 'n60_sub2_result.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'rows'}, indent=1))
