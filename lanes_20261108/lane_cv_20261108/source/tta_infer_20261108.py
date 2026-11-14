"""CV lane (2026-11-08). Full-data heatmap averaging of the three pre-only detectors (seeds 46/47/48, each fold's own detector on that fold's rows,
the same rows as the shipped preonly peaks). Per frame: 3 seeds x 4 exact flips. Writes peaks in the shipped format (SG.peaks2, >= 0.2, NMS 12) for
sets s46_tta4, s47_tta4, s48_tta4, seeds3_single, seeds3_tta4 under out/tta_peaks/<set>/fold<f>/fold<f>_sg2_peaks.npz.
usage (from /workspace/clear-ec-research, one visible GPU): tta_infer_20261108.py FOLDS (e.g. 04123)"""
import sys, time, json
from pathlib import Path
import numpy as np
sys.path[:0] = ['experiments/green_track_20261028/source']
import torch
from torch.utils.data import DataLoader
import green_point_features_20261027 as G
import green_sg2_specialist_20261028 as SG
import green_selftrain_20261028 as ST
R, Cc = SG.ROWS2, SG.COLS2; torch.set_num_threads(2)
DIRS = {46: 'experiments/green_track_20261028/external_srep/fold{f}', 47: 'experiments/green_track_20261028/external_srep/s47/fold{f}', 48: 'experiments/green_track_20261028/external_srep/s48/fold{f}'}
OUT = Path('experiments/lane_cv_20261108/out/tta_peaks'); SETS = ['s46_tta4', 's47_tta4', 's48_tta4', 'seeds3_single', 'seeds3_tta4']
_, _, ids, yall, _, _ = ST.green(0); support = SG.frame_support()
for f in [int(c) for c in sys.argv[1]]:
    if all((OUT / s / f'fold{f}' / f'fold{f}_sg2_peaks.npz').exists() for s in SETS): print('skip fold', f, flush=True); continue
    va = np.load(f'experiments/green_track_20261028/external_srep/preonly/fold{f}/fold{f}_sg2_peaks.npz', allow_pickle=True)['indices']
    nets = {}
    for seed, d in DIRS.items():
        ck = torch.load(d.format(f=f) + '/sg2_pre.pt', map_location='cpu', weights_only=False)
        enc = G.load_encoder(ST.ENCODERS[f]); enc.load_state_dict(ck['encoder'], strict=True); enc.eval().requires_grad_(False)
        head = SG.Head(seed=seed).cuda(); head.load_state_dict(ck['head']); head.eval(); nets[seed] = (enc, head)
    dl = DataLoader(G.b.Frames(va, yall, len(ids)), batch_size=4, shuffle=False, num_workers=1, pin_memory=True)
    acc = {s: ([], [0]) for s in SETS}; seen = []; t0 = time.time(); n = 0
    for x, _, ix in dl:
        single = {}; tta = {}
        for seed, (enc, head) in nets.items():
            a = np.zeros((len(x), 496, 656), np.float32)
            for fl, (h, v) in enumerate(G.FLIPS):
                m = SG.heat(enc, head, G.flip_image(x, fl)); mv = m[:, :R, :Cc]
                if h: mv = mv[:, :, ::-1]
                if v: mv = mv[:, ::-1, :]
                a[:, :R, :Cc] += mv
                if fl == 0: single[seed] = m
            tta[seed] = a / 4
        for i in range(len(x)):
            mp = {'s46_tta4': tta[46][i], 's47_tta4': tta[47][i], 's48_tta4': tta[48][i], 'seeds3_single': np.mean([single[s][i] for s in nets], 0), 'seeds3_tta4': np.mean([tta[s][i] for s in nets], 0)}
            for s in SETS:
                pk = SG.peaks2(mp[s], support, 0.2).astype(np.float32); acc[s][0].append(pk); acc[s][1].append(acc[s][1][-1] + len(pk))
        seen.append(ix.numpy()); n += len(x)
        if n % 200 < 4: print(json.dumps(dict(fold=f, done=n, of=len(va), sec=round(time.time() - t0))), flush=True)
    seen = np.concatenate(seen); np.testing.assert_array_equal(seen, va)
    for s in SETS:
        d = OUT / s / f'fold{f}'; d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / f'fold{f}_sg2_peaks.npz', indices=va, ids=ids[va], peaks=np.concatenate(acc[s][0]), offsets=np.array(acc[s][1]), point_labels_seen=np.zeros(len(va), bool), store_threshold=0.2, nms_radius=12.0)
    print('fold done', f, round(time.time() - t0), flush=True)
    del nets; torch.cuda.empty_cache()
print('TTADONE', flush=True)
