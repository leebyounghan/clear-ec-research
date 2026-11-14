"""CV lane (2026-11-08). Flip TTA and seed ensembling of the pre-only detectors, read on the 25 expert frames.
Each frame uses the pre-only detectors of its own fold (seeds 46/47/48: external_srep/{fold,s47,s48}/fold{f}/sg2_pre.pt). Heatmaps: one pass,
mean of 4 exact flips, mean over 3 seeds, mean over seeds x flips. Peaks as shipped (SG.peaks2, >= 0.2, NMS 12). Read as peak_select25:
miss/FP/jitter/F1 in hull, hull CV statistics r and LOO MAPE, for plain t0.35 / t0.2 and spacing-prior NMS a0.7 t0.2. Also times one pass.
Run from /workspace/clear-ec-research with one visible GPU."""
import sys, time, json
import numpy as np
sys.path[:0] = ['experiments/green_track_20261028/source', 'experiments/lane_cv_20261108/source']
import torch
import peak_select25_20261108 as P
import green_point_features_20261027 as G
import green_sg2_specialist_20261028 as SG
import green_selftrain_20261028 as ST
from scipy.spatial import Delaunay
R, Cc = SG.ROWS2, SG.COLS2
DIRS = {46: 'experiments/green_track_20261028/external_srep/fold{f}', 47: 'experiments/green_track_20261028/external_srep/s47/fold{f}', 48: 'experiments/green_track_20261028/external_srep/s48/fold{f}'}
names = P.names; rows = [P.pos[n] for n in names]; fold = P.fold; dsidx = [int(P.idx[r]) for r in rows]
_, _, ids, yall, _, _ = ST.green(0); support = SG.frame_support()
maps = {}  # (name, seed) -> (single, tta)
t_pass = []
for f in sorted(set(int(fold[r]) for r in rows)):
    sel = [k for k, r in enumerate(rows) if int(fold[r]) == f]
    frames = G.b.Frames(np.array([dsidx[k] for k in sel]), yall, len(ids))
    for seed, d in DIRS.items():
        ck = torch.load(d.format(f=f) + '/sg2_pre.pt', map_location='cpu', weights_only=False)
        enc = G.load_encoder(ST.ENCODERS[f]); enc.load_state_dict(ck['encoder'], strict=True); enc.eval().requires_grad_(False)
        head = SG.Head(seed=seed).cuda(); head.load_state_dict(ck['head']); head.eval()
        for j, k in enumerate(sel):
            x = frames[j][0][None]; acc = np.zeros((496, 656), np.float32); single = None
            for fl, (h, v) in enumerate(G.FLIPS):
                xf = G.flip_image(x, fl); torch.cuda.synchronize(); t0 = time.time(); m = SG.heat(enc, head, xf)[0]; torch.cuda.synchronize(); t_pass.append(time.time() - t0)
                mv = m[:R, :Cc]
                if h: mv = mv[:, ::-1]
                if v: mv = mv[::-1, :]
                acc[:R, :Cc] += mv
                if fl == 0: single = m.copy()
            maps[(names[k], seed)] = (single, acc / 4)
        del enc, head; torch.cuda.empty_cache()
print('seconds per pass median', round(float(np.median(t_pass[5:])), 4), flush=True)
def peaks_of(m): return SG.peaks2(m, support, 0.2)
variants = {
    's46 single': lambda n: maps[(n, 46)][0], 's46 tta4': lambda n: maps[(n, 46)][1],
    'seeds3 single': lambda n: np.mean([maps[(n, s)][0] for s in DIRS], 0), 'seeds3 tta4': lambda n: np.mean([maps[(n, s)][1] for s in DIRS], 0)}
out = {}
for vn, fn in variants.items():
    pks = {n: peaks_of(fn(n)) for n in names}
    for sn, sel_fn in {'t0.35': lambda pk, s: pk[pk[:, 2] >= 0.35, :2].astype(float), 't0.2': lambda pk, s: pk[:, :2].astype(float), 'sp07 t0.2': lambda pk, s: P.select(pk, s, t_hi=0.2, alpha=0.7), 'sp06 t0.2': lambda pk, s: P.select(pk, s, t_hi=0.2, alpha=0.6)}.items():
        mm, vals = [], []
        for n, mk, r in zip(names, P.MK, rows):
            pk = pks[n]; s = P.spacing_from_cd(P.B[r, 0]); q = sel_fn(pk, s); hull = Delaunay(mk); qi = q[hull.find_simplex(q) >= 0]; sp_m = float(np.median(P.cKDTree(mk).query(mk, k=2)[0][:, 1]))
            mm.append(P.match(qi, mk, sp_m)); vals.append(P.stats(qi))
        mm = np.array(mm); m = dict(miss=round(float(np.nanmean(mm[:, 0])), 3), fp=round(float(np.nanmean(mm[:, 1])), 3), jit=round(float(np.nanmean(mm[:, 2])), 3), f1=round(float(np.nanmean(mm[:, 3])), 3)); rd = P.read(vals)
        out[f'{vn} {sn}'] = dict(match=m, cv=rd); print(f'{vn} {sn}'.ljust(26), json.dumps(m), json.dumps(rd), flush=True)
json.dump(out, open('experiments/lane_cv_20261108/out/tta25.json', 'w'), indent=1)
# stored check: s46 single t0.35 should match the shipped peaks for these frames
d = [abs(len(peaks_of(maps[(n, 46)][0])) - len(P.PK[int(P.idx[P.pos[n]])])) for n in names]; print('count diff vs shipped peaks (s46 single)', d, flush=True)
