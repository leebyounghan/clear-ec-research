"""CD lane read over whatever folds are available (2026-11-08). usage: cd_read_20261108.py SET [SET ...]
SET = comma list of member keys (see MEMBERS). Rows = folds where every member of the set exists. For each such fold f, blend weights
(P1, A2, F = geometric mean of the set's members, epochs 16-20 log mean) and the stretch are fit on the other available folds only,
then applied to f. With all five folds this is the best pipeline's own nesting. D (detector density on source B rows) stacked as in screen 3.
Reference: best pipeline CD (4-fold nested stretch) on the same rows."""
import sys, json, numpy as np
from cd_common import *
pos = {int(i): k for k, i in enumerate(idx)}
b = E.load_base(); seeds, single, hist = C.load_members(b)
sc = np.array([json.load(open(X / 'recipeA_analysis_20261028/hybrid_oof.json'))['recipeA_lofo_scalars'][str(f)][0] for f in fold])
A2 = 0.5 * (seeds[46][:, 0] + seeds[47][:, 0]) * sc; P1 = b['P1'][:, 0]
import os
if os.environ.get('A2SNAP') == '1':          # half tta4 epoch 20, half single-view epochs 8-19 log mean, per seed (cd_snapshot_20261108)
    def _rec(seed):
        out = np.full(len(yc), np.nan)
        for f, root in C.RECIPE[seed].items():
            d = X / root / 'runs' / f'fold{f}' / 'cpdino_vitb'; L = []
            for e in range(8, 20):
                z = np.load(d / f'predictions_epoch_{e:02d}.npz'); p = np.full(len(yc), np.nan); p[[pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]; L.append(np.log(p))
            m = np.exp(np.mean(L, 0)); out[fold == f] = m[fold == f]
        return out
    A2 = 0.5 * (np.exp(0.5 * np.log(seeds[46][:, 0]) + 0.5 * np.log(_rec(46))) + np.exp(0.5 * np.log(seeds[47][:, 0]) + 0.5 * np.log(_rec(47)))) * sc
    print('A2 snapshot on, alone CD', round(cd(A2), 4), flush=True)
G3, G2, CDL = 'lane_gpu3_20261106/runs', 'lane_gpu2_20261106/runs', 'lane_cd_20261108/arms'
def roots(fam, seed, tag=''):
    r = {}
    for f in range(5):
        cands = [f'{CDL}/{fam}{tag}_f{f}_s{seed}', f'{G3}/{fam}{tag}_f{f}_s{seed}', f'{G2}/{fam}{tag}_f{f}_s{seed}']
        for c in cands:
            if (X / c / f'runs/fold{f}/{fam}/complete.json').exists(): r[f] = c; break
    return r
MEMBERS = {'dinov3_s46': ('dinov3_b16', 46, ''), 'dinov3_s47': ('dinov3_b16', 47, ''), 'convnext_s46': ('convnext_b32', 46, ''),
           'dinov2_s46': ('dinov2_b14', 46, ''), 'convnext_cdw3_s46': ('convnext_b32', 46, '_cdw3')}
def member(key):
    out = np.full(len(yc), np.nan)
    if key.endswith('_tta'):                      # four-flip TTA over epochs 16-20, files in lane_cd/tta/<name>_f<fold>_tta4_e16_20.npz
        for f in range(5):
            pth = LANE / 'tta' / f'{key[:-4]}_f{f}_tta4_e16_20.npz'
            if pth.exists():
                z = np.load(pth); out[[pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]
        return out
    if key in ('vitl_s46', 'vitl_s46_tt', 'vitl_s47', 'vitl_s47_tt', 'vitl_s48', 'vitl_s48_tt'):   # ViT-L/16 raw track: s46 folds 0/4 from the backbone lane (read-only), rest from this lane
        sd = key.split('_')[1]
        for f in range(5):
            r = (X / f'lane_backbone_20261108/runs/vitl16_avg_f{f}_s46' if (f in (0, 4) and sd == 's46') else LANE / f'arms/vitl16_f{f}_{sd}') / f'raw/runs/fold{f}/dinov3_l16t'
            if not (r / 'complete.json').exists(): continue
            L = []
            for e in range(8, 13):
                z = np.load(r / f'predictions_epoch_{e:02d}.npz'); p = np.full(len(yc), np.nan); p[[pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]; L.append(np.log(p))
            m = np.mean(L, 0)
            if key.endswith('_tt') and (r / 'predictions_final_tta.npz').exists():
                z = np.load(r / 'predictions_final_tta.npz'); p = np.full(len(yc), np.nan); p[[pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]
                m = 0.5 * np.mean(L[:4], 0) + 0.5 * np.log(p)
            out[fold == f] = np.exp(m[fold == f])
        return out
    if key in ('win2x_s46', 'win2x_s46_tt'):      # backbone lane: recipe A (CPDINO s8, SSL5) on the 2x technician-window crop, raw track (read-only)
        for f in range(5):
            r = X / f'lane_backbone_20261108/runs/recipeA_win2x_f{f}_s46/raw/runs/fold{f}/cpdino_b8'
            if not (r / 'predictions_final_tta.npz').exists(): continue   # these runs write no complete.json; final TTA marks the end
            L = []
            for e in range(8, 13):
                z = np.load(r / f'predictions_epoch_{e:02d}.npz'); p = np.full(len(yc), np.nan); p[[pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]; L.append(np.log(p))
            m = np.mean(L, 0)
            if key.endswith('_tt') and (r / 'predictions_final_tta.npz').exists():
                z = np.load(r / 'predictions_final_tta.npz'); p = np.full(len(yc), np.nan); p[[pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]
                m = 0.5 * np.mean(L[:4], 0) + 0.5 * np.log(p)
            out[fold == f] = np.exp(m[fold == f])
        return out
    if key in ('vith_s46', 'vith_s46_tt'):        # DINOv3 ViT-H+/16 raw track (same read as ViT-L), lane_cd arms/vith16_f<f>_s46 (folds run on GPUs 0, 2, 3)
        for f in range(5):
            r = LANE / f'arms/vith16_f{f}_s46/raw/runs/fold{f}/dinov3_hp16t'
            if not (r / 'complete.json').exists(): continue
            L = []
            for e in range(8, 13):
                z = np.load(r / f'predictions_epoch_{e:02d}.npz'); p = np.full(len(yc), np.nan); p[[pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]; L.append(np.log(p))
            m = np.mean(L, 0)
            if key.endswith('_tt') and (r / 'predictions_final_tta.npz').exists():
                z = np.load(r / 'predictions_final_tta.npz'); p = np.full(len(yc), np.nan); p[[pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]
                m = 0.5 * np.mean(L[:4], 0) + 0.5 * np.log(p)
            out[fold == f] = np.exp(m[fold == f])
        return out
    if key == 'vitl_s4647_tt':                    # ViT-L s46 and s47 (tt form), geometric mean
        return np.exp(0.5 * np.log(member('vitl_s46_tt')) + 0.5 * np.log(member('vitl_s47_tt')))
    if key.startswith('xl:'):                     # cross-lane member, read-only: xl:<lane>/<arm prefix>/<family>, arm dirs <prefix>_f<fold>_s46
        lane, pre, fam = key[3:].split('/')
        for f in range(5):
            r = X / lane / 'arms' / f'{pre}_f{f}_s46' / f'runs/fold{f}/{fam}'
            if not (r / 'complete.json').exists(): continue
            L = []
            for e in range(16, 21):
                z = np.load(r / f'predictions_epoch_{e:02d}.npz'); p = np.full(len(yc), np.nan); p[[pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]; L.append(np.log(p))
            m = np.exp(np.mean(L, 0)); out[fold == f] = m[fold == f]
        return out
    fam, seed, tag = MEMBERS[key]
    for f, r in roots(fam, seed, tag).items():
        L = [np.log(np.load(X / r / f'runs/fold{f}/{fam}/predictions_epoch_{e:02d}.npz')['pred'][:, 0]) for e in range(16, 21)]
        z = np.load(X / r / f'runs/fold{f}/{fam}/predictions_epoch_20.npz'); out[[pos[int(i)] for i in z['indices']]] = np.exp(np.mean(L, 0))
    return out
z = np.load(X / 'cd_density_channel_20261106/features.npz'); names = [str(n) for n in z['names']]; Dr = z['F'][idx].astype(float)[:, names.index('s48_top3_tri_med')]
ly = np.log(np.where(yc > 0, yc, np.nan))
def calibD(tr):
    out = np.full(len(yc), np.nan)
    for s in (False, True):
        q = tr & good & (yc > 0) & (SRC_NEW == s) & np.isfinite(Dr); co = np.polyfit(Dr[q], ly[q], 1); m = SRC_NEW == s; out[m] = np.exp(np.polyval(co, Dr[m]))
    return out
G = [(round(1 - a - c, 2), round(a, 2), round(c, 2)) for a in np.arange(0.2, 0.901, 0.05) for c in np.arange(0, 0.701, 0.05) if a + c <= 1.0001]
WD = (0, 0.05, 0.1, 0.15)
def read(keys):
    Ms = [member(k) for k in keys]; avail = [f for f in range(5) if all(np.isfinite(m[fold == f]).all() for m in Ms)]
    if len(avail) < 2: print(json.dumps(dict(set=keys, avail=avail, note='fewer than two folds')), flush=True); return
    F = np.exp(np.mean([np.log(m) for m in Ms], 0)); rows = np.isin(fold, avail); out = np.full(len(yc), np.nan); outD = out.copy(); ws = {}
    Lc = [np.log(P1), np.log(A2), np.log(np.where(np.isfinite(F), F, 1.))]
    Q = {wt: np.exp(sum(w * l for w, l in zip(wt, Lc))) for wt in G}
    for f in avail:
        tr = rows & (fold != f); D = calibD(fold != f); D = np.where(np.isfinite(D), D, 1.)
        best = None; bestD = None
        for wt, q in Q.items():
            prm = fit_stretch1(q, tr & good); e = cd(apply1(q, prm), tr)
            if best is None or e < best[0]: best = (e, wt, prm)
            for wd in WD:
                qq = np.where(SRC_NEW, np.exp((1 - wd) * np.log(q) + wd * np.log(D)), q); prm2 = fit_stretch1(qq, tr & good); e2 = cd(apply1(qq, prm2), tr)
                if bestD is None or e2 < bestD[0]: bestD = (e2, wt, wd, prm2)
        _, wt, prm = best; out[fold == f] = apply1(Q[wt][fold == f], prm)
        _, wt2, wd, prm2 = bestD; q = Q[wt2]; qq = np.where(SRC_NEW, np.exp((1 - wd) * np.log(q) + wd * np.log(D)), q); outD[fold == f] = apply1(qq[fold == f], prm2)
        ws[f] = dict(w=wt, wD=(wt2, wd))
    # two-component variant: B (arithmetic hybrid, ratio fixed) and F, one weight
    Bc = b['B'][:, 0]; outB = np.full(len(yc), np.nan); wsB = {}
    for f in avail:
        tr = rows & (fold != f); best = None
        for w in np.round(np.arange(0, 0.61, 0.05), 2):
            q = np.exp((1 - w) * np.log(Bc) + w * Lc[2]); prm = fit_stretch1(q, tr & good); e = cd(apply1(q, prm), tr)
            if best is None or e < best[0]: best = (e, w, prm)
        _, w, prm = best; wsB[f] = w; q = np.exp((1 - w) * np.log(Bc) + w * Lc[2]); outB[fold == f] = apply1(q[fold == f], prm)
    rep(f'B+F {"+".join(keys)} folds {avail}', outB, REF, rows=rows, note=json.dumps(wsB))
    r1 = rep(f'{"+".join(keys)} folds {avail}', out, REF, rows=rows, note=json.dumps(ws))
    r2 = rep(f'{"+".join(keys)} +D folds {avail}', outD, REF, rows=rows)
    for k, m in zip(keys, Ms): print(' ', k, 'alone raw CD', round(cd(m, rows), 3), flush=True)
    return r1, r2
def read_sep(keys, step=0.05):
    """Separate nested weight per member: log blend of P1, A2 and each member, weights on a grid (sum 1), stretch, fit on the other folds."""
    Ms = [member(k) for k in keys]; avail = [f for f in range(5) if all(np.isfinite(m[fold == f]).all() for m in Ms)]
    rows = np.isin(fold, avail); Lc = [np.log(P1), np.log(A2)] + [np.log(np.where(np.isfinite(m), m, 1.)) for m in Ms]
    import itertools
    ws_m = np.round(np.arange(0, 0.401, step), 3); grids = []
    for a in np.round(np.arange(0.25, 0.751, step), 3):
        for mw in itertools.product(ws_m, repeat=len(Ms)):
            r = 1 - a - sum(mw)
            if r >= 0.099: grids.append((round(r, 3), a) + tuple(mw))
    out = np.full(len(yc), np.nan); info = {}
    for f in avail:
        tr = rows & (fold != f); best = None
        for wt in grids:
            q = np.exp(sum(w * l for w, l in zip(wt, Lc))); prm = fit_stretch1(q, tr & good); e = cd(apply1(q, prm), tr)
            if best is None or e < best[0]: best = (e, wt, prm)
        _, wt, prm = best; info[f] = wt; q = np.exp(sum(w * l for w, l in zip(wt, Lc))); out[fold == f] = apply1(q[fold == f], prm)
    return rep(f'SEP {"+".join(keys)} folds {avail}', out, REF, rows=rows, note=json.dumps({str(k): v for k, v in info.items()}))
if __name__ == '__main__':
    for s in sys.argv[1:]:
        if s.startswith('sep:'): read_sep(s[4:].split(','))
        else: read(s.split(','))
