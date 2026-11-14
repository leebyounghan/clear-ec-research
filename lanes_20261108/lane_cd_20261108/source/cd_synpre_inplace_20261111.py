"""D-SYN-002 / D-SYN-003 CD read (coordinator 11-11): synthetic-pretrained recipe A s46 and/or ViT-L s46 IN PLACE OF the plain s46 run inside
their members, strict cross-fit over the folds where the synthetic runs exist (folds 0, 4), incumbent recomputed in the same run.
- A member (freeze recipe): per seed exp(0.5 log TTA4@final + 0.5 mean log single@e8..19), seeds 46/47 averaged, times a scalar fit on the training
  folds (strict). Synthetic s46 replaces plain s46 in that average, same form (if its run keeps epochs 8-19 singles; else stated in the output).
- V member: per seed exp(0.5 mean log single@e8..11 + 0.5 log TTA4@final), geometric mean of s46/s47; synthetic s46 replaces plain s46.
Strict: P1 / A scalars, weights, mask calibration and stretch from the other available fold(s). Kill rule (D-SYN-002): no strict gain on folds 0+4.
usage: SYN_A='<root with {f}>' SYN_A_FAM=cpdino_b8 SYN_V='<root with {f}>' SYN_V_FAM=dinov3_l16t cd_synpre_inplace_20261111.py
 (a root is the run directory that holds runs/fold{f}/<family>/ or <track>/runs/fold{f}/<family>/; set SYN_*_TRACK=raw if tracks exist)"""
import os, itertools, json, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
import cd_stack_20261109 as K
from cd_common import *


def vec(p):
    z = np.load(p); v = np.full(len(yc), np.nan); v[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]; return v


def rundir(tmpl, fam, track, f):
    root = X / tmpl.format(f=f); return (root / track if track else root) / f'runs/fold{f}' / fam


def syn_member(kind):
    tmpl = os.environ.get(f'SYN_{kind}'); fam = os.environ.get(f'SYN_{kind}_FAM'); track = os.environ.get(f'SYN_{kind}_TRACK', '')
    if not tmpl: return None, 'not given'
    out = np.full(len(yc), np.nan); notes = {}
    for f in range(5):
        d = rundir(tmpl, fam, track, f)
        fin = d / 'predictions_final_tta.npz' if (d / 'predictions_final_tta.npz').exists() else d / 'predictions_tta4.npz'
        if not fin.exists(): continue
        z = np.load(fin); key = 'pred' if 'pred' in z.files else 'pred_tta4'; p = np.full(len(yc), np.nan); p[[R.pos[int(i)] for i in z['indices']]] = z[key][:, 0]
        rng = range(8, 20) if kind == 'A' else range(8, 12)
        singles = [d / f'predictions_epoch_{e:02d}.npz' for e in rng]; have = [s for s in singles if s.exists()]
        m = 0.5 * np.mean([np.log(vec(s)) for s in have], 0) + 0.5 * np.log(p) if have else np.log(p)
        out[fold == f] = np.exp(m[fold == f]); notes[f] = f'{len(have)} single epochs + {fin.name}'
    return out, notes


Asyn, nA = syn_member('A'); Vsyn, nV = syn_member('V')
P1raw = R.P1 / 0.9970446110788349
A46 = np.exp(0.5 * np.log(R.seeds[46][:, 0]) + 0.5 * np.log(R._rec(46))); A47 = np.exp(0.5 * np.log(R.seeds[47][:, 0]) + 0.5 * np.log(R._rec(47)))
D = np.exp(0.5 * np.log(R.member('dinov3_s46_tta')) + 0.5 * np.log(R.member('dinov3_s47_tta')))
V46 = R.member('vitl_s46_tt'); V47 = R.member('vitl_s47_tt')
avail = [set(np.unique(fold[np.isfinite(m)])) for m in (Asyn, Vsyn) if m is not None]
AV = sorted(set.intersection(*avail)) if avail else []
AV = [f for f in AV if all(np.isfinite(m[fold == f]).all() for m in (Asyn, Vsyn) if m is not None and f in np.unique(fold[np.isfinite(m)]))]
rows = np.isin(fold, AV)
print('folds', AV, '| synthetic A:', nA, '| synthetic V:', nV, flush=True)
for nm, m in [('plain A s46', A46), ('synthetic A s46', Asyn), ('plain ViT-L s46', V46), ('synthetic ViT-L s46', Vsyn)]:
    if m is not None and AV: print(f'  alone CD {nm}: {cd(m, rows):.3f}', flush=True)


def wmed_scalar(p, tr):
    k = tr & (yc > 0); return float(C.wmed(yc[k] / p[k], p[k] / yc[k]))


def stack(Araw, Vm, name):
    out = np.full(len(yc), np.nan); info = {}
    for f in AV:
        tr = rows & (fold != f)
        Lc = [np.log(P1raw * wmed_scalar(P1raw, tr)), np.log(Araw * wmed_scalar(Araw, tr)), np.log(D), np.log(Vm)]; grids = []
        for a in np.round(np.arange(0.25, 0.751, 0.05), 3):
            for mw in itertools.product(np.round(np.arange(0, 0.401, 0.05), 3), repeat=2):
                r = 1 - a - sum(mw)
                if r >= 0.099: grids.append((round(r, 3), a) + tuple(mw))
        best = None
        for wt in grids:
            q = np.exp(sum(w * l for w, l in zip(wt, Lc))); e = cd(apply1(q, fit_stretch1(q, tr & good)), tr)
            if best is None or e < best[0]: best = (e, wt, q)
        q = best[2]; trm = tr & K.have & (yc > 0) & good; co = np.polyfit(K.lm[trm], K.ly[trm], 1); Mc = np.polyval(co, K.lm); b2 = None
        for wmk in (0, 0.05, 0.1, 0.15, 0.2):
            qq = np.where(K.have, np.exp((1 - wmk) * np.log(q) + wmk * Mc), q); p2 = fit_stretch1(qq, tr & good); e = cd(apply1(qq, p2), tr)
            if b2 is None or e < b2[0]: b2 = (e, wmk, qq, p2)
        out[fold == f] = apply1(b2[2][fold == f], b2[3]); info[f] = (best[1], b2[1])
    rep(name, out, REF, rows=rows, note='weights (P1, A, D, V, mask) ' + str(info)); return out


if len(AV) >= 2:
    Vinc = np.sqrt(V46 * V47); Ainc = 0.5 * (A46 + A47)
    inc = stack(Ainc, Vinc, f'incumbent v3 (strict) on folds {AV}')
    if Asyn is not None:
        a = stack(0.5 * (Asyn + A47), Vinc, 'synthetic recipe A s46 in place of plain s46'); rep('synthetic A vs incumbent', a, inc, rows=rows)
    if Vsyn is not None:
        v = stack(Ainc, np.sqrt(Vsyn * V47), 'synthetic ViT-L s46 in place of plain s46'); rep('synthetic ViT-L vs incumbent', v, inc, rows=rows)
    if Asyn is not None and Vsyn is not None:
        b = stack(0.5 * (Asyn + A47), np.sqrt(Vsyn * V47), 'both synthetic'); rep('both synthetic vs incumbent', b, inc, rows=rows)
else:
    print('fewer than two folds with synthetic runs; nothing read', flush=True)
