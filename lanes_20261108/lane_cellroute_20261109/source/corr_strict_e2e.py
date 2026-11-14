"""End-to-end strict HEX with the cell-route step (deploy lane, 2026-11-11; coordinator after F-DEPLOY-001): the cell-route step's strict procedure
(feature-version choice among the 15 versions + 'none' by inner 4-fold CV on the training folds, ridge alpha 100 / >= 50 cells fixed, weights and
scales fitted on the training folds) on top of the HEX lane's strict fold runs for D-HEX-004 (runs/strict/strict_f{f}.npz 'joint_attnp': the 39-column
joint-ridge procedure rerun with fold f's labels hidden; assembled as the lane does, chain.final = per-fold stretch fitted on the other folds -> 9.71977).
Two conventions for the base on the training rows inside outer fold f (the scored fold f always gets its strict value):
  (a) chain convention: run f's candidate stretched with run f's stretch on every row (no label of fold f anywhere; training rows in-sample to run f);
  (b) OOF convention: the assembled strict OOF on every row (each row from its own fold's run, like a deploy base; fold f's labels reach the training
      rows' base second-hand through runs g != f).
The kept version (no version choice) is reported on the same bases for reference.
usage: corr_strict_e2e.py [FOLDFILE_PATTERN KEY TAG]   (default runs/strict/strict_f{f}.npz joint_attnp, tag '' = D-HEX-004; tiers: runs/strict_ladder/lad_f{f}.npz t2 | t3x)"""
import sys, json
import numpy as np
R = '/workspace/clear-ec-research/experiments/'; L = R + 'lane_cellroute_20261109/'; LH = R + 'lane_hex_20261108/'
sys.path[:0] = [LH + 'source', R + 'meta_ceiling_20261103/source', L + 'source', R + 'lane_cv_20261108/source']
import chain as C
import best_pipeline_20261106 as BP
import ceiling_meta_cpu_20261103 as CM
idx, fold, good = BP.idx.astype(int), BP.fold, BP.good; y = BP.y[:, 2]; ok = y > 0
assert (np.asarray(C.idx).astype(int) == idx).all() and (C.fold == fold).all()
PAT, KEY, TAG = (sys.argv[1], sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ('runs/strict/strict_f{f}.npz', 'joint_attnp', '')
cand = {f: np.load(LH + PAT.format(f=f))[KEY] for f in range(5)}
base_oof = C.final(cand)[:, 2]
base_run = {}
for f in range(5):
    H = C.BP.H.copy(); H[:, 2] = cand[f]; prm = C.NS.fit_stretch(H, C.y, (C.fold != f) & C.good); base_run[f] = C.NS.apply_stretch(H, prm)[:, 2]   # exactly chain.final's stretch
    dmax = float(np.max(np.abs(base_run[f][fold == f] / base_oof[fold == f] - 1))); print('fold', f, 'run-f stretch on fold f vs assembled strict OOF max rel', dmax, flush=True); assert dmax < 1e-12
lg = lambda a: np.log(np.maximum(a, 1e-4))
z = np.load(L + 'feats/cellroute_feats.npz'); F1, N1 = z['F'][idx], list(z['names'])
z = np.load(L + 'feats/trim_feats.npz'); F3, N3 = z['F'][idx], list(z['names'])
z = np.load(L + 'feats/click_feats_P2_P5_m1_K8.npz'); F2, N2 = z['F'][idx], list(z['names'])
def five(F, N, p):
    g = lambda c: F[:, N.index(f'{p}_{c}')]; return [lg(g('cv_k0')), lg(g('cv_k2')), g('hex_k0'), g('hex_k2'), np.log1p(np.nan_to_num(g('n')))]
def sets(ss, regs):
    cols = [c for s in ss for rg in regs if f'{s}_{rg}_cv_k2' in N1 for c in five(F1, N1, f'{s}_{rg}')]
    return np.column_stack(cols), np.nan_to_num(F1[:, N1.index(f'{ss[0]}_full_n')]) >= 50
def blocks(bl):
    Xs, m = [], np.ones(len(y), bool)
    for b in bl:
        if b == 'P2raw': X = np.column_stack(five(F1, N1, 'P2_win') + five(F1, N1, 'P2_full')); n = np.nan_to_num(F1[:, N1.index('P2_full_n')])
        elif b in ('P2c', 'P5c'):
            g = lambda c: F2[:, N2.index(f'{b}_win_{c}')]; X = np.column_stack(five(F2, N2, f'{b}_win') + [g('s5_k2'), g('s7_k2')]); n = np.nan_to_num(g('n'))
        else: X = np.column_stack(five(F3, N3, b)); n = np.nan_to_num(F3[:, N3.index(f'{b}_n')])
        Xs.append(X); m &= np.isfinite(X).all(1) & (n >= 50)
    return np.column_stack(Xs), m
CAND = {'P2 win+full (kept)': sets(['P2'], ['win', 'full']), 'P0 win+full': sets(['P0'], ['win', 'full']), 'P1 win+full': sets(['P1'], ['win', 'full']),
        'P3 win+full': sets(['P3'], ['win', 'full']), 'P2 win+full+creg': sets(['P2'], ['win', 'full', 'creg']), 'P0-P3 win+full': sets(['P0', 'P1', 'P2', 'P3'], ['win', 'full']),
        'P2raw+P2c+P5c': blocks(['P2raw', 'P2c', 'P5c']), 'P2c': blocks(['P2c']), 'P5c': blocks(['P5c']), 'P2raw+P2c': blocks(['P2raw', 'P2c']), 'P2raw+P5c': blocks(['P2raw', 'P5c']),
        'P2 box raw': blocks(['P2_box_raw']), 'P2 p150 raw': blocks(['P2_p150_raw']), 'P2 box trim': blocks(['P2_box_trim']), 'P2raw+P2 box raw': blocks(['P2raw', 'P2_box_raw'])}
KEPT = 'P2 win+full (kept)'; NAMES = ['none'] + list(CAND)
def fitpred(name, base, trn, app):
    P = base.copy()
    if name == 'none': return P
    X, m = CAND[name]; X = np.c_[X, np.log(base)]; m = m & np.isfinite(X).all(1); X = np.where(m[:, None], X, 0.0)
    res = np.log(np.where(ok, y, 1.0)) - np.log(base); tr = trn & good & m & ok; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; A = (X[tr] - mu) / sd
    w = np.linalg.solve(A.T @ A + 100.0 * np.eye(A.shape[1]), A.T @ (res[tr] - res[tr].mean())); q = base * np.exp(((X - mu) / sd) @ w)
    c = CM.wmed(y[tr] / q[tr], q[tr] / y[tr]); c0 = CM.wmed(y[tr] / base[tr], base[tr] / y[tr]); s = app & m; P[s] = q[s] * c / c0; return P
ape = lambda p, r: float(np.mean(np.abs(p[r & ok] - y[r & ok]) / y[r & ok]) * 100)
import peak_select25_20261108 as PS
gate = np.isin(idx, [int(PS.idx[PS.pos[n]]) for n in PS.names]); allr = np.ones(len(y), bool); f04 = (fold == 0) | (fold == 4)
def run(basef):
    Ps, Pk, pick = base_oof.copy(), base_oof.copy(), {}
    for f in range(5):
        b = basef(f); trf = fold != f; sc = {}
        for nm in NAMES:
            Pin = b.copy()
            for g in range(5):
                if g != f: Pin[fold == g] = fitpred(nm, b, trf & (fold != g), fold == g)[fold == g]
            sc[nm] = ape(Pin, trf)
        pick[f] = min(sc, key=sc.get); Ps[fold == f] = fitpred(pick[f], b, trf, fold == f)[fold == f]; Pk[fold == f] = fitpred(KEPT, b, trf, fold == f)[fold == f]
    return Ps, Pk, pick
def summ(P):
    return dict(five=round(ape(P, allr), 5), f04=round(ape(P, f04), 5), d_five=round(ape(P, allr) - ape(base_oof, allr), 5), d_f04=round(ape(P, f04) - ape(base_oof, f04), 5),
                folds=[round(ape(P, fold == f) - ape(base_oof, fold == f), 4) for f in range(5)], d_five_no_gate=round(ape(P, ~gate) - ape(base_oof, ~gate), 5))
out = dict(hex_strict_incumbent=dict(five=round(ape(base_oof, allr), 5), f04=round(ape(base_oof, f04), 5), folds=[round(ape(base_oof, fold == f), 4) for f in range(5)]))
for tag, bf in (('a_chain_convention', lambda f: base_run[f]), ('b_oof_convention', lambda f: base_oof)):
    Ps, Pk, pick = run(bf); out[tag] = dict(strict=summ(Ps), kept_version=summ(Pk), picks=pick)
    np.savez_compressed(L + f'out/corr_hex_e2e_strict{TAG}_{tag}.npz', P=Ps, P_kept=Pk, idx=idx, fold=fold, base=base_oof)
out['fold_runs'] = dict(pattern=PAT, key=KEY)
print('SUMMARY', json.dumps(out), flush=True); json.dump(out, open(L + f'out/corr_hex_e2e_strict{TAG}.json', 'w'), indent=1)
