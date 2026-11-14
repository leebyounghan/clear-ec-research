#!/usr/bin/env python3
"""CD lane freeze candidate: combine member CD predictions into the final CD (2026-11-10). Pure numpy, no torch.
Per image inputs (all in cells/mm2, CD column only):
  P1  : hybrid B's Phase-1 half (historical five crop members, calibrated), as in submission_hybrid_20261030
  A   : recipe A half, per seed s46/s47: exp(0.5*log(TTA4 at epoch 20) + 0.5*mean_e log(single view, epochs 8-19)); seeds averaged
        arithmetically; times the recipe A scalar
  D   : DINOv3-B/16 s46, four-flip TTA over epochs 16-20 (exp of mean log)
  V   : ViT-L/16 s46 raw track, exp(0.5*mean log(single view, epochs 8-11) + 0.5*log(four-flip TTA at epoch 12))
  mask_cd, mask_n_int : Cellpose-SAM cpsam_v2 zero-shot masks (flow 0.8, cellprob -2, CLAHE) -> seg_run.feats(): 'cd_mm2' and 'n_int'
combine(): q = exp(sum w_k log m_k) over (P1, A, D, V); if n_int >= 50: log q += wm * (a + b*log mask_cd - log q); then the stretch
c * exp(g * (log q - mu) + mu).  usage: cd_combine_20261110.py check [OOF.npz PARAMS.json]  -> recomputes the nested five-fold OOF CD."""
import sys, json
import numpy as np
def combine(P1, A, D, V, mask_cd, mask_n_int, prm):
    w = prm['weights']; lq = w['P1'] * np.log(P1) + w['A'] * np.log(A) + w['D'] * np.log(D) + w['V'] * np.log(V)
    have = np.isfinite(mask_cd) & (mask_cd > 0) & np.isfinite(mask_n_int) & (mask_n_int >= prm['mask_min_cells'])
    lm = np.log(np.where(have, mask_cd, 1.0)); mc = prm['mask_a'] + prm['mask_b'] * lm
    lq = np.where(have, lq + prm['mask_w'] * (mc - lq), lq)
    g, c, mu = prm['stretch_g'], prm['stretch_c'], prm['stretch_mu']
    return c * np.exp(g * (lq - mu) + mu)
def check(npz, pj):
    z = np.load(npz); P = json.load(open(pj)); y = z['y_cd']; fold = z['fold']; out = np.full(len(y), np.nan)
    for f in range(5):
        m = fold == f
        out[m] = combine(z['P1'][m], z['A'][m], z['D'][m], z['V'][m], z['mask_cd'][m], z['mask_n_int'][m], P['per_fold'][str(f)])
    ok = y > 0; v = float(np.mean(np.abs(out[ok] - y[ok]) / y[ok]) * 100)
    print(json.dumps(dict(cd_oof_recomputed=round(v, 4), cd_oof_stored=round(float(np.mean(np.abs(z['oof_cd'][ok] - y[ok]) / y[ok]) * 100), 4),
                          max_rel_diff_vs_stored=float(np.nanmax(np.abs(out / z['oof_cd'] - 1))), baseline_best_pipeline=P.get('baseline_cd'))))
if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'check':
        L = '/workspace/clear-ec-research/experiments/lane_cd_20261108/reads/'
        check(sys.argv[2] if len(sys.argv) > 2 else L + 'freeze_cd_oof_20261109.npz', sys.argv[3] if len(sys.argv) > 3 else L + 'freeze_cd_params_20261109.json')
