"""D-CD-TRAIN-SCREEN fold-0 read (CPU). Arms (bb_runner_20261108, DINOv3-B/16 fold 0 s46, same SSL init): plain control P, A llrd075, B wd005,
C appear05 (20 epochs; raw track), D swa6e12 (12 epochs; swa6 track). Member forms: sv = exp(mean log single view epochs 16-20) (D: swa6 single
view epochs 8-12, the last five) and tta = four-flip TTA of final.pt. Member-level CD/CV/HEX on fold-0 rows vs the control in the same form, paired
donor-bootstrap interval. Fold-0 stack read: tier-p CD form (P1, A e20 TTA, D, V final TTA, mask, stretch) with every constant fitted on fold 4
with the incumbent members, applied to fold 0 with D's s46 half replaced by the arm (tta form) vs by the control (tta form).
Added slot (fixed in advance): the arm joins D as a third equal member (geometric mean of plain s46, s47 and the arm). Report-only subgroups:
lowest third of fold-0 CD labels; least legible third by seg_loose interior-cell count n_int. Sweep arms (arms/sweep_*_f0_s46) are read too.
Writes reads/cd_screen_read_20261113.json. Only fold 0 is used to choose (L2)."""
import os, json, itertools, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
import cd_stack_20261109 as K
from cd_common import *
f0 = fold == 0; y3 = BP.y.astype(float)
ARMS = {'P_plain': ('screen_plain_f0_s46', 'raw', 20), 'A_llrd075': ('screen_llrd075_f0_s46', 'raw', 20), 'B_wd005': ('screen_wd005_f0_s46', 'raw', 20),
        'C_appear05': ('screen_appear05_f0_s46', 'raw', 20), 'D_swa6e12': ('screen_swa6e12_f0_s46', 'swa6', 12),
        'S1_ssl_photometric': ('screen_sslS1_f0_s46', 'raw', 20), 'S2_ssl_lamb005': ('screen_sslS2b_f0_s46', 'raw', 20)}   # S2 read from the re-queued fine-tune (sslS2b)
import glob as _g
for _r in sorted(_g.glob(str(LANE / 'arms/sweep_*_f0_s46'))):
    _n = _r.split('/')[-1]; ARMS['W_' + _n[6:-7]] = (_n, 'raw', 20)
_known = {v[0] for v in ARMS.values()}
for _r in sorted(_g.glob(str(LANE / 'arms/screen_*_f0_s46'))):   # other screen roots, e.g. the build-box arm screen_lr2e5llrd075 (RTX 4090, micro 8: not bit-comparable)
    _n = _r.split('/')[-1]
    if _n not in _known and 'ssl' not in _n: ARMS['X_' + _n[7:-7]] = (_n, 'raw', 20)
def vec3(p):
    z = np.load(p); v = np.full((len(yc), 3), np.nan); v[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, :3]; return v
def member(name):
    root, track, E = ARMS[name]; fd = 'fold301' if 'srcold' in root else ('fold302' if 'srcnew' in root else 'fold0'); d = LANE / 'arms' / root / track / 'runs' / fd / 'dinov3_b16'
    if not (d / 'predictions_final_tta.npz').exists(): return None
    sv = np.exp(np.mean([np.log(vec3(d / f'predictions_epoch_{e:02d}.npz')) for e in range(E - 4, E + 1)], 0)); o = dict(sv=sv, tta=vec3(d / 'predictions_final_tta.npz'))
    t16 = LANE / 'tta' / (root.split('_f0_')[0] + '_f0_tta4_e16_20.npz')
    o['tta1620'] = vec3(t16) if t16.exists() else o['tta']   # D (12 epochs, swa6) has no e16-20: its final TTA stands in
    return o
def m3(p, rows):
    return [float(np.mean(np.abs(p[rows & (y3[:, j] > 0), j] / y3[rows & (y3[:, j] > 0), j] - 1)) * 100) for j in range(3)]
def paired(a, b, j):
    rows = f0 & (y3[:, j] > 0); d = (np.abs(a[rows, j] / y3[rows, j] - 1) - np.abs(b[rows, j] / y3[rows, j] - 1)) * 100; g = donors[rows]
    u, inv = np.unique(g, return_inverse=True); s = np.bincount(inv, d); n = np.bincount(inv); rng = np.random.default_rng(0)
    bs = [s[k].sum() / n[k].sum() for k in (rng.integers(0, len(u), len(u)) for _ in range(2000))]
    return [round(float(d.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
M = {k: member(k) for k in ARMS}
import re as _re
_new = np.array([bool(_re.match(r'^\d{4}-\d{4}', str(i).strip())) for i in ids])
if M.get('W_srcold') is not None and M.get('W_srcnew') is not None:   # routed by ID format (an image classifier would route at test time; AUC 0.998 in the records)
    M['R_src_routed'] = {f_: np.where(_new[:, None], M['W_srcnew'][f_], M['W_srcold'][f_]) for f_ in ('sv', 'tta', 'tta1620')}
_t1 = np.load(LANE / 'reads/cd_tier_t1_20261112.npz')['oof_cd']; EX100 = (yc > 0) & (np.abs(_t1 / np.where(yc > 0, yc, 1) - 1) > 1.0)   # fixed once from the frozen tier-p CD OOF
out = {'available': [k for k, v in M.items() if v is not None], 'ex100_rows_total': int(EX100.sum()), 'ex100_rows_fold0': int((EX100 & f0).sum())}
old = LANE.parent / 'lane_gpu3_20261106/runs/dinov3_b16_f0_s46/runs/fold0/dinov3_b16'
old_sv = np.exp(np.mean([np.log(vec3(old / f'predictions_epoch_{e:02d}.npz')) for e in range(16, 21)], 0))
out['old_family_runner_plain_sv1620'] = [round(v, 4) for v in m3(old_sv, f0)]
ctrl = M.get('P_plain')
for k, v in M.items():
    if v is None: continue
    r = {form: dict(cd_cv_hex=[round(x, 4) for x in m3(v[form], f0)], cd_ex100=round(m3(v[form], f0 & ~EX100)[0], 4),
                    old_cd_cv_hex=[round(x, 4) for x in m3(v[form], f0 & ~_new)], new_cd_cv_hex=[round(x, 4) for x in m3(v[form], f0 & _new)]) for form in ('sv', 'tta', 'tta1620')}
    if ctrl is not None and k != 'P_plain':
        for form in ('sv', 'tta', 'tta1620'): r[form]['vs_control_paired_mean_ci'] = {kk: paired(v[form], ctrl[form], j) for j, kk in enumerate(('CD', 'CV', 'HEX'))}
    if ctrl is not None and k != 'P_plain':
        lab = yc.copy(); m0 = f0 & (lab > 0); t_lab = np.percentile(lab[m0], 100 / 3); nint = K.F[:, K.nm['n_int']]; t_leg = np.nanpercentile(np.where(f0, nint, np.nan), 100 / 3)
        sub = {}
        for nm_, mask_ in (('lowest_third_CD_label', f0 & (lab <= t_lab)), ('least_legible_third_seg_loose_n_int', f0 & (np.nan_to_num(nint, nan=-1) <= t_leg))):
            rows_ = mask_ & (lab > 0); dd = (np.abs(v['tta1620'][rows_, 0] / lab[rows_] - 1) - np.abs(ctrl['tta1620'][rows_, 0] / lab[rows_] - 1)) * 100
            sub[nm_] = dict(rows=int(rows_.sum()), cd_diff_vs_control=round(float(dd.mean()), 4))
        r['report_only_subgroups_tta1620'] = sub
    out[k] = r
# fold-0 stack read
SCALE_CD = 0.9970446110788349; P1raw = R.P1 / SCALE_CD; ok = yc > 0; Afin_raw = 0.5 * (R.seeds[46][:, 0] + R.seeds[47][:, 0])
D46 = R.member('dinov3_s46_tta'); D47 = R.member('dinov3_s47_tta'); Dm = np.exp(0.5 * np.log(D46) + 0.5 * np.log(D47))
def vit(sd):
    o = np.full(len(yc), np.nan)
    for f in range(5):
        r = (X / f'lane_backbone_20261108/runs/vitl16_avg_f{f}_s46' if (f in (0, 4) and sd == 's46') else LANE / f'arms/vitl16_f{f}_{sd}') / f'raw/runs/fold{f}/dinov3_l16t'
        z = np.load(r / 'predictions_final_tta.npz'); p = np.full(len(yc), np.nan); p[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]; o[fold == f] = p[fold == f]
    return o
Vf = np.exp(0.5 * np.log(vit('s46')) + 0.5 * np.log(vit('s47')))
def wsc(p, tr): k = tr & ok; return float(C.wmed(yc[k] / p[k], p[k] / yc[k]))
tr = (fold == 4) & ok; sP = wsc(P1raw, tr); sA = wsc(Afin_raw, tr)
Lc = [np.log(P1raw * sP), np.log(Afin_raw * sA), np.log(Dm), np.log(Vf)]; grids = []
for a in np.round(np.arange(0.25, 0.751, 0.05), 3):
    for mw in itertools.product(np.round(np.arange(0, 0.401, 0.05), 3), repeat=2):
        rr = 1 - a - sum(mw)
        if rr >= 0.099: grids.append((round(rr, 3), a) + tuple(mw))
best = None
for wt in grids:
    q = np.exp(sum(w * l for w, l in zip(wt, Lc))); e = cd(apply1(q, fit_stretch1(q, tr & good)), tr)
    if best is None or e < best[0]: best = (e, wt, q)
wt = best[1]; q = best[2]; trm = tr & K.have & good; co = np.polyfit(K.lm[trm], K.ly[trm], 1); Mc = np.polyval(co, K.lm); b2 = None
for wmk in (0, 0.05, 0.1, 0.15, 0.2):
    qq = np.where(K.have, np.exp((1 - wmk) * np.log(q) + wmk * Mc), q); p2 = fit_stretch1(qq, tr & good); e = cd(apply1(qq, p2), tr)
    if b2 is None or e < b2[0]: b2 = (e, wmk, p2)
def stack(D):
    qa = np.exp(wt[0] * np.log(P1raw * sP) + wt[1] * np.log(Afin_raw * sA) + wt[2] * np.log(D) + wt[3] * np.log(Vf))
    qa = np.where(K.have, np.exp((1 - b2[1]) * np.log(qa) + b2[1] * Mc), qa); return apply1(qa, b2[2])
st = {'params_fit_on_fold4': dict(weights=wt, mask_w=b2[1]), 'incumbent_fold0': round(cd(stack(Dm), f0), 4), 'incumbent_fold0_ex100': round(cd(stack(Dm), f0 & ~EX100), 4)}
for k, v in M.items():
    if v is None: continue
    Dx = Dm.copy(); Dx[f0] = np.exp(0.5 * np.log(v['tta1620'][f0, 0]) + 0.5 * np.log(D47[f0])); st[k] = round(cd(stack(Dx), f0), 4); st[k + '_ex100'] = round(cd(stack(Dx), f0 & ~EX100), 4)
    Da = Dm.copy(); Da[f0] = np.exp((np.log(v['tta1620'][f0, 0]) + np.log(D46[f0]) + np.log(D47[f0])) / 3); st[k + '_added_as_third_D_member'] = round(cd(stack(Da), f0), 4)
if ctrl is not None:
    for k in M:
        if k != 'P_plain' and k in st: st[k + '_minus_control'] = round(st[k] - st['P_plain'], 4); st[k + '_added_minus_control_added'] = round(st[k + '_added_as_third_D_member'] - st['P_plain_added_as_third_D_member'], 4)
out['fold0_stack_kept_params'] = st
print(json.dumps(out, indent=1)); json.dump(out, open(LANE / 'reads/cd_screen_read_20261113.json', 'w'), indent=1, default=str)
