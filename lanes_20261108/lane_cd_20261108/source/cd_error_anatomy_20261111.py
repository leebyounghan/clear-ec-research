"""D-CDANAT-001 (PI question via coordinator, 11-11): anatomy of the CD error of the frozen candidate D-CD-001, strict five-fold OOF (6.1743), CPU
only, nothing in the freeze changed. Part 1: where the error sits (source, label decile, interior cells, legibility, image quality, tail, label-side
oddities). Part 2: why P1 is harder (label range, image quality, within-eye label spread vs prediction spread). Part 4: bounds. Part 3 (levers) is a
separate script. Writes reads/cd_error_anatomy_20261111.json."""
import os, sys, re, json, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
import cd_stack_20261109 as K
from cd_common import *
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
E = X
z = np.load(E / 'lane_cd_20261108/reads/strict_nesting_20261111.npz'); assert (z['idx'] == BP.idx).all()
p = z['v3_strict']; y = BP.y[:, 0].astype(float); ok = y > 0; N = len(y)
ape = np.abs(p - y) / y * 100; lr = np.log(p / y)
idsS = np.array([str(i).strip() for i in BP.ids]); P2 = np.array([bool(re.match(r'^\d{4}-\d{4}', i)) for i in idsS]); P1 = ~P2
out = {'overall': round(float(ape.mean()), 4), 'n': int(N)}


def summ(mask):
    m = mask & ok
    return dict(n=int(m.sum()), mape=round(float(ape[m].mean()), 3), share_of_error=round(float(ape[m].sum() / ape.sum()), 3),
                median_log_ratio=round(float(np.median(lr[m])), 4), p90_ape=round(float(np.percentile(ape[m], 90)), 2))


def bins(v, edges, names=None):
    res = {}
    for i in range(len(edges) - 1):
        m = (v >= edges[i]) & (v < edges[i + 1]); key = names[i] if names else f'[{edges[i]:.3g},{edges[i + 1]:.3g})'
        if (m & ok).sum(): res[key] = summ(m)
    return res


def qbins(v, q=5, mask=None):
    mm = np.isfinite(v) if mask is None else (mask & np.isfinite(v)); e = np.nanpercentile(v[mm], np.linspace(0, 100, q + 1)); e[-1] += 1e-9
    r = bins(np.where(mm, v, np.nan), list(e)); return r


# ---- part 1
out['by_source'] = {'P1 (old-format ids)': summ(P1), 'P2 (YYYY-NNNN ids)': summ(P2)}
out['by_label_decile'] = qbins(y, 10)
out['by_label_decile_P1'] = qbins(y, 5, P1); out['by_label_decile_P2'] = qbins(y, 5, P2)
nint = K.F[:, K.nm['n_int']]; cov = K.F[:, K.nm['coverage']]
out['by_interior_cells'] = bins(nint, [-1, 0.5, 50, 100, 150, 250, 1e9], ['0', '1-49', '50-99', '100-149', '150-249', '>=250'])
out['by_interior_cells_P1'] = bins(np.where(P1, nint, np.nan), [-1, 0.5, 50, 150, 1e9], ['0', '1-49', '50-149', '>=150'])
out['by_interior_cells_P2'] = bins(np.where(P2, nint, np.nan), [-1, 0.5, 50, 150, 1e9], ['0', '1-49', '50-149', '>=150'])
out['by_mask_coverage'] = qbins(cov, 5)
pk = np.load(E / 'error_case_analysis_20261106/peak_traits.npz'); A_ = pk['A']; na = list(pk['names_a'])
conf = A_[:, na.index('conf_med')].astype(float); hif = A_[:, na.index('hi_frac')].astype(float); n50 = A_[:, na.index('n50')].astype(float)
out['by_detector_confidence'] = qbins(conf, 5); out['by_detector_hi_frac'] = qbins(hif, 5); out['by_detector_n50'] = qbins(n50, 5)
try:
    sys.path.insert(0, str(E / 'error_case_analysis_20261106/source')); sys.path.insert(0, str(E / 'lane_gpu0_20261106/source'))
    cwd = os.getcwd(); os.chdir('/workspace/clear-ec-research'); import common as EC; import calib2_20261106 as C2
    T, Rr, a_, c_, LP, freq = EC.load_traits(); os.chdir(cwd)
    sg = np.asarray(C2.sigma_floor(LP), float); fc = np.asarray(C2.focus2(LP, (0.07, 0.14)), float)
    out['by_noise_sigma_floor'] = qbins(sg, 5); out['by_focus2'] = qbins(fc, 5)
    out['image_quality_by_source'] = {s: dict(sigma_floor_med=round(float(np.nanmedian(sg[m])), 3), focus2_med=round(float(np.nanmedian(fc[m])), 4)) for s, m in (('P1', P1), ('P2', P2))}
except Exception as e:
    out['image_quality_error'] = repr(e)[:300]
out['legibility_by_source'] = {s: dict(conf_med=round(float(np.nanmedian(conf[m])), 3), hi_frac_med=round(float(np.nanmedian(hif[m])), 3), n_int_med=float(np.nanmedian(nint[m])),
                                       share_n_int_ge50=round(float(np.mean(nint[m] >= 50)), 3)) for s, m in (('P1', P1), ('P2', P2))}
# tail
order = np.argsort(-ape); tot = ape.sum(); tail = {}
fz = np.load(E / 'error_case_analysis_20261106/tables/forensics_rows.npz'); det = fz['det'][:, 0].astype(float)
Mc_all = np.full(N, np.nan); trm = K.have & ok & good; co = np.polyfit(K.lm[trm], K.ly[trm], 1); Mc_all[K.have] = np.exp(np.polyval(co, K.lm[K.have]))
impossible_ids = [i for i in idsS if re.match(r'^2020-06(4[4-9]|5[01])\b', i)]
out['detector_det_col0_range'] = [float(np.nanpercentile(det, q)) for q in (1, 50, 99)]
odd_mask = np.isfinite(Mc_all) & (np.abs(np.log(y / np.where(np.isfinite(Mc_all), Mc_all, y))) > np.log(2))   # label vs calibrated mask CD (n_int >= 50 rows) off by > 2x
hexzero = BP.y[:, 2] <= 0
for frac in (0.001, 0.01, 0.05):
    k = max(1, int(round(frac * N))); top = order[:k]; mtop = np.zeros(N, bool); mtop[top] = True
    tail[f'worst_{frac * 100:g}%'] = dict(rows=k, share_of_total_ape=round(float(ape[top].sum() / tot), 4), mape_if_these_were_at_median=round(float((tot - ape[top].sum() + k * np.median(ape)) / N), 4),
                                         P1_share=round(float(P1[top].mean()), 3), label_off_mask_or_detector_2x=int((odd_mask & mtop).sum()), hex_zero_rows=int((hexzero & mtop).sum()),
                                         kg_impossible_serials=int(np.isin(idsS[top], impossible_ids).sum()), median_label=float(np.median(y[top])), median_pred=float(np.median(p[top])),
                                         under_predicted=int((p[top] < y[top]).sum()))
out['tail'] = tail; out['label_off_2x_total'] = int(odd_mask.sum()); out['kg_impossible_rows_found'] = len(impossible_ids)
out['kg_impossible_mape_contribution'] = round(float(ape[np.isin(idsS, impossible_ids)].sum() / N), 4)
out['label_off_2x_mape_contribution'] = round(float(ape[odd_mask].sum() / N), 4)
# ---- part 2: why P1 is harder
out['label_range_by_source'] = {s: [round(float(v), 0) for v in np.percentile(y[m], [5, 25, 50, 75, 95])] for s, m in (('P1', P1), ('P2', P2))}
# P2 error at P1's label distribution (reweight P2 rows by label decile counts of P1)
ed = np.percentile(y, np.linspace(0, 100, 11)); ed[-1] += 1; dbin = np.digitize(y, ed) - 1
w = np.zeros(N)
for b in range(10):
    n1 = (P1 & (dbin == b)).sum(); n2 = (P2 & (dbin == b)).sum()
    if n2: w[P2 & (dbin == b)] = n1 / n2
out['P2_error_reweighted_to_P1_label_mix'] = round(float(np.sum(w * ape) / np.sum(w)), 3)
w1 = np.zeros(N)
for b in range(10):
    n1 = (P1 & (dbin == b)).sum(); n2 = (P2 & (dbin == b)).sum()
    if n1: w1[P1 & (dbin == b)] = n2 / n1
out['P1_error_reweighted_to_P2_label_mix'] = round(float(np.sum(w1 * ape) / np.sum(w1)), 3)
# within-eye spread: group = donor key + eye (OD/OS) where the id has it; else donor key
def gkey(i):   # P1 ids 'DDDD-NN' -> donor 'DDDD' (several frames per donor); P2 ids 'YYYY-NNNN-EYE..' -> 'YYYY-NNNN' + eye OD/OS
    m2 = re.match(r'^(\d{4}-\d{4})\D*(OD|OS)?', i.upper())
    if m2: return m2.group(1) + '|' + (m2.group(2) or '')
    return i.split('-')[0] + '|'
grp = np.array([gkey(i) for i in idsS])
ly = np.log(y); lp = np.log(p)
res = {}
for s, m in (('P1', P1), ('P2', P2)):
    g = grp[m]; u, inv, cnt = np.unique(g, return_inverse=True, return_counts=True); keep = cnt[inv] >= 2
    gi = inv[keep]; Y = ly[m][keep]; Pp = lp[m][keep]
    my = np.bincount(gi, Y) / np.bincount(gi); mp = np.bincount(gi, Pp) / np.bincount(gi)
    dy = Y - my[gi]; dp = Pp - mp[gi]
    res[s] = dict(groups=int((cnt >= 2).sum()), rows=int(keep.sum()), median_group_size=float(np.median(cnt[cnt >= 2])),
                  within_sd_label=round(float(np.sqrt(np.mean(dy ** 2))), 4), within_sd_pred=round(float(np.sqrt(np.mean(dp ** 2))), 4),
                  within_sd_residual=round(float(np.sqrt(np.mean((dy - dp) ** 2))), 4), within_corr=round(float(np.corrcoef(dy, dp)[0, 1]), 3),
                  between_sd_residual=round(float(np.std((my - mp)[np.bincount(gi) > 0])), 4))
out['within_eye'] = res
out['donor_key_examples'] = {s: list(np.unique(grp[m])[:3]) for s, m in (('P1', P1), ('P2', P2))}
# ---- part 4: bounds
cvlab = BP.y[:, 1].astype(float)          # CV label is a fraction (e.g. 0.30)
out['cv_label_median'] = round(float(np.median(cvlab[cvlab > 0])), 4)
floor150 = np.sqrt(2 / np.pi) * cvlab / np.sqrt(150) * 100
out['sampling_floor_150_cells'] = dict(mape_floor_all=round(float(floor150[ok].mean()), 3), P1=round(float(floor150[P1 & ok].mean()), 3), P2=round(float(floor150[P2 & ok].mean()), 3),
                                       note='E|rel. error of a 150-cell mean area| = sqrt(2/pi) x CV/sqrt(150), CV = the frame CV label; CD = 1e6/mean area so the same relative error')
out['if_P1_had_P2_error'] = dict(overall=round(float(ape[P2].mean()), 3), note='P1 rows at P2 mean APE; with P2 error at P1 label mix see P2_error_reweighted_to_P1_label_mix')
out['test_noise'] = dict(sd_1000_row_mean=0.318, our_projection=6.11, second_place_cd=5.975, v14_cd=6.049,
                         gap_to_second_in_sd=round((6.11 - 5.975) / 0.318, 2), gap_to_v14_in_sd=round((6.11 - 6.049) / 0.318, 2))
print(json.dumps(out, indent=1))
json.dump(out, open(E / 'lane_cd_20261108/reads/cd_error_anatomy_20261111.json', 'w'), indent=1)
