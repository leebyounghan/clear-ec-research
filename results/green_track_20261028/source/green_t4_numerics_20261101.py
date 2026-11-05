"""T4-numerics re-extraction of the hex6 OOF features (parent request 2026-11-01).

The submission-2 container runs each detector encoder under fp16 autocast with SDPA restricted to
[EFFICIENT_ATTENTION, MATH] (no flash attention, no bf16).  This re-extracts the fold-clean OOF outputs of the
ACTIVE families under exactly that numerics (encoder forward only; heads in float32 and peak finding unchanged),
then re-runs the LOFO hex6 blend on base B and compares with the bf16 features.

Families with fold-clean checkpoints (7 of the 10 active): s8_s46, sg2_s46, sg2_s47, sg2_s48, sg2_s49, sg2_s50,
sg2pre_s46.  The three stride-8 seed/multi-layer families (s8_s47, s8_s48, s8multi_s47) have OOF maps but no saved
fold-clean checkpoints; in the 10-family analysis their bf16 features are kept and flagged.

extract --family F --fold f [--dtype fp16|bf16]    analyze
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'p2_voronoi_features'))
import green_point_features_20261027 as G
import green_selftrain_20261028 as ST
import green_sg2_specialist_20261028 as SG
import cd_estimators_20261028 as P
import voronoi_features as V

ROOT = Path('experiments/green_track_20261028')
OUT = ROOT/'t4_numerics'
BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
FAMILIES = {
    's8_s46': dict(kind='s8', ckpt=lambda f: ST.TEACHERS[f]),
    'sg2_s46': dict(kind='sg2', ckpt=lambda f: ROOT/'sg2_specialist'/f'fold{f}'/'specialist_sg2.pt'),
    'sg2_s47': dict(kind='sg2', ckpt=lambda f: ROOT/'sg2_specialist_s47'/f'fold{f}'/'specialist_sg2.pt'),
    'sg2_s48': dict(kind='sg2', ckpt=lambda f: ROOT/'sg2_specialist_s48'/f'fold{f}'/'specialist_sg2.pt'),
    'sg2_s49': dict(kind='sg2', ckpt=lambda f: ROOT/'sg2_specialist_s49'/f'fold{f}'/'specialist_sg2.pt'),
    'sg2_s50': dict(kind='sg2', ckpt=lambda f: ROOT/'sg2_specialist_s50'/f'fold{f}'/'specialist_sg2.pt'),
    'sg2pre_s46': dict(kind='sg2', ckpt=lambda f: ROOT/'external_srep'/f'fold{f}'/'sg2_pre.pt'),
}
BF16_SOURCES = {   # where the original bf16 OOF outputs live (for the comparison)
    's8_s46': ('maps', lambda f: P.MAPS/f'fold{f}_specialist_s46.npz'),
    'sg2_s46': ('peaks', lambda f: ROOT/'sg2_specialist'/f'fold{f}'/f'fold{f}_sg2_peaks.npz'),
    'sg2_s47': ('peaks', lambda f: ROOT/'sg2_specialist_s47'/f'fold{f}'/f'fold{f}_sg2_peaks.npz'),
    'sg2_s48': ('peaks', lambda f: ROOT/'sg2_specialist_s48'/f'fold{f}'/f'fold{f}_sg2_peaks.npz'),
    'sg2_s49': ('peaks', lambda f: ROOT/'sg2_specialist_s49'/f'fold{f}'/f'fold{f}_sg2_peaks.npz'),
    'sg2_s50': ('peaks', lambda f: ROOT/'sg2_specialist_s50'/f'fold{f}'/f'fold{f}_sg2_peaks.npz'),
    'sg2pre_s46': ('peaks', lambda f: ROOT/'external_srep'/'preonly'/f'fold{f}'/f'fold{f}_sg2_peaks.npz'),
    's8_s47': ('maps', lambda f: ROOT/'s8_specialist_s47'/f'fold{f}'/f'fold{f}_specialist_s47.npz'),
    's8_s48': ('maps', lambda f: ROOT/'s8_specialist_s48'/f'fold{f}'/f'fold{f}_specialist_s48.npz'),
    's8multi_s47': ('maps', lambda f: ROOT/'s8multi_specialist_s47'/f'fold{f}'/f'fold{f}_multi_s47.npz'),
}
ACTIVE_ORDER = ['s8_s46', 's8_s47', 's8_s48', 'sg2_s46', 'sg2_s47', 'sg2_s48', 'sg2_s49', 'sg2_s50', 's8multi_s47', 'sg2pre_s46']


def log(msg, **kw):
    print(json.dumps(dict(utc=G.now(), msg=msg, **kw)), flush=True)


def cmd_extract(a):
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    fam = FAMILIES[a.family]
    out = OUT/a.dtype/a.family/f'fold{a.fold}'
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if a.dtype == 'fp16' else torch.bfloat16
    ann, binding, ids, y, tr, va = ST.green(a.fold)
    ck = torch.load(fam['ckpt'](a.fold), map_location='cpu', weights_only=False)
    enc = G.load_encoder(ST.ENCODERS[a.fold]); enc.load_state_dict(ck['encoder'], strict=True); enc.eval().requires_grad_(False)
    if fam['kind'] == 's8':
        head = G.build_head(768, 'linear'); head.load_state_dict(ck['head']); head.eval()
    else:
        head = SG.Head().cuda(); head.load_state_dict(ck['head']); head.eval()
    support = SG.frame_support()
    dl = DataLoader(G.b.Frames(va, y, len(ids)), batch_size=4, shuffle=False, num_workers=3, pin_memory=True)
    probs, peaks, offsets, seen, t0 = [], [], [0], [], time.time()
    for i, (x, _, ix) in enumerate(dl):
        with torch.no_grad(), sdpa_kernel(BACKENDS), torch.autocast('cuda', dtype=dtype):
            tok = enc.forward_features(G.b.inputs(x, 'cpdino_vitb'))['x_norm_patchtokens']
        with torch.no_grad():
            if fam['kind'] == 's8':
                p = head(tok.float()).squeeze(-1).sigmoid().reshape(len(x), *G.GRID).cpu().numpy()
                probs.append(p.astype(np.float16))
            else:
                p = head(tok).sigmoid().float().cpu().numpy()
                for m in p:
                    pk = SG.peaks2(m, support, SG.STORE_THRESHOLD)
                    peaks.append(pk.astype(np.float32)); offsets.append(offsets[-1]+len(pk))
        seen.append(ix.numpy())
        if i % 100 == 0:
            log('extract', family=a.family, fold=a.fold, dtype=a.dtype, batch=i, of=len(dl), seconds=round(time.time()-t0, 1))
    seen = np.concatenate(seen)
    np.testing.assert_array_equal(seen, va)
    train_green = set(binding['train_ids'])
    flag = np.array([n in train_green for n in ids[va]])
    if fam['kind'] == 's8':
        np.savez_compressed(out/f'fold{a.fold}_maps.npz', indices=va, ids=ids[va], probability=np.concatenate(probs), point_labels_seen=flag)
    else:
        np.savez_compressed(out/f'fold{a.fold}_sg2_peaks.npz', indices=va, ids=ids[va], peaks=np.concatenate(peaks), offsets=np.array(offsets), point_labels_seen=flag,
                            store_threshold=SG.STORE_THRESHOLD, nms_radius=SG.NMS2)
    (out/'extract.json').write_text(json.dumps(dict(created_utc=G.now(), family=a.family, fold=a.fold, dtype=a.dtype, backends='EFFICIENT_ATTENTION,MATH', n=int(len(va)),
                                                   checkpoint=str(fam['ckpt'](a.fold)), seconds=round(time.time()-t0, 1), seconds_per_image=round((time.time()-t0)/len(va), 3)), indent=1))
    log('done', family=a.family, fold=a.fold, dtype=a.dtype, seconds=round(time.time()-t0, 1))


def hex6_from(kind, path, n):
    H = np.full(n, np.nan)
    with np.load(path) as z:
        if kind == 'maps':
            for r, m in zip(z['indices'], z['probability']):
                H[r] = V.voronoi_feats(V.peaks_sg(m))['hex6_all']
        else:
            rows, pk, off = z['indices'], z['peaks'], z['offsets']
            for i, r in enumerate(rows):
                q = pk[off[i]:off[i+1]]
                H[r] = V.voronoi_feats(q[q[:, 2] >= 0.35, :2].astype(np.float64))['hex6_all']
    return H


def cmd_analyze(a):
    y, tta4, baseA, fold, seen, keys, F, aa = P.load()
    ok = ~seen
    n = len(y)
    with np.load(P.E/'recipeA_analysis_20261028/hybrid_oof.npz', allow_pickle=True) as h:
        order = np.argsort(h['indices']); baseB = h['B_hybrid_w0p5'][order]
    bf, fp, corr, missing = {}, {}, {}, []
    for fam in ACTIVE_ORDER:
        kind, src = BF16_SOURCES[fam]
        H = np.full(n, np.nan)
        for f in range(5):
            H_f = hex6_from(kind, src(f), n); m = np.isfinite(H_f); H[m] = H_f[m]
        bf[fam] = H
        if fam in FAMILIES:
            H2 = np.full(n, np.nan); present = True
            for f in range(5):
                p = OUT/'fp16'/fam/f'fold{f}'/(f'fold{f}_maps.npz' if FAMILIES[fam]['kind'] == 's8' else f'fold{f}_sg2_peaks.npz')
                if not p.exists():
                    present = False; break
                H_f = hex6_from('maps' if FAMILIES[fam]['kind'] == 's8' else 'peaks', p, n); m = np.isfinite(H_f); H2[m] = H_f[m]
            if present:
                fp[fam] = H2
                mm = ok & np.isfinite(H) & np.isfinite(H2)
                corr[fam] = dict(r_log=float(np.corrcoef(np.log(H[mm]), np.log(H2[mm]))[0, 1]), mean_abs_log_diff=float(np.abs(np.log(H[mm])-np.log(H2[mm])).mean()),
                                 frac_rows_changed=float((np.abs(H[mm]-H2[mm]) > 1e-6).mean()), n=int(mm.sum()))
            else:
                missing.append(fam+' (fp16 extraction incomplete)')
        else:
            missing.append(fam+' (no fold-clean checkpoint; bf16 feature kept)')
        print('family', fam, 'bf16 rows', int(np.isfinite(H).sum()), 'fp16', 'yes' if fam in fp else 'no', corr.get(fam, ''), flush=True)

    def mean_log(fams, use_fp):
        A = np.log(np.stack([fp[k] if (use_fp and k in fp) else bf[k] for k in fams]))
        with np.errstate(all='ignore'):
            return np.nanmean(A, 0)
    sets = {'ten_bf16': (ACTIVE_ORDER, False), 'ten_fp16_where_available': (ACTIVE_ORDER, True), 'seven_bf16': (list(fp), False), 'seven_fp16': (list(fp), True)}
    result = dict(created_utc=P.now(), missing_or_flagged=missing, per_family_bf16_vs_fp16=corr, blends={})
    feats = {}
    for name, (fams, use_fp) in sets.items():
        feat = mean_log(fams, use_fp); feats[name] = feat
        r = P.lofo_blend(y, baseB, fold, ok, np.exp(feat), t=2); o = r['overall']
        result['blends'][name] = dict(families=fams, delta_hex_pp=o['delta_cd_pp'], delta_mean_pp=o['delta_cd_pp']/3, folds_improving=o['folds_improving'],
                                      per_fold_mean_pp=[r[str(h)]['delta_cd_pp']/3 for h in range(5)], w=[r[str(h)]['w'] for h in range(5)], est_hex_mape=o['est_cd_mape'], err_corr=o['err_corr_mean'])
        print(name, 'B dHEX %.4f dMEAN %.4f impr %d' % (o['delta_cd_pp'], o['delta_cd_pp']/3, o['folds_improving']), [round(r[str(h)]['delta_cd_pp']/3, 4) for h in range(5)], flush=True)
    m = ok & np.isfinite(feats['ten_bf16']) & np.isfinite(feats['ten_fp16_where_available'])
    result['feature_corr_ten'] = dict(r=float(np.corrcoef(feats['ten_bf16'][m], feats['ten_fp16_where_available'][m])[0, 1]), mean_abs_diff=float(np.abs(feats['ten_bf16'][m]-feats['ten_fp16_where_available'][m]).mean()), max_abs_diff=float(np.abs(feats['ten_bf16'][m]-feats['ten_fp16_where_available'][m]).max()))
    m7 = ok & np.isfinite(feats['seven_bf16']) & np.isfinite(feats['seven_fp16'])
    result['feature_corr_seven'] = dict(r=float(np.corrcoef(feats['seven_bf16'][m7], feats['seven_fp16'][m7])[0, 1]), mean_abs_diff=float(np.abs(feats['seven_bf16'][m7]-feats['seven_fp16'][m7]).mean()))
    # deployed a/b/w applied unchanged (full fit; in-sample) to bf16 and fp16 ten-family features
    params = json.loads((ROOT/'deploy'/'hex6_blend_params.json').read_text())
    for name in ('ten_bf16', 'ten_fp16_where_available'):
        feat = feats[name]; have = np.isfinite(feat) & ok & (y[:, 2] > 0)
        pred = baseB.copy()
        pred[have, 2] = np.exp((1-params['w'])*np.log(baseB[have, 2])+params['w']*(params['a']+params['b']*feat[have]))
        sc, sb = P.official(pred[ok], y[ok]), P.official(baseB[ok], y[ok])
        per = [P.official(pred[ok & (fold == h)], y[ok & (fold == h)])['mean']-P.official(baseB[ok & (fold == h)], y[ok & (fold == h)])['mean'] for h in range(5)]
        result['blends'][name]['deployed_params_applied'] = dict(delta_mean_pp=sc['mean']-sb['mean'], delta_hex_pp=sc['HEX']-sb['HEX'], per_fold_mean_pp=per)
        print(name, 'deployed a/b/w applied: dMEAN %.4f dHEX %.4f' % (sc['mean']-sb['mean'], sc['HEX']-sb['HEX']), [round(v, 4) for v in per], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'result.json').write_text(json.dumps(result, indent=1))
    print('feature corr ten', result['feature_corr_ten'], 'seven', result['feature_corr_seven'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['extract', 'analyze'])
    ap.add_argument('--family', choices=list(FAMILIES))
    ap.add_argument('--fold', type=int)
    ap.add_argument('--dtype', choices=['fp16', 'bf16'], default='fp16')
    a = ap.parse_args()
    {'extract': cmd_extract, 'analyze': cmd_analyze}[a.command](a)
