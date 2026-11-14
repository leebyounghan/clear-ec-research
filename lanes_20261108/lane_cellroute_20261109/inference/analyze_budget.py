"""Analysis of test_budget.py outputs (cell-route lane, 2026-11-11). usage: analyze_budget.py -> docker/budget_analysis.json + printout
(1) runtime per stage / member group from out/budget_fp32.json: first frame (one-time builds + weight reads) and steady state (median of frames 2..N),
    member stages split into weight load (state dict -> GPU) and forward;
(2) storage variants (out/budget_fp16.json, out/budget_bf16mix.json) against fp32: relative change per column and per scalar part;
(3) run-to-run check: fp32 CD / CV against the earlier deploy run (out/gpu_deploy_20.json stores |deploy/OOF - 1| on the same frames; CD and CV do
    not depend on cps5, the only input that differs between the two runs)."""
import json, os, collections
import numpy as np
import sys
E = '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/'
SUF = sys.argv[1] if len(sys.argv) > 1 else ''     # '' = 11-11 first runs (old stacks); '_v2' = current stacks
ld = lambda v: json.load(open(E + f'out/budget_{v}{SUF}.json')) if os.path.exists(E + f'out/budget_{v}{SUF}.json') else None
b = ld('fp32'); out = {}
R = b['rows']; steady = R[1:]
keys = sorted({k for r in R for k in r['stages']})
def med(k, rows): return float(np.median([r['stages'].get(k, 0.0) for r in rows]))
st = {k: dict(first=R[0]['stages'].get(k, 0.0), steady=med(k, steady)) for k in keys}
out['stages'] = st
top = {k: v for k, v in st.items() if '/' not in k}
out['wall'] = dict(first=R[0]['wall'], steady_median=float(np.median([r['wall'] for r in steady])), steady_sum_top=sum(v['steady'] for v in top.values()),
                   gpu_peak_gb=max(r['gpu_peak_gb'] for r in R))
GROUP = collections.OrderedDict([('ViT-L s46+s47', ['V']), ('recipe A e8-19 (A_cd)', ['A_cd']), ('recipe A e20 (A, TTA4)', ['A']),
    ('P1 five members', ['p1']), ('DINOv3-B D (TTA4)', ['D']), ('DINOv3-B CV 3 seeds', ['DINO_cv']), ('order-DINOv3', ['orderdino']), ('cvw3', ['cvw3']), ('attention-pool DINOv3-B attnp5', ['attnp5']),
    ('CPSAM 512 cps5', ['cps5']), ('CPSAM segmentation + seg features', ['seg']), ('pre-only detectors (TTA4 x 3)', ['peaks']),
    ('13 other detectors (single view)', ['other_peaks']), ('hex6 corrector', ['hex6']), ('HEX features + HEX graph nets', ['hex_features']),
    ('CV parts (graphs, graph nets, sim, frozen heads + recipe A tokens)', ['cv_parts']), ('cell-route features', ['cellroute']), ('combine (CD / HEX / CV apply)', ['combine'])])
gt = {}
for g, ks in GROUP.items():
    gt[g] = dict(first=sum(st.get(k, {}).get('first', 0) for k in ks), steady=sum(st.get(k, {}).get('steady', 0) for k in ks),
                 load=sum(st.get(k + '/load', {}).get('steady', 0) for k in ks), fwd=sum(st.get(k + '/fwd', {}).get('steady', 0) for k in ks),
                 sub={kk: round(v['steady'], 2) for kk, v in st.items() for k in ks if kk.startswith(k + '/') and kk.count('/') == 1})
out['groups'] = gt; out['unassigned_top'] = {k: v for k, v in top.items() if not any(k in ks for ks in GROUP.values())}
out['reads'] = dict(read_s_total=b['summary']['read_s_total'], read_gb_total=b['summary']['read_gb_total'], init_stages=b['summary'].get('init_stages'))
uncov = {f: v for f, v in b['files'].items() if v['bytes'] and v['bytes'] > 1 << 20 and not v.get('cast') and not f.endswith('.py')}
out['files_not_through_cast_hooks'] = {f: dict(gb=round(v['bytes'] / 1e9, 3), stages=v['stages']) for f, v in uncov.items()}
base = {f: v['stages'] for f, v in b['files'].items() if 'hf_cache' in f or 'phase2_foundation_20261002/weights' in f}
out['base_weight_files_opened_by'] = base
print('wall first %.1f s, steady median %.1f s, sum of top stages %.1f s, gpu peak %.1f GB' % (out['wall']['first'], out['wall']['steady_median'], out['wall']['steady_sum_top'], out['wall']['gpu_peak_gb']))
for g, v in gt.items(): print('%-70s first %6.1f  steady %6.1f  (load %5.1f fwd %5.1f) %s' % (g, v['first'], v['steady'], v['load'], v['fwd'], v['sub'] if not v['load'] else ''))
print('unassigned', out['unassigned_top']); print('reads', out['reads']['read_s_total'], 's for', out['reads']['read_gb_total'], 'GB'); print('init', out['reads']['init_stages'])
print('not through cast hooks (>1 MB):', out['files_not_through_cast_hooks']); print('base weights opened by:', base)
# (3) run-to-run
X_ = '/workspace/clear-ec-research/experiments/'
if SUF == '':
    dep = json.load(open(E + 'out/gpu_deploy_20.json'))['rows']; zc = np.load(X_ + 'lane_cd_20261108/reads/freeze_cd_oof_20261109.npz'); REFS = (('CD', zc['oof_cd']),)
else:      # current stacks: the no-stub deploy run 20d used the same members, so all three columns must agree
    DEP, CRR = {'_v2': ('gpu_deploy_20d', 'corr_hex_on_seedaudit10'), '_v3': ('gpu_deploy_20f', 'corr_hex_on_joint_attnp')}[SUF]
    dep = json.load(open(E + f'out/{DEP}.json'))['rows']; zc = np.load(X_ + 'lane_cd_20261108/reads/freeze_cd_oof_20261111.npz')
    REFS = (('CD', zc['oof_cd']), ('HEX', np.load(E + f'out/{CRR}.npz')['P']))
zv = np.load(X_ + 'lane_cv_20261108/deploy/cv_inputs_oof.npz', allow_pickle=True)
pos = {int(d): i for i, d in enumerate(zc['idx'])}; rr = {}
for c, ref in REFS + (('CV', zv['cv_oof']),):
    a = np.array([abs(r[c] / ref[pos[r['frame']]] - 1) for r in R]); d0 = {x['frame']: x[c] for x in dep}; bb = np.array([d0[r['frame']] for r in R])
    rr[c] = dict(max_abs_diff_of_rel=float(np.max(np.abs(a - bb))), identical_frames=int(np.sum(np.abs(a - bb) < 1e-12)))
out['run_to_run_vs_gpu_deploy_20'] = rr; print('run-to-run (fp32 vs earlier deploy run, |rel to OOF| compared):', rr)
# (2) storage variants
for v in ('fp16', 'bf16mix', 'fp16-nodet', 'fp16-cdonly'):
    x = ld(v)
    if x is None: continue
    assert [r['frame'] for r in x['rows']] == [r['frame'] for r in R]
    col = {}
    for c in ('CD', 'CV', 'HEX'):
        d = np.array([abs(r2[c] / r1[c] - 1) for r1, r2 in zip(R, x['rows'])])
        col[c] = dict(median=float(np.median(d)), max=float(np.max(d)), p90=float(np.percentile(d, 90)), identical_frames=int(np.sum(d == 0)))
    parts = {}
    for q in R[0]['parts']:
        def vec(r):
            v_ = r['parts'].get(q)
            if isinstance(v_, (int, float)): return np.array([v_], float)
            if isinstance(v_, list): return np.array(v_, float)
            return None
        A = [vec(r) for r in R]; B = [vec(r) for r in x['rows']]
        if any(a_ is None for a_ in A) or any(b_ is None for b_ in B) or any(a_.shape != b_.shape for a_, b_ in zip(A, B)): continue
        A = np.concatenate(A); B = np.concatenate(B); den = np.maximum(np.abs(A), 1e-9)
        parts[q] = dict(max_rel=float(np.max(np.abs(B - A) / den)), median_rel=float(np.median(np.abs(B - A) / den)), max_abs=float(np.max(np.abs(B - A))))
    out[v] = dict(columns=col, parts=parts, cast_gb=x['summary']['cast_gb'], kept_float_gb=x['summary']['kept_float_gb'], wall_first=x['summary']['wall_first'], wall_steady=x['summary']['wall_median_rest'])
    print(v, 'cast %.1f GB kept %.2f GB' % (x['summary']['cast_gb'], x['summary']['kept_float_gb']), {c: {k: '%.2e' % w for k, w in d.items()} for c, d in col.items()})
    for q, d in sorted(parts.items(), key=lambda kv: -kv[1]['max_rel'])[:14]: print('   %-22s max_rel %.2e median %.2e max_abs %.3g' % (q, d['max_rel'], d['median_rel'], d['max_abs']))
json.dump(out, open(E + f'inference/docker/budget_analysis{SUF}.json', 'w'), indent=1)
