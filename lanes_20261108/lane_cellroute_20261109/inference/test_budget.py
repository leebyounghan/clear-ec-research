"""Size / runtime / half-precision storage budget (cell-route lane, 2026-11-11). Test harness on one GPU (CUDA_VISIBLE_DEVICES), <= 4 CPU threads.
usage: test_budget.py N VARIANT OUT
VARIANT is a storage simulation applied when weights are read (nothing is written to disk):
  fp32    as stored;
  fp16    every floating tensor with > 16 elements cast to float16 and back (metadata such as log_mean / log_std, 3 elements, stays fp32);
  bf16mix tensors with >= 2 dims (linear / conv / embedding weights) cast to bfloat16 and back; 1-D tensors (norm weights, biases) stay fp32.
  <fp16|bf16mix>-nodet   the cast only for files that are not thresholded into peaks / masks: detectors (all16, pre-only, hex6), detector
                         frozen heads and their recipe A e20 encoder, the CPSAM segmentation backbone (cpsam_v2, also the P1 backbone) and
                         the HEX / CV graph models stay fp32 (added 2026-11-11 after full fp16 failed the CV / HEX rule);
  <fp16|bf16mix>-cdonly  the cast only for the CD-only groups: ViT-L s46/s47 and recipe A epochs 8-19.
Covered loaders: torch.load (all .pt, cellpose cpsam_v2, P1, graph nets, detectors, hex6) and safetensors.torch.load_file; anything else is reported as
uncovered. Same 20 training frames as test_gpu_deploy.py. cps5 uses the fold-0 CPSAM 512 weights (same architecture; f99 not complete at launch) so all
three variants read identical files. Records per frame: CD / CV / HEX and every small part (storage comparison), per-stage wall time (cuda-synchronised,
nested stages as 'outer/inner'), peak GPU memory; per file: first-read seconds, bytes, which stage opened it, how many tensors / bytes were cast."""
import sys, os, json, time, functools, builtins, io, collections
import numpy as np, torch
torch.set_num_threads(4)
import re
N = int(sys.argv[1]); VAR = sys.argv[2]; OUT = sys.argv[3]
DT, _, SEL = VAR.partition('-'); assert DT in ('fp32', 'fp16', 'bf16mix') and SEL in ('', 'nodet', 'cdonly')
KEEP_NODET = ('/detectors/', '/frozen_heads', 'cpsam_v2', 'graph_models')


def selected(p):
    p = os.fspath(p)
    if SEL == 'nodet': return not (any(k in p for k in KEEP_NODET) or ('fulldata_recipeA_20261028' in p and p.endswith('epoch_20.pt')))
    if SEL == 'cdonly': return 'vitl16_f99' in p or re.search(r'fulldata_recipeA_20261028/.+/epoch_(0[89]|1\d)\.pt$', p) is not None
    return True
ROOT = '/workspace/clear-ec-research/'
STACK = []; T = collections.defaultdict(float); OPENS = collections.defaultdict(set); READ = {}; CAST = {}


def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()


def stage(): return '/'.join(STACK) if STACK else 'setup'


def timed(name, fn):
    @functools.wraps(fn)
    def w(*a, **k):
        sync(); STACK.append(name); t = time.perf_counter()
        try: return fn(*a, **k)
        finally:
            sync(); T[stage()] += time.perf_counter() - t; STACK.pop()
    return w


# ---- file-open attribution ----
def note(p):
    try:
        p = os.path.abspath(os.fspath(p))
        if p.startswith(ROOT) and os.path.isfile(p): OPENS[p].add(stage())
    except TypeError: pass
_open = builtins.open
def hopen(f, *a, **k): note(f); return _open(f, *a, **k)
builtins.open = hopen; io.open = hopen
_oso = os.open
def hosopen(f, *a, **k): note(f); return _oso(f, *a, **k)
os.open = hosopen


# ---- storage simulation ----
def cast(o, rec, on=True):
    if isinstance(o, torch.Tensor):
        if on and DT != 'fp32' and o.is_floating_point() and o.numel() > 16 and (DT == 'fp16' or o.dim() >= 2):
            rec[0] += 1; rec[1] += o.numel() * o.element_size()
            return o.to(torch.float16 if DT == 'fp16' else torch.bfloat16).to(o.dtype)
        rec[2] += o.numel() * o.element_size() if o.is_floating_point() else 0
        return o
    if isinstance(o, torch.nn.Module):
        for p in list(o.parameters()) + list(o.buffers()): p.data = cast(p.data, rec, on)
        return o
    if isinstance(o, dict):
        for k in list(o.keys()): o[k] = cast(o[k], rec, on)
        return o
    if isinstance(o, list): return [cast(v, rec, on) for v in o]
    if isinstance(o, tuple) and not hasattr(o, '_fields'): return tuple(cast(v, rec, on) for v in o)
    return o


_load = torch.load
@functools.lru_cache(maxsize=None)
def _cached(path, wo):
    t = time.perf_counter(); o = _load(path, map_location='cpu', weights_only=wo); READ[path] = time.perf_counter() - t
    rec = [0, 0, 0]; o = cast(o, rec, selected(path)); CAST[path] = rec; return o
def cached_load(f, *a, **k):
    if isinstance(f, (str, os.PathLike)):
        p = os.path.abspath(os.fspath(f)); note(p); return _cached(p, bool(k.get('weights_only', False)))
    rec = [0, 0, 0]; o = cast(_load(f, *a, **k), rec, SEL == ''); CAST['<file object>'] = rec; return o
torch.load = cached_load
try:
    import safetensors, safetensors.torch as st
    _lf = st.load_file
    def hlf(f, *a, **k):
        p = os.path.abspath(os.fspath(f)); note(p); t = time.perf_counter(); o = _lf(f, *a, **k); READ[p] = READ.get(p, 0) + time.perf_counter() - t
        rec = [0, 0, 0]; o = cast(o, rec, selected(p)); CAST[p] = rec; return o
    st.load_file = hlf
    _so = safetensors.safe_open
    def hso(f, *a, **k): note(f); OPENS[os.path.abspath(os.fspath(f))].add(stage() + ' [safe_open: not cast]'); return _so(f, *a, **k)
    safetensors.safe_open = hso
except ImportError: pass

sys.path.insert(0, ROOT + 'experiments/lane_cellroute_20261109/inference')
import recipe_infer as RI, members as M
E = 'experiments/'
argv = sys.argv; sys.argv = ['feat_fuse.py', '/dev/null', 'all16']
import hex_test_features as HT, gnn_hex_infer as GI
sys.argv = argv
import seg_run_feats as SR, cv_image_parts_20261110 as CVP

# ---- stage timers ----
for name, fn in (('A', 'recipeA_tta'), ('A_cd', 'recipeA_cd'), ('D', 'dino_cd'), ('DINO_cv', 'dino_cv'), ('orderdino', 'orderdino_cv'), ('V', 'vitl_cd'), ('tokens', 'recipeA_tokens')):
    setattr(M, fn, timed(name, getattr(M, fn)))
_fm = M.family_mean
def family_mean(x, ckpts, fam): return timed('cvw3' if fam == 'dinov3_b16' else 'cps5' if fam == 'cpsam_c512' else 'family_' + fam, _fm)(x, ckpts, fam)
M.family_mean = family_mean
M.attnpool_hex = timed('attnp5', M.attnpool_hex); M.Member.__init__ = timed('load', M.Member.__init__); M.Member.logpred = timed('fwd', M.Member.logpred)
for name in ('p1', 'seg', 'peaks', 'other_peaks', 'hex_features', 'hex6', 'cv_parts'): setattr(RI.Recipe, name, timed(name, getattr(RI.Recipe, name)))
SR.feats = timed('feats', SR.feats); RI.DK.seg_points = timed('seg_points', RI.DK.seg_points)
HT.build_arrays = timed('build_arrays', HT.build_arrays); GI.predict = timed('graphnets', GI.predict)
CVP.graphs_from_peaks = timed('graphs', CVP.graphs_from_peaks); CVP.GraphNets.out = timed('gnets', CVP.GraphNets.out); CVP.SimNets.embed = timed('sim', CVP.SimNets.embed)
CVP.FrozenHeads.logcv = timed('frozen', CVP.FrozenHeads.logcv); CVP.token_blocks = timed('token_blocks', CVP.token_blocks)
RI.DK.features = timed('cellroute', RI.DK.features)
for mod, fn in ((RI.CDC, 'combine'), (RI.HA, 'apply'), (RI.CA, 'apply'), (RI.DK, 'correct')): setattr(mod, fn, timed('combine', getattr(mod, fn)))
RI.Recipe.__init__ = timed('recipe_init', RI.Recipe.__init__)


def small(v):
    if isinstance(v, (bool, np.bool_)): return bool(v)
    if isinstance(v, (int, float, np.floating, np.integer)): return float(v)
    if isinstance(v, np.ndarray) and v.size <= 64 and v.dtype.kind in 'fiub': return np.asarray(v, float).ravel().tolist()
    if isinstance(v, (list, tuple)) and len(v) <= 64 and all(isinstance(u, (int, float, np.floating, np.integer)) for u in v): return [float(u) for u in v]
    if isinstance(v, dict):
        d = {str(k): small(u) for k, u in v.items()}; d = {k: u for k, u in d.items() if u is not None}; return d or None
    return None


zc = np.load(E + 'lane_cd_20261108/reads/freeze_cd_oof_20261111.npz')   # frame list only (idx identical to 20261109)
MM = np.memmap(M.ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
rows = np.linspace(0, len(zc['idx']) - 1, N).astype(int)
R = RI.Recipe('manifests/deploy.json', 'params/deploy.json')
if os.environ.get('BUDGET_CPS5_FOLD0') == '1':      # 11-11 first runs: CPSAM f99 was not complete, so cps5 read the fold-0 weights (same architecture / size)
    R.m['cps5'] = json.load(open(RI.HERE + '/manifests/fold0.json'))['cps5']
assert all(os.path.exists(p) for p in R.m['cps5'])
INIT = dict(T); res = []
for n, k in enumerate(rows):
    d = int(zc['idx'][k]); M.clear_cache(); T.clear(); STACK.clear()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    sync(); t0 = time.perf_counter(); o = R.predict(np.asarray(MM[d])); sync(); wall = time.perf_counter() - t0
    p = o['parts']; parts = {q: small(v) for q, v in p.items() if q not in ('peaks', 'hexfeat')}; parts = {q: v for q, v in parts.items() if v is not None}
    parts['n_peaks'] = [len(p['peaks'][0])] + [len(a) for a in p['peaks'][1]] + [len(a) for a in p['peaks'][2]]
    row = dict(frame=d, wall=wall, stubs=o['stubs'], CD=o['CD'], CV=o['CV'], HEX=o['HEX'], stages=dict(T),
               gpu_peak_gb=(torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None), parts=parts)
    res.append(row); print(json.dumps(dict(frame=d, wall=round(wall, 1), CD=o['CD'], CV=o['CV'], HEX=o['HEX'], gpu_peak_gb=row['gpu_peak_gb'],
                                          top={q: round(v, 1) for q, v in sorted(T.items(), key=lambda x: -x[1]) if '/' not in q})), flush=True)
files = {}
for f in sorted(set(OPENS) | set(READ) | set(CAST)):
    sz = os.stat(f).st_size if os.path.isfile(f) else None
    files[os.path.relpath(f, ROOT) if f.startswith(ROOT) else f] = dict(bytes=sz, read_s=READ.get(f), stages=sorted(OPENS.get(f, [])), cast=CAST.get(f))
summ = dict(variant=VAR, frames=len(res), init_stages=INIT, wall_first=res[0]['wall'], wall_median_rest=float(np.median([r['wall'] for r in res[1:]])) if len(res) > 1 else None,
            read_s_total=float(sum(v for v in READ.values())), read_gb_total=float(sum(os.stat(f).st_size for f in READ if os.path.isfile(f)) / 1e9),
            cast_gb=float(sum(c[1] for c in CAST.values()) / 1e9), kept_float_gb=float(sum(c[2] for c in CAST.values()) / 1e9))
print('SUMMARY', json.dumps(summ), flush=True)
json.dump(dict(summary=summ, rows=res, files=files), open(E + f'lane_cellroute_20261109/out/{OUT}.json', 'w'), indent=1)
