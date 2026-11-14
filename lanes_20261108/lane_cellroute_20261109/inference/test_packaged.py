"""Packaged-mode parity and file census (deploy lane, 2026-11-12, D-PKG-001). GPU, <= 4 CPU threads.
Runs recipe_infer on the same 20 deploy frames as test_budget.py with the packaging hooks on (fp16 copies upcast at load from deploy_fp16/, base weights
not read, one cpsam_v2) and compares CD / CV / HEX with (a) the fp16-nodet simulation run (out/budget_fp16-nodet_v3.json; must be identical: same tensors)
and (b) the fp32 run (out/budget_fp32_v3.json; the parity-rule numbers). Census: every file under the repo opened (Python open / os.open, torch.load,
safetensors), every repo .py module imported, third-party package versions, GPU peak. usage: test_packaged.py N OUT"""
import os, sys, json, time, builtins, io, collections
ROOT = '/workspace/clear-ec-research/'; L = ROOT + 'experiments/lane_cellroute_20261109/'; D = L + 'inference/docker/'
os.environ.update(CLEAR_EC_FP16_LIST=D + 'fp16_files.txt', CLEAR_EC_FP16_ROOT=L + 'deploy_fp16', CLEAR_EC_SKIP_BASE=D + 'base_shapes.json')
OPENED = collections.defaultdict(int)
def note(p):
    try:
        p = os.path.abspath(os.fspath(p))
        if p.startswith(ROOT) and os.path.isfile(p): OPENED[os.path.relpath(p, ROOT)] += 1
    except TypeError: pass
_open = builtins.open
def hopen(f, *a, **k): note(f); return _open(f, *a, **k)
builtins.open = hopen; io.open = hopen
_oso = os.open
def hosopen(f, *a, **k): note(f); return _oso(f, *a, **k)
os.open = hosopen
import numpy as np, torch
torch.set_num_threads(4)
sys.path.insert(0, L + 'inference')
import pack_hooks as PH
assert PH.ACTIVE
import safetensors, safetensors.torch as st
_lf = st.load_file
def lf(f, *a, **k):
    if PH.rel(f) not in (PH.BASE or {}).get('safetensors', {}): note(f)        # base paths are replaced by the hook, never read
    return _lf(f, *a, **k)
st.load_file = lf
_so = safetensors.safe_open
def so(f, *a, **k): note(f); return _so(f, *a, **k)
safetensors.safe_open = so
import recipe_infer as RI, members as M
zc = np.load(ROOT + 'experiments/lane_cd_20261108/reads/freeze_cd_oof_20261111.npz')
MM = np.memmap(M.ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
N = int(sys.argv[1]); OUT = sys.argv[2]; rows = np.linspace(0, len(zc['idx']) - 1, 20).astype(int)     # the 20 deploy frames of every budget run
rows = rows[np.linspace(0, 19, N).astype(int)] if N < 20 else rows
TIER = os.environ.get('CLEAR_EC_TIER', ''); SUFT = f'_{TIER}' if TIER else ''
R = RI.Recipe(f'manifests/deploy{SUFT}.json', f'params/deploy{SUFT}.json'); res = []
SHARED = ('A', 'P1', 'f6', 'have', 'best', 'graph', 'g_s10x3', 'img', 'prior_h6c', 'ridge_k3', 'seg17', 'DINO_logcv', 'graph_cv', 'ridge_cv', 'qplain_cv', 'qorder_cv')
def sv(v):
    a = np.asarray(v, float).ravel(); return a.tolist() if a.size <= 32 else None
for k in rows:
    d = int(zc['idx'][k]); M.clear_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.perf_counter()
    o = R.predict(np.asarray(MM[d])); torch.cuda.synchronize()
    res.append(dict(frame=d, CD=o['CD'], CV=o['CV'], HEX=o['HEX'], stubs=o['stubs'], shared={q: sv(o['parts'][q]) for q in SHARED if q in o['parts']}, wall=round(time.perf_counter() - t, 1), gpu_peak_gb=torch.cuda.max_memory_allocated() / 1e9))
    print(json.dumps(res[-1]), flush=True)
json.dump(dict(opened=dict(OPENED), repo_modules=sorted({os.path.relpath(m.__file__, ROOT) for m in list(sys.modules.values()) if getattr(m, '__file__', None) and os.path.abspath(m.__file__).startswith(ROOT)})),
          open(L + f'out/{OUT}_census_early.json', 'w'))      # written before the comparison so a comparison error cannot lose it
REFS = dict(fp16nodet='budget_fp16-nodet_v3', fp32='budget_fp32_v3', **({'bf16_packaged': 'packaged_parity_20'} if OUT != 'packaged_parity_20' else {}))
cmp = {}
for name, fn in REFS.items():
    ref = {r['frame']: r for r in json.load(open(L + f'out/{fn}.json'))['rows']}; rr = [r for r in res if r['frame'] in ref]
    for c in ('CD', 'CV', 'HEX'):
        a = np.array([abs(r[c] / ref[r['frame']][c] - 1) for r in rr])
        cmp[f'{c}_vs_{name}'] = dict(frames=len(rr), median_pct=float(np.median(a) * 100), max_pct=float(a.max() * 100), identical=int((a == 0).sum()))
mods = sorted({os.path.relpath(m.__file__, ROOT) for m in list(sys.modules.values()) if getattr(m, '__file__', None) and os.path.abspath(m.__file__).startswith(ROOT)})
import importlib.metadata as md
top = sorted({n.split('.')[0] for n in sys.modules if n and not n.startswith('_')})
vers = {}
for n in top:
    try: vers[n] = md.version({'cv2': 'opencv-python-headless', 'PIL': 'pillow', 'sklearn': 'scikit-learn', 'skimage': 'scikit-image', 'yaml': 'pyyaml'}.get(n, n))
    except Exception: pass
base_opened = [f for f in OPENED if 'hf_cache' in f or ('phase2_foundation_20261002/weights' in f and not f.endswith('.json'))]
orig_fp16_opened = [f for f in OPENED if f in PH.FP16]
summ = dict(frames=len(res), stubs=res[0]['stubs'], compare=cmp, autocast=PH.AUTOCAST, unload=RI.UNLOAD, tier=TIER or 'full', wall_first=res[0]['wall'], wall_median_rest=float(np.median([r['wall'] for r in res[1:]])),
            gpu_peak_gb=max(r['gpu_peak_gb'] for r in res), hook_stats=PH.STATS, base_weight_files_opened=base_opened, fp16_originals_opened=orig_fp16_opened,
            files_opened=len(OPENED), repo_modules=len(mods))
print('SUMMARY', json.dumps(summ), flush=True)
json.dump(dict(summary=summ, rows=res), open(L + f'out/{OUT}.json', 'w'), indent=1)
json.dump(dict(opened=dict(OPENED), repo_modules=mods, packages=vers, torch=torch.__version__, cuda=torch.version.cuda), open(D + f'pack_census{SUFT}.json', 'w'), indent=1)
