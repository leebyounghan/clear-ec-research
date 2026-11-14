"""List every file the deploy inference reads (cell-route lane, 2026-11-10), for the container build context. Runs one deploy prediction on a training
frame with file-open hooks (builtins.open, os.open, np.load / torch.load via open, safetensors.safe_open), adds the Python modules imported from the repo
and the manifest's weights that are not built yet. Writes docker/files.txt (repo-relative paths) and prints the total size (os.stat; no du).
Stubs for the not-yet-built inputs come from stored values (this is a file census, not a parity test)."""
import builtins, io, os, sys, json
import numpy as np
ROOT = '/workspace/clear-ec-research/'; seen = set()
def note(p):
    try:
        p = os.path.abspath(os.fspath(p))
        if p.startswith(ROOT) and os.path.isfile(p): seen.add(p)
    except TypeError: pass
_open = builtins.open
def hopen(f, *a, **k): note(f); return _open(f, *a, **k)
builtins.open = hopen; io.open = hopen
_oso = os.open
def hosopen(f, *a, **k): note(f); return _oso(f, *a, **k)
os.open = hosopen
try:
    import safetensors, safetensors.torch as st
    _so = safetensors.safe_open
    def hso(f, *a, **k): note(f); return _so(f, *a, **k)
    safetensors.safe_open = hso; _lf = st.load_file
    def hlf(f, *a, **k): note(f); return _lf(f, *a, **k)
    st.load_file = hlf
except ImportError: pass
sys.path.insert(0, ROOT + 'experiments/lane_cellroute_20261109/inference')
import recipe_infer as RI, members as M
E = 'experiments/'; zc = np.load(E + 'lane_cd_20261108/reads/freeze_cd_oof_20261109.npz'); zh = np.load(E + 'lane_hex_20261108/preds/hex_inputs_oof.npz')
census_only = set(); seen_before = set(seen); seen.clear()     # the stored-value files above are test inputs, not inference files
k = int(np.where(zc['fold'] == 3)[0][7]); d = int(zc['idx'][k]); MM = np.memmap(ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
img = np.asarray(MM[d]).copy(); seen.discard(ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat')
R = RI.Recipe('manifests/deploy.json', 'params/deploy.json')
prov = dict(V=zc['V'][k], f6=zh['f6'][k], have=bool(zh['have'][k]), dino_cvw3=zh['dino_cvw3'][k], cps5=zh['cps5'][k])
other = {}
for dname in RI.ALL16[3:]:
    if not os.path.exists(R.m['detectors_all16'][dname][1]):
        z = np.load(E + f'green_track_20261028/{dname}/fold3/fold3_sg2_peaks.npz', allow_pickle=True); j = list(z['indices']).index(d); other[dname] = z['peaks'][z['offsets'][j]:z['offsets'][j + 1]]
import torch
vok = all(os.path.exists(ROOT + q) for sd in R.m['vitl_seeds'] for q in sd['single'] + [sd['tta']])
skip = ([] if vok else ['V']) + ([] if torch.cuda.is_available() else ['f6']) + (['peaks_other13'] if other else [])   # hex6 runs on GPU only
if other: prov['peaks_other13'] = other   # census of the existing detectors is added from the manifest below
o = R.predict(img, provided=prov, skip=tuple(skip)); seen.discard(ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat')
for name, mod in list(sys.modules.items()):
    f = getattr(mod, '__file__', None)
    if f and os.path.abspath(f).startswith(ROOT): note(f)
def walk(v):
    if isinstance(v, str) and (v.endswith('.pt') or v.endswith('.json')): note(ROOT + v if not v.startswith('/') else v); seen.add(os.path.abspath(ROOT + v)) if not os.path.exists(ROOT + v) and v.endswith('.pt') else None
    elif isinstance(v, list): [walk(u) for u in v]
    elif isinstance(v, dict): [walk(u) for u in v.values()]
walk(R.m)
for p in os.listdir(RI.HERE + '/params'): note(os.path.join(RI.HERE, 'params', p))
for p in os.listdir(RI.HERE + '/manifests'): note(os.path.join(RI.HERE, 'manifests', p))
files = sorted(seen); size = 0; missing = []
for f in files:
    if os.path.exists(f): size += os.stat(f).st_size
    else: missing.append(f)
open(RI.HERE + '/docker/files.txt', 'w').write('\n'.join(os.path.relpath(f, ROOT) for f in files) + '\n')
print(json.dumps(dict(files=len(files), total_gb=round(size / 1e9, 2), not_built_yet=[os.path.relpath(f, ROOT) for f in missing], stubs=o['stubs'])), flush=True)
