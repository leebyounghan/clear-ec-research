"""Are the pretrained base files needed at inference? (cell-route lane, 2026-11-11). Test harness, CPU.
The builders read base weights (timm ViT-L dinov3 model.safetensors, DINOv3-B model.safetensors, CPDINO ViT-B cellpose file) and then load each
checkpoint with a strict load_state_dict. Here every float tensor read from those base files is replaced by random noise; if the outputs after the
strict load are bit-identical to a normal build, the base tensors are fully overwritten and only the architecture (config) is needed.
Families: recipe A (cpdino_recipeA), DINOv3-B, ViT-L (members.Member) and one full detector (deploy_hexkonan_image.load_any + single_peaks).
usage: test_base_weights.py THREADS"""
import sys, os, json
import numpy as np, torch
torch.set_num_threads(int(sys.argv[1]))
BASE = ('phase2_backbone_screen_20261001/hf_cache', 'phase2_foundation_20261002/weights')
STATE = dict(random=False, hits=0)
def isbase(f):
    try: return any(b in os.fspath(f) for b in BASE)
    except TypeError: return False
def rnd(o):
    if isinstance(o, torch.Tensor):
        if o.is_floating_point(): STATE['hits'] += 1; return torch.randn_like(o)
        return o
    if isinstance(o, dict):
        for k in list(o.keys()): o[k] = rnd(o[k])
        return o
    if isinstance(o, list): return [rnd(v) for v in o]
    return o
_load = torch.load
def hload(f, *a, **k):
    o = _load(f, *a, **k)
    return rnd(o) if STATE['random'] and isbase(f) else o
torch.load = hload
import safetensors, safetensors.torch as st
_lf = st.load_file
def hlf(f, *a, **k):
    o = _lf(f, *a, **k)
    return rnd(o) if STATE['random'] and isbase(f) else o
st.load_file = hlf
_so = safetensors.safe_open
class SO:
    def __init__(s, f, *a, **k): s._f = _so(f, *a, **k); s._r = STATE['random'] and isbase(f)
    def __enter__(s): s._f.__enter__(); return s
    def __exit__(s, *a): return s._f.__exit__(*a)
    def get_tensor(s, k):
        t = s._f.get_tensor(k)
        return rnd(t) if s._r else t
    def get_slice(s, k):
        sl = s._f.get_slice(k)
        if not s._r: return sl
        class R:
            def __getitem__(self, idx): return rnd(sl[idx])
            def __getattr__(self, n): return getattr(sl, n)
        return R()
    def __getattr__(s, n): return getattr(s._f, n)
safetensors.safe_open = SO
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference')
import members as M
import deploy_hexkonan_image as DI
MM = np.memmap(M.ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
img = np.asarray(MM[1286]); x = M.preprocess(img)
m = json.load(open(M.ROOT + 'experiments/lane_cellroute_20261109/inference/manifests/deploy.json'))
cases = [('cpdino_recipeA', m['recipeA_tta'][0]), ('dinov3_b16', m['dino_cd'][-1]), ('dinov3_l16t', m['vitl_seeds'][0]['tta'])]
out = {}
for fam, path in cases:
    r = {}
    for mode in (False, True):
        STATE['random'] = mode; STATE['hits'] = 0; M.Member._cache.clear()
        r[mode] = (M.Member(path, fam).logpred(x, False), STATE['hits'])
    out[fam] = dict(normal=r[False][0].tolist(), random_base=r[True][0].tolist(), identical=bool(np.array_equal(r[False][0], r[True][0])), base_tensors_randomised=r[True][1])
    print(fam, out[fam], flush=True)
kind, path = m['detectors_all16']['sg2_specialist']; r = {}
for mode in (False, True):
    STATE['random'] = mode; STATE['hits'] = 0
    enc, head = DI.load_any(kind, path, 'cpu')
    with torch.no_grad(): hm = DI.heat(enc, head, DI.preprocess(img).repeat(4, 1, 1, 1))[0]
    r[mode] = (np.asarray(hm), STATE['hits'])
out['detector_sg2_s46'] = dict(identical=bool(np.array_equal(r[False][0], r[True][0])), max_abs=float(np.max(np.abs(r[False][0] - r[True][0]))), base_tensors_randomised=r[True][1])
print('detector', out['detector_sg2_s46'], flush=True)
print('SUMMARY', json.dumps({k: (v['identical'], v['base_tensors_randomised']) for k, v in out.items()}), flush=True)
json.dump(out, open(M.ROOT + 'experiments/lane_cellroute_20261109/out/base_weights_check.json', 'w'), indent=1)
