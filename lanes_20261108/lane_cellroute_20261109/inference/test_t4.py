"""T4-safe dtype path tests (deploy lane, 2026-11-12, D-PKG-004). GPU; THREADS CPU threads (default 2).
Runs recipe_infer on the 20 deploy frames of every budget / packaged run with the packaging hooks on, unloading on, tier CLEAR_EC_TIER (t1), and the
precision variant given by the environment (pack_hooks: CLEAR_EC_PREC, CLEAR_EC_FP32_MODULES, CLEAR_EC_NO_TF32). REF32=1: original fp32 weight files
instead of the fp16 copies (the reference the lanes' numbers were computed with). Records per frame: CD / CV / HEX, every small part, a hash and
count of every detector's peak set and of the segmentation points, CUDA-synchronised seconds per stage, GPU peak (allocated / reserved), TF32 flags.
No comparison here (compare_t4.py). usage: test_t4.py N OUT [FIRST]   (frames FIRST .. FIRST+N-1 of the 20)"""
import os, sys, json, time, hashlib
ROOT = '/workspace/clear-ec-research/'; L = ROOT + 'experiments/lane_cellroute_20261109/'; D = L + 'inference/docker/'
os.environ['CLEAR_EC_SKIP_BASE'] = D + 'base_shapes.json'
if os.environ.get('REF32') != '1': os.environ.update(CLEAR_EC_FP16_LIST=D + 'fp16_files.txt', CLEAR_EC_FP16_ROOT=L + 'deploy_fp16')
os.environ.setdefault('CLEAR_EC_UNLOAD', '1')
import numpy as np, torch
torch.set_num_threads(int(os.environ.get('THREADS', '2')))
sys.path.insert(0, L + 'inference')
import pack_hooks as PH
import recipe_infer as RI, members as M, deploy_hexkonan as DK
N = int(sys.argv[1]); OUT = sys.argv[2]; FIRST = int(sys.argv[3]) if len(sys.argv) > 3 else 0
zc = np.load(ROOT + 'experiments/lane_cd_20261108/reads/freeze_cd_oof_20261111.npz')
MM = np.memmap(M.ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
rows = (np.load(os.environ['ROWS_FILE']) if os.environ.get('ROWS_FILE') else np.linspace(0, len(zc['idx']) - 1, 20).astype(int))[FIRST:FIRST + N]
# ROWS_FILE (D-GPU-001): rows of the label table (zc / best_pipeline order) instead of the 20 deploy frames; CLEAR_EC_UNLOAD=0 keeps the built
# models across frames (same numerics, about half the time per frame); the member log cache is still cleared per frame
TIER = os.environ.get('CLEAR_EC_TIER', ''); SUFT = f'_{TIER}' if TIER else ''
TIMES, EXTRA = {}, {}


def h(a):
    a = np.ascontiguousarray(np.asarray(a, np.float64)); return dict(n=int(len(a)), md5=hashlib.md5(np.round(a, 4).tobytes()).hexdigest()[:12])


def post_peaks(r):
    EXTRA['peaks'] = dict(mean=h(r[0]), tta=[h(a) for a in r[1]], single=[h(a) for a in r[2]])
def post_other(r): EXTRA['other'] = {d: h(a) for d, a in r.items()}
def post_seg(r):
    if isinstance(r, tuple) and r[1] is not None: EXTRA['segpts'] = h(r[1])


FLOPS = {} if os.environ.get('FLOPS') == '1' else None      # FLOPS=1: matmul / conv / attention FLOPs per stage (torch.utils.flop_counter formulas)
if FLOPS is not None:
    from torch.utils.flop_counter import flop_registry
    from torch.utils._python_dispatch import TorchDispatchMode
    class FlopCounterMode(TorchDispatchMode):      # FlopCounterMode without its module tracker (which needs autograd hooks the attention pool cannot take)
        def __init__(self, display=False): super().__init__(); self.total = 0
        def get_total_flops(self): return self.total
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}; out = func(*args, **kwargs); p = func._overloadpacket
            if p in flop_registry: self.total += flop_registry[p](*args, **kwargs, out_val=out)
            return out
# kernel-variation proxies (native bf16): on the H200 torch 2.11 runs bf16 attention as cuDNN attention; an A10G (sm86) runs FlashAttention-2
if os.environ.get('SDPA_BACKEND') == 'flash': torch.backends.cuda.enable_cudnn_sdp(False)
if os.environ.get('SDPA_BACKEND') == 'efficient': torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_cudnn_sdp(False)


def timed(owner, name, post=None):
    f = getattr(owner, name)
    def w(*a, **k):
        if PH.CUR_STAGE != '-': return f(*a, **k)            # nested call: counted in the outer stage
        lab = name + (f':{a[2] if len(a) > 2 else k.get("fam")}' if name == 'family_mean' else '')
        PH.CUR_STAGE = lab; torch.cuda.synchronize(); t = time.perf_counter()
        try:
            if FLOPS is None: r = f(*a, **k)
            else:
                fc = FlopCounterMode(display=False)
                with fc: r = f(*a, **k)
                FLOPS[lab] = FLOPS.get(lab, 0) + fc.get_total_flops()
        finally: torch.cuda.synchronize(); TIMES[lab] = TIMES.get(lab, 0.0) + time.perf_counter() - t; PH.CUR_STAGE = '-'
        if post: post(r)
        return r
    setattr(owner, name, w)


for n, p in (('seg', post_seg), ('p1', None), ('peaks', post_peaks), ('other_peaks', post_other), ('hex_features', None), ('hex6', None), ('cv_parts', None), ('_stage_end', None)):
    timed(RI.Recipe, n, p)
for n in ('preprocess', 'recipeA_tta', 'recipeA_cd', 'dino_cd', 'dino_cv', 'orderdino_cv', 'vitl_tta_only', 'vitl_cd', 'family_mean', 'attnpool_hex'):
    if hasattr(M, n): timed(M, n)
timed(DK, 'features')


def flat(v, pre, out):
    if isinstance(v, dict):
        for k, x in v.items(): flat(x, f'{pre}.{k}', out)
    elif isinstance(v, (list, tuple)) and v and isinstance(v[0], (dict, list, tuple, np.ndarray)):
        for k, x in enumerate(v): flat(x, f'{pre}.{k}', out)
    else:
        try: a = np.asarray(v, float).ravel()
        except (TypeError, ValueError): return
        if 0 < a.size <= 64: out[pre] = a.tolist()


R = RI.Recipe(f'manifests/deploy{SUFT}.json', f'params/deploy{SUFT}.json'); res = []
for k in rows:
    d = int(zc['idx'][k]); M.clear_cache(); TIMES.clear(); EXTRA.clear()
    if FLOPS is not None: FLOPS.clear()
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.perf_counter()
    o = R.predict(np.asarray(MM[d])); torch.cuda.synchronize(); wall = time.perf_counter() - t
    parts = {}
    for q, v in o['parts'].items():
        if q != 'peaks': flat(v, q, parts)
    st = {q: round(v, 3) for q, v in TIMES.items()}; st['rest'] = round(wall - sum(TIMES.values()), 3)
    res.append(dict(frame=d, CD=o['CD'], CV=o['CV'], HEX=o['HEX'], stubs=o['stubs'], wall=round(wall, 2), stage_s=st, parts=parts, **EXTRA,
                    gpu_alloc_gb=torch.cuda.max_memory_allocated() / 1e9, gpu_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
                    tf32=dict(matmul=bool(torch.backends.cuda.matmul.allow_tf32), cudnn=bool(torch.backends.cudnn.allow_tf32)),
                    **({'tflop': {q: round(v / 1e12, 3) for q, v in FLOPS.items()}} if FLOPS is not None else {})))
    if FLOPS is not None: print('TFLOP', json.dumps(res[-1]['tflop']), flush=True)
    print(json.dumps({q: res[-1][q] for q in ('frame', 'CD', 'CV', 'HEX', 'wall', 'gpu_alloc_gb', 'gpu_reserved_gb', 'tf32', 'stage_s')}), flush=True)
    if len(res) % 10 == 0: json.dump(dict(summary=dict(partial=True, frames=len(res)), rows=res), open(L + f'out/{OUT}_partial.json', 'w'))
summ = dict(frames=len(res), tier=TIER or 'full', ref32=os.environ.get('REF32') == '1', prec=PH.PREC, autocast=PH.AUTOCAST, unload=RI.UNLOAD,
            env={q: os.environ.get(q) for q in ('CLEAR_EC_PREC', 'CLEAR_EC_FP32_MODULES', 'CLEAR_EC_NO_TF32', 'THREADS', 'CUDA_VISIBLE_DEVICES', 'SDPA_BACKEND', 'REF32', 'FLOPS', 'CLEAR_EC_DET_COPIES', 'CLEAR_EC_UNLOAD', 'ROWS_FILE')},
            gpu=torch.cuda.get_device_name(0), torch=torch.__version__, hook_stats=PH.STATS, wall_median=float(np.median([r['wall'] for r in res])),
            gpu_alloc_max_gb=max(r['gpu_alloc_gb'] for r in res), gpu_reserved_max_gb=max(r['gpu_reserved_gb'] for r in res))
print('SUMMARY', json.dumps(summ), flush=True)
json.dump(dict(summary=summ, rows=res), open(L + f'out/{OUT}.json', 'w'), indent=1)
