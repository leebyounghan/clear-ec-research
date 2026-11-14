"""Tier p2 parity / timing harness (deploy lane, 2026-11-14, D-COORD-P2-PREP). One process runs smoke frames through recipe_infer imported from
CODEDIR (the tier-p tree or the p2 context copy) with manifests/params deploy_<TIER>.json, as test_t4.py does: packaging hooks on (fp16 copies
from deploy_fp16, base weights skipped), unloading on, per-frame member log cache cleared, CUDA-synchronised stage timers. Precision path and the
p2 switches come from the environment (fp16b1 = CLEAR_EC_PREC=fp16 CLEAR_EC_DET_COPIES=1; CLEAR_EC_RO_{DINOCV,RECIPEA,HEX6,CPSAM}).
Smoke frames = the 20 deploy frames of test_t4.py (np.linspace over freeze_cd_oof_20261111 idx), positions IDX. Records per frame CD / CV / HEX
(full float), every small part, stage seconds, read bytes / seconds and read-once cache hits of the frame, and ru_maxrss after the frame (the first
frame's value = one fresh job's peak RSS). usage: p2_parity_20261114.py CODEDIR TIER OUT.json IDX(comma list, e.g. 0,5,10,15,19)"""
import os, sys, json, time, resource
ROOT = '/workspace/clear-ec-research/'; L = ROOT + 'experiments/lane_cellroute_20261109/'
CODE = os.path.abspath(sys.argv[1]); TIER = sys.argv[2]; OUT = sys.argv[3]; IDX = [int(v) for v in sys.argv[4].split(',')]
os.environ.update(CLEAR_EC_SKIP_BASE=CODE + '/docker/base_shapes.json', CLEAR_EC_FP16_LIST=CODE + '/docker/fp16_files.txt', CLEAR_EC_FP16_ROOT=L + 'deploy_fp16')
os.environ.setdefault('CLEAR_EC_UNLOAD', '1')
import numpy as np, torch
torch.set_num_threads(int(os.environ.get('THREADS', '2')))
sys.path.insert(0, CODE)
import pack_hooks as PH
import recipe_infer as RI, members as M, deploy_hexkonan as DK
for mod in (PH, RI, M): assert os.path.dirname(os.path.abspath(mod.__file__)) == CODE, (mod.__name__, mod.__file__)
zc = np.load(ROOT + 'experiments/lane_cd_20261108/reads/freeze_cd_oof_20261111.npz')
MM = np.memmap(M.ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
rows = np.linspace(0, len(zc['idx']) - 1, 20).astype(int)[IDX]
TIMES = {}


def timed(owner, name):
    f = getattr(owner, name)
    def w(*a, **k):
        if PH.CUR_STAGE != '-': return f(*a, **k)
        lab = name + (f':{a[2] if len(a) > 2 else k.get("fam")}' if name == 'family_mean' else '')
        PH.CUR_STAGE = lab; torch.cuda.synchronize(); t = time.perf_counter()
        try: return f(*a, **k)
        finally: torch.cuda.synchronize(); TIMES[lab] = TIMES.get(lab, 0.0) + time.perf_counter() - t; PH.CUR_STAGE = '-'
    setattr(owner, name, w)


for n in ('seg', 'p1', 'peaks', 'other_peaks', 'hex_features', 'hex6', 'cv_parts', '_stage_end'): timed(RI.Recipe, n)
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


R = RI.Recipe(f'manifests/deploy_{TIER}.json', f'params/deploy_{TIER}.json'); res = []
KEYS = ('read_bytes', 'read_s', 'ro_hits', 'ro_hit_bytes', 'fp16_loads')
env = {q: os.environ.get(q) for q in ('CLEAR_EC_PREC', 'CLEAR_EC_DET_COPIES', 'CLEAR_EC_NO_TF32', 'CLEAR_EC_UNLOAD', 'THREADS', 'CUDA_VISIBLE_DEVICES',
                                      'CLEAR_EC_RO_DINOCV', 'CLEAR_EC_RO_RECIPEA', 'CLEAR_EC_RO_HEX6', 'CLEAR_EC_RO_CPSAM')}
for k in rows:
    d = int(zc['idx'][k]); M.clear_cache(); TIMES.clear(); s0 = {q: PH.STATS.get(q, 0) for q in KEYS}; PH.STATS['ro_peak_bytes'] = 0
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.perf_counter()
    o = R.predict(np.asarray(MM[d])); torch.cuda.synchronize(); wall = time.perf_counter() - t
    parts = {}
    for q, v in o['parts'].items():
        if q != 'peaks': flat(v, q, parts)
    st = {q: round(v, 3) for q, v in TIMES.items()}; st['rest'] = round(wall - sum(TIMES.values()), 3)
    res.append(dict(frame=d, CD=float(o['CD']), CV=float(o['CV']), HEX=float(o['HEX']), stubs=o['stubs'], wall=round(wall, 2), stage_s=st, parts=parts,
                    reads={q: PH.STATS.get(q, 0) - s0[q] for q in KEYS}, ro_peak_bytes=PH.STATS.get('ro_peak_bytes', 0),
                    ru_maxrss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, gpu_alloc_gb=torch.cuda.max_memory_allocated() / 1e9))
    print(json.dumps({q: res[-1][q] for q in ('frame', 'CD', 'CV', 'HEX', 'wall', 'reads', 'ru_maxrss_gb')}), flush=True)
    json.dump(dict(code=CODE, tier=TIER, idx=IDX, env=env, ro_groups=PH.STATS.get('ro_groups'), gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                   done=len(res) == len(rows), rows=res), open(OUT, 'w'))
print('DONE', OUT, flush=True)
