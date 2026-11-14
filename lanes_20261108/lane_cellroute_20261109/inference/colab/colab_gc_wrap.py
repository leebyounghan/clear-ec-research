"""gc_entry.py with CUDA-synchronised stage timers (deploy lane, 2026-11-12, D-PKG-006, real-T4 check on Colab). Same process model as the
container (one image per process): imports gc_entry (its module-level setup), pre-imports recipe_infer with the sys.path / cwd gc_entry.main uses,
wraps the stage builders, runs gc_entry.main() unchanged, then prints 'STAGES {...}' (seconds per stage; nested calls counted in the outer stage)."""
import os, sys, json, time
APP = '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference'
sys.path.insert(0, APP + '/docker')
import gc_entry as GE
os.chdir('/workspace/clear-ec-research'); sys.path.insert(0, APP)
import torch
import recipe_infer as RI, members as M, deploy_hexkonan as DK, pack_hooks as PH
TIMES = {}; CUR = ['-']


def timed(owner, name):
    f = getattr(owner, name)
    def w(*a, **k):
        if CUR[0] != '-': return f(*a, **k)
        lab = name + (f':{a[2] if len(a) > 2 else k.get("fam")}' if name == 'family_mean' else '')
        CUR[0] = lab; torch.cuda.synchronize(); t = time.perf_counter()
        try: return f(*a, **k)
        finally: torch.cuda.synchronize(); TIMES[lab] = TIMES.get(lab, 0.0) + time.perf_counter() - t; CUR[0] = '-'
    setattr(owner, name, w)


for n in ('seg', 'p1', 'peaks', 'other_peaks', 'hex_features', 'hex6', 'cv_parts', '_stage_end'): timed(RI.Recipe, n)
for n in ('preprocess', 'recipeA_tta', 'recipeA_cd', 'dino_cd', 'dino_cv', 'orderdino_cv', 'vitl_tta_only', 'vitl_cd', 'family_mean', 'attnpool_hex'):
    if hasattr(M, n): timed(M, n)
timed(DK, 'features')
GE.main()
print('STAGES', json.dumps({k: round(v, 2) for k, v in TIMES.items()}), flush=True)
