"""Who opens the base weight files with the packaging hooks on? (deploy lane, 2026-11-12; CPU). Prints a short stack for every Python-level open of
a base file while the CPDINO-based builders run (recipe A member, the recipe A token encoder, one full and one frozen-head detector, hex6, the
pre-only detectors) and the timm ViT-L / DINOv3-B builders."""
import os, sys, builtins, io, traceback, json
ROOT = '/workspace/clear-ec-research/'; L = ROOT + 'experiments/lane_cellroute_20261109/'; D = L + 'inference/docker/'
os.environ.update(CLEAR_EC_FP16_LIST=D + 'fp16_files.txt', CLEAR_EC_FP16_ROOT=L + 'deploy_fp16', CLEAR_EC_SKIP_BASE=D + 'base_shapes.json')
BASES = ('cpdino-vitb', 'vit_large_patch16_dinov3', 'dinov3-vitb16/model.safetensors'); HITS = []
def chk(f):
    s = str(f)
    if any(b in s for b in BASES): HITS.append(dict(file=s, stack=[f'{os.path.basename(fr.filename)}:{fr.lineno} {fr.name}' for fr in traceback.extract_stack()[-8:-2]]))
_open = builtins.open
def hopen(f, *a, **k): chk(f); return _open(f, *a, **k)
builtins.open = hopen; io.open = hopen
_oso = os.open
def hosopen(f, *a, **k): chk(f); return _oso(f, *a, **k)
os.open = hosopen
import torch; torch.set_num_threads(8)
sys.path.insert(0, L + 'inference'); import pack_hooks as PH; assert PH.ACTIVE
import recipe_infer as RI, members as M
m = RI.Recipe('manifests/deploy.json', 'params/deploy.json').m
steps = [('recipeA member', lambda: M.Member(m['recipeA_tta'][0], 'cpdino_recipeA')), ('dinov3_b16 member', lambda: M.Member(m['dino_cd'][0][0], 'dinov3_b16')),
         ('vitl member', lambda: M.Member(m['vitl_seeds'][0]['tta'], 'dinov3_l16t'))]
import deploy_hexkonan_image as DI
k0, (kind0, p0) = next(iter(m['detectors_all16'].items())); kf, (kindf, pf) = [(k, v) for k, v in m['detectors_all16'].items() if v[0] == 'frozen'][0]
steps += [('full detector ' + k0, lambda: DI.load_any(kind0, p0, 'cpu')), ('frozen detector ' + kf, lambda: DI.load_any(kindf, pf, 'cpu'))]
out = {}
for name, fn in steps:
    n0 = len(HITS)
    try: fn(); out[name] = dict(ok=True, base_opens=HITS[n0:])
    except Exception as e: out[name] = dict(ok=False, error=repr(e)[:300], base_opens=HITS[n0:])
    print(name, json.dumps(out[name])[:1500], flush=True)
print('SUMMARY', json.dumps(dict(hook_stats=PH.STATS, n_base_opens=len(HITS))), flush=True)
