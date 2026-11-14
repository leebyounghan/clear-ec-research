"""Stage the docker build context and the model archive tree for the 11-12 freeze image (deploy lane, D-PKG-001), from the packaged-mode census
(docker/pack_census.json, written by test_packaged.py). Nothing is copied twice on disk: the model tree is hard links into the existing files.
  package_20261112/context/code/<rel>  the repo .py modules imported in deploy mode + the small non-weight files they open (params, manifests, lists)
  package_20261112/context/{Dockerfile, requirements.txt, constraints.txt, model_files.txt, MANIFEST.json}
  package_20261112/model/<rel>         weights at their research-tree paths (fp16 copies at the original paths; fp32 for the rest; base weights not
                                       shipped except the DINOv3-B config.json; one cpsam_v2), = the archive extracted at /opt/ml/model
Test-only inputs (the dataset memmap, stored OOF files, out/ results, test_*.py) are excluded. Sizes by os.stat (no du).
usage: stage_package.py [TIER]   (TIER t1|t2|t3: census docker/pack_census_<tier>.json, output package_20261112/<tier>/; default = the freeze candidate)"""
import os, json, shutil, hashlib, re
ROOT = '/workspace/clear-ec-research/'; L = ROOT + 'experiments/lane_cellroute_20261109/'; D = L + 'inference/docker/'; P = L + 'package_20261112/'
FP16PRE = 'experiments/lane_cellroute_20261109/deploy_fp16/'
import sys
TIER = sys.argv[1] if len(sys.argv) > 1 else ''; SUFT = f'_{TIER}' if TIER else ''
if TIER: P = P + TIER + '/'
C = json.load(open(D + f'pack_census{SUFT}.json')); fp16 = {l.strip() for l in open(D + 'fp16_files.txt') if l.strip()}
TEST = (lambda r: r.startswith('datasets/') or r.startswith('experiments/lane_cellroute_20261109/out/') or '/reads/freeze_cd_oof' in r
        or (r.startswith('experiments/lane_cellroute_20261109/inference/') and re.search(r'/test_[^/]*\.py$', r) is not None) or r.startswith(FP16PRE + 'x'))
# (2026-11-12: the test-file rule used to match any test_*.py, which dropped timm's own runtime/timm/layers/test_time_pool.py; the real-T4 run
#  then failed at ViT-L. It now applies to the lane's own inference tests only.)
WEIGHT = lambda r, n: r.endswith(('.pt', '.pth', '.safetensors', '.bin')) or os.path.basename(r) in ('cpsam_v2',) or n >= (1 << 20)
model, code = {}, set()
for r in C['opened']:
    if TEST(r): continue
    n = os.stat(ROOT + r).st_size
    if r.startswith(FP16PRE): model[r[len(FP16PRE):]] = ROOT + r          # fp16 copy at the original path
    elif WEIGHT(r, n): model[r] = ROOT + r
    else: code.add(r)
skipped = []
for r in C['repo_modules']:
    if TEST(r): continue
    if os.path.isfile(ROOT + r): code.add(r)
    else: skipped.append(r)      # torch dynamic modules (_classes.py, _ops.py) report a non-file __file__
# the runtime packages the pod imports through PYTHONPATH (dinov3, timm 1.0.20 copy) ship whole, not only the modules one census run imported:
# a module first imported on another code path (or dropped by a filter) must not break the image (2026-11-12 real-T4 check)
RT = 'experiments/phase2_foundation_20261002/runtime/'
for top in ('dinov3', 'timm', 'dinov3-0.0.1.dist-info', 'timm-1.0.20.dist-info'):
    for dp, dn, fn in os.walk(ROOT + RT + top):
        dn[:] = [d for d in dn if d != '__pycache__']
        for f in fn:
            r = os.path.relpath(os.path.join(dp, f), ROOT)
            if not WEIGHT(r, os.stat(ROOT + r).st_size): code.add(r)
# docker-side files that the entry reads but the census run did not open (the lists themselves are env-driven in the container)
for r in ('experiments/lane_cellroute_20261109/inference/docker/gc_entry.py', 'experiments/lane_cellroute_20261109/inference/docker/fp16_files.txt',
          'experiments/lane_cellroute_20261109/inference/docker/base_shapes.json', 'experiments/lane_cellroute_20261109/inference/manifests/deploy.json',
          'experiments/lane_cellroute_20261109/inference/params/deploy.json', 'experiments/lane_cellroute_20261109/inference/params/cellroute_deploy.json',
          *[f'experiments/lane_cellroute_20261109/inference/{d}/{n}_{x}.json' for x in ('t1', 't2', 't3', 'p') for d, n in (('manifests', 'deploy'), ('params', 'deploy'), ('params', 'cellroute_deploy'))]):
    code.add(r)
assert not any(r in model for r in fp16 if (FP16PRE + r) not in C['opened']), 'an fp16-listed file was read from its fp32 original'
bad = [r for r in model if 'hf_cache' in r or ('phase2_foundation_20261002/weights' in r and not r.endswith('.json'))]
assert not bad, f'base weights in the model tree: {bad}'
assert sum(1 for r in model if os.path.basename(r) == 'cpsam_v2') == 1, 'cpsam_v2 must ship once'
os.makedirs(P + 'context/code', exist_ok=True); os.makedirs(P + 'model', exist_ok=True)
for r in sorted(code):
    d = P + 'context/code/' + r; os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy2(ROOT + r, d)
for r, src in sorted(model.items()):
    d = P + 'model/' + r; os.makedirs(os.path.dirname(d), exist_ok=True)
    if not os.path.exists(d): os.link(src, d)
open(P + 'context/model_files.txt', 'w').write('\n'.join(sorted(model)) + '\n')
shutil.copy2(D + 'Dockerfile.freeze', P + 'context/Dockerfile')
pk = C['packages']; pins = {k: pk[k] for k in ('timm', 'transformers', 'cellpose', 'SimpleITK', 'scipy', 'numpy', 'safetensors', 'tifffile', 'fastremap', 'natsort', 'roifile', 'segment_anything', 'skimage', 'sklearn', 'PIL', 'opencv-python-headless', 'cv2', 'huggingface_hub', 'tokenizers', 'tqdm') if k in pk}
# the Dockerfile asserts timm 1.0.20 at build time, but the t2 / t3 census runs never import timm (no ViT-L member), so their requirements had no
# timm and the t3 build failed (2026-11-13 build box: ModuleNotFoundError at step 4/9); pinned for every tier, as the p / t1 / full images have it
pins.setdefault('timm', '1.0.20')
name = {'skimage': 'scikit-image', 'sklearn': 'scikit-learn', 'PIL': 'pillow', 'cv2': 'opencv-python-headless'}
req = sorted({f'{name.get(k, k)}=={v}' for k, v in pins.items()})
open(P + 'context/requirements.txt', 'w').write('\n'.join(req) + '\n'); open(P + 'context/constraints.txt', 'w').write('torch==2.11.0\n')
mf = ROOT + 'experiments/lane_cellroute_20261109/inference/manifests/deploy.json'; sha7 = hashlib.sha256(open(mf, 'rb').read()).hexdigest()[:7]
gb = lambda paths: round(sum(os.stat(p).st_size for p in paths) / 1e9, 3)
man = dict(tier=TIER or 'full', tag=f'clear-ec:20261112-freeze-{sha7}' + (f'-{TIER}' if TIER else ''), manifest_sha256_7=sha7, decisions=json.load(open(ROOT + 'experiments/lane_cellroute_20261109/inference/params/deploy.json')).get('decisions'),
           code_files=len(code), code_gb=gb([ROOT + r for r in code]), model_files=len(model), model_gb=gb(model.values()),
           model_fp16_files=sum(1 for r in model if r in fp16), model_fp16_gb=gb([v for r, v in model.items() if r in fp16]),
           requirements=req, torch_on_pod=C['torch'], cuda_on_pod=C['cuda'], skipped_non_file_modules=skipped)
json.dump(man, open(P + 'context/MANIFEST.json', 'w'), indent=1); print('SUMMARY', json.dumps(man), flush=True)
