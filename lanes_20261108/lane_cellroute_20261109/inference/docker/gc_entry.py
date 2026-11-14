"""Grand Challenge entrypoint for the 11-12 freeze candidate (deploy lane, D-PKG-001): D-CD-001 + D-CV-001 + D-HEX-004 + D-HEX-005 through
recipe_infer.Recipe('manifests/deploy.json', 'params/deploy.json'). Same I/O contract as submissions 1 and 2:
  /input/inputs.json with the single socket 'corneal-specular-microscopy-image', one .mha under /input/images/corneal-specular-microscopy-image/,
  three JSON floats written to /output/{cell-density,coefficient-of-variation,hexagonality}.json (never overwritten).
Weights are mounted read-only at /opt/ml/model with the research-tree relative paths; the image links them into /workspace/clear-ec-research.
Packaging hooks (pack_hooks.py): CLEAR_EC_FP16_LIST (fp16 copies upcast at load), CLEAR_EC_SKIP_BASE (base weights not read). Start-up checks stay
loud (CUDA, model files present). A per-image failure of the recipe falls back to the MAPE-optimal training constants of submission 1 (CD 2770,
CV 0.36, HEX 0.53) and is logged, so one bad hidden image cannot fail the whole evaluation; a non-uint8 or multi-channel input is converted first."""
import json, os, sys, time, traceback
from pathlib import Path
T0 = time.time()
APP = '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference'
os.environ.setdefault('CLEAR_EC_FP16_LIST', APP + '/docker/fp16_files.txt'); os.environ.setdefault('CLEAR_EC_SKIP_BASE', APP + '/docker/base_shapes.json')
os.environ.setdefault('CLEAR_EC_UNLOAD', '1')   # one image per job: free each stage's models (GPU peak 4.3 GB instead of ~17 GB; a T4 has 15 GB)
# the research pod runs with PYTHONPATH = this runtime folder (CPDINO backbones runtime/dinov3, timm 1.0.20 copy); first on the path, as there,
# before anything imports cellpose / timm (2026-11-12 real-T4 check: without it every CPDINO member raised NameError -> constants on every image)
RUNTIME = '/workspace/clear-ec-research/experiments/phase2_foundation_20261002/runtime'
if RUNTIME not in sys.path: sys.path.insert(0, RUNTIME)
FALLBACK = (2770.0, 0.36, 0.53)
SLUG = 'corneal-specular-microscopy-image'; OUTS = ('cell-density', 'coefficient-of-variation', 'hexagonality')


def coerce(a):
    import numpy as np
    a = np.asarray(a); notes = []
    while a.ndim > 2:
        if a.shape[0] == 1: a = a[0]
        elif a.shape[-1] in (3, 4): a = a[..., :3].mean(-1); notes.append('rgb->gray')
        else: a = a[0]; notes.append('first plane')
    if a.dtype != np.uint8:
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a)); a = np.clip((a - lo) / (hi - lo + 1e-12) * 255.0 + 0.5, 0, 255).astype(np.uint8); notes.append('to uint8')
    return a, notes


def main():
    import torch, SimpleITK as sitk
    inp, out = Path(os.environ.get('GC_INPUT', '/input')), Path(os.environ.get('GC_OUTPUT', '/output'))
    if not torch.cuda.is_available(): raise RuntimeError('CUDA not available')
    sockets = json.loads((inp / 'inputs.json').read_text())
    if [x['socket']['slug'] for x in sockets] != [SLUG]: raise ValueError('expected the single microscopy input socket')
    paths = sorted((inp / 'images' / SLUG).glob('*.mha'))
    if len(paths) != 1: raise ValueError(f'expected exactly one MHA, found {len(paths)}')
    outputs = [out / f'{k}.json' for k in OUTS]
    if any(p.exists() for p in outputs): raise FileExistsError('refusing to overwrite an existing prediction')
    os.chdir('/workspace/clear-ec-research'); sys.path.insert(0, APP)
    import recipe_infer as RI, pack_hooks as PH
    import cellpose.vit as CVIT       # loud start-up check: cellpose hides a failed CPDINO backbone import behind a bare except
    if not hasattr(CVIT, 'dinov3_vitb16'): raise ImportError(f'CPDINO backbones not importable (dinov3 from {RUNTIME}); every detector would fail')
    tier = os.environ.get('CLEAR_EC_TIER', ''); suf = f'_{tier}' if tier else ''
    R = RI.Recipe(f'manifests/deploy{suf}.json', f'params/deploy{suf}.json')
    missing = [p for v in (R.m.get('p1_members', []), R.m.get('cvw3', []), R.m.get('cps5', [])) for p in v if not os.path.exists(p)]
    if missing: raise FileNotFoundError(f'model files missing: {missing[:3]}')
    # recipe_infer turns a missing CV parameter folder into cvp=None and every image would get the constants (2026-11-12, tiers t2 / t3 on the
    # real T4 with the t1 code tree): loud instead
    if R.cvp is None: raise FileNotFoundError(f"CV deploy parameters missing: {R.p['cv'].get('deploy_dir')}")
    t_setup = time.time() - T0; info = dict(mode='recipe', errors={}, notes=[])
    try:
        img, info['notes'] = coerce(sitk.GetArrayFromImage(sitk.ReadImage(str(paths[0]))))
        if img.shape != (972, 1296): raise ValueError(f'unsupported frame shape {img.shape}')
        o = RI.Recipe.predict(R, img); value = (o['CD'], o['CV'], o['HEX'])
        import math
        if not all(math.isfinite(v) and v > 0 for v in value): raise ValueError(f'invalid output {value}')
        if o['stubs']: raise RuntimeError(f'stubbed inputs in deploy mode: {o["stubs"]}')
    except Exception as e:
        value = FALLBACK; info.update(mode='constant', errors=dict(predict=f'{type(e).__name__}: {e}'[:400]), trace=traceback.format_exc()[-1500:])
    out.mkdir(parents=True, exist_ok=True)
    for p, v in zip(outputs, value):
        with p.open('x') as h: json.dump(float(v), h, allow_nan=False)
    import resource
    print(json.dumps(dict(seconds=round(time.time() - T0, 1), setup_s=round(t_setup, 1), ram_peak_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2), unload=RI.UNLOAD, output=[float(v) for v in value], gpu=torch.cuda.get_device_name(0),
                          gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), hooks=PH.STATS, decisions=R.p.get('decisions'), **info)), flush=True)


if __name__ == '__main__':
    main()
