"""Bundle list for one bb_runner_20261108.py dinov3_b16 fold-0 seed-46 run on the build box (deploy lane, 2026-11-13). CPU, seconds; copies nothing.
Contents = what the traced smoke run touched (out/boxtrain_trace_20261113: audit opens + /proc maps / fds in the main process and the DataLoader
workers; the tracer's own metadata scans excluded), plus the whole runtime folder (PYTHONPATH of the pod and of the image; dinov3 / timm 1.0.20
code), file list taken from the staged p context, which holds the same runtime files without weights. Writes, next to this script,
bundle_files_20261113.txt (paths relative to /workspace/clear-ec-research, for rsync --files-from) and bundle_files_20261113.tsv (path, bytes,
sha256 of files below 5 MB)."""
import os, hashlib
ROOT = '/workspace/clear-ec-research/'; HERE = ROOT + 'experiments/lane_cellroute_20261109/inference/boxtrain_20261113/'
TRACED = [
    'datasets/clear_ec_fullframe_20260904/img_full_raw.dat',                       # whole-frame memmap (8954, 972, 1296) uint8, workers
    'datasets/clear_ec_official/cache_native_meta_512.npz',                        # ids + labels Y (b.META)
    'experiments/ssl_screen_20260911/preflight/fold0.npz',                         # fold-0 split
    'experiments/lane_gpu3_20261106/runs/dinov3_b16_f0_s46/ssl/last.pt',           # fold-0 SSL checkpoint (--ssl)
    'experiments/phase2_foundation_20261002/weights/dinov3-vitb16/config.json',    # base DINOv3-B/16 (HF from_pretrained)
    'experiments/phase2_foundation_20261002/weights/dinov3-vitb16/model.safetensors',
    'experiments/lane_backbone_20261108/source/bb_runner_20261108.py',
    'experiments/lane_backbone_20261108/source/family_bb_20261108.py',
    'experiments/oof_recipeA_20261027/staging_regression_zero/phase2_backbone_ssl_regression_20261020.py',
    'experiments/oof_recipeA_20261027/staging_regression_zero/phase2_foundation_preflight_20261002.py',
    'experiments/oof_recipeA_20261027/staging_regression_zero/phase2_foundation_train_20261002.py',
    'experiments/oof_recipeA_20261027/staging_regression_zero/phase2_lejepa_regression_20261003.py',
    'experiments/oof_recipeA_20261027/staging_regression_zero/phase2_pooling_20261019.py',
    'experiments/phase2_crossfold_20261015/source/phase2_crossfold_ssl_20261015.py',
    'experiments/phase2_crossfold_20261015/source/phase2_lejepa_20261003.py',
]
CTX = ROOT + 'experiments/lane_cellroute_20261109/package_20261112/p/context/code/'
RT = 'experiments/phase2_foundation_20261002/runtime/'
runtime = sorted(os.path.relpath(os.path.join(dp, f), CTX) for dp, dn, fn in os.walk(CTX + RT) for f in fn)
files = TRACED + [r for r in runtime if r not in TRACED]
missing = [r for r in files if not os.path.isfile(ROOT + r)]
assert not missing, missing
rows = []
for r in files:
    n = os.stat(ROOT + r).st_size
    h = hashlib.sha256(open(ROOT + r, 'rb').read()).hexdigest() if n < 5 << 20 else ''
    rows.append((r, n, h))
open(HERE + 'bundle_files_20261113.txt', 'w').write(''.join(r + '\n' for r, _, _ in rows))
open(HERE + 'bundle_files_20261113.tsv', 'w').write(''.join(f'{r}\t{n}\t{h}\n' for r, n, h in rows))
tot = sum(n for _, n, _ in rows)
print(f'files {len(rows)} (traced {len(TRACED)}, runtime {len(runtime)}) bytes {tot:,} = {tot / 1e9:.3f} GB; runtime {sum(n for r, n, _ in rows if r.startswith(RT)) / 1e6:.1f} MB')
for r, n, _ in rows[:15]: print(f'  {n:>14,}  {r}')
