"""Grand Challenge style job test outside docker (deploy lane, 2026-11-12, D-PKG-001): every frame is its own job, a fresh python process running
docker/gc_entry.py on a fresh /input layout (inputs.json + one .mha written with SimpleITK) and an empty output dir, exactly the container's
contract; wall time measured from process start to exit (cold start included). On the pod the fp16 copies are read from deploy_fp16/
(CLEAR_EC_FP16_ROOT); in the container they sit at the original paths. Outputs compared with out/packaged_parity_20.json (same frames, same hooks).
usage: test_gc_jobs.py N OUT"""
import os, sys, json, time, subprocess, resource, shutil
import numpy as np
ROOT = '/workspace/clear-ec-research/'; L = ROOT + 'experiments/lane_cellroute_20261109/'; D = L + 'inference/docker/'
import SimpleITK as sitk
zc = np.load(ROOT + 'experiments/lane_cd_20261108/reads/freeze_cd_oof_20261111.npz')
MM = np.memmap(ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
N = int(sys.argv[1]); OUT = sys.argv[2]; rows = np.linspace(0, len(zc['idx']) - 1, N).astype(int)
TIERX = os.environ.get('CLEAR_EC_TIER', '')
RF = L + (f'out/packaged_tier_{TIERX}_20.json' if TIERX else 'out/packaged_parity_20.json')     # the in-process run of the same tier
ref = {r['frame']: r for r in json.load(open(RF))['rows']} if os.path.exists(RF) else {}
W = L + f'gcjobs/{OUT}/'; res = []
env = dict(os.environ, CLEAR_EC_FP16_ROOT=L + 'deploy_fp16', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
for k in rows:
    d = int(zc['idx'][k]); jd = W + f'job_{d}/'; shutil.rmtree(jd, ignore_errors=True)
    os.makedirs(jd + 'input/images/corneal-specular-microscopy-image'); os.makedirs(jd + 'output')
    json.dump([{'socket': {'slug': 'corneal-specular-microscopy-image'}, 'image': {'name': f'{d}.mha'}}], open(jd + 'input/inputs.json', 'w'))
    sitk.WriteImage(sitk.GetImageFromArray(np.asarray(MM[d])), jd + f'input/images/corneal-specular-microscopy-image/{d}.mha')
    t = time.time()
    p = subprocess.run([sys.executable, D + 'gc_entry.py'], env=dict(env, GC_INPUT=jd + 'input', GC_OUTPUT=jd + 'output'), capture_output=True, text=True)
    wall = time.time() - t; log = [l for l in p.stdout.splitlines() if l.startswith('{')]
    o = [json.load(open(jd + f'output/{n}.json')) for n in ('cell-density', 'coefficient-of-variation', 'hexagonality')] if p.returncode == 0 else None
    info = json.loads(log[-1]) if log else {}
    r = ref.get(d); rel = [abs(a / b - 1) for a, b in zip(o, (r['CD'], r['CV'], r['HEX']))] if (o and r) else None
    res.append(dict(frame=d, rc=p.returncode, wall_s=round(wall, 1), entry_seconds=info.get('seconds'), setup_s=info.get('setup_s'), mode=info.get('mode'), errors=info.get('errors'),
                    gpu=info.get('gpu'), gpu_peak_gb=info.get('gpu_peak_gb'), ram_peak_gb=info.get('ram_peak_gb'), hooks=info.get('hooks'), tier=os.environ.get('CLEAR_EC_TIER', 'full'), output=o, rel_vs_packaged=rel, maxrss_children_gb=round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6, 2),
                    stderr_tail=p.stderr[-400:] if p.returncode else ''))
    print(json.dumps(res[-1]), flush=True)
ok = [r for r in res if r['rc'] == 0 and r['mode'] == 'recipe']; R = np.array([r['rel_vs_packaged'] for r in ok]) if ok else np.zeros((0, 3))
H = [r['hooks'] or {} for r in ok]
summ = dict(tier=os.environ.get('CLEAR_EC_TIER', 'full'), read_gb_median=float(np.median([h.get('read_bytes', 0) for h in H]) / 1e9) if H else None,
            read_s_median=float(np.median([h.get('read_s', 0) for h in H])) if H else None, setup_s_median=float(np.median([r['setup_s'] or 0 for r in ok])) if ok else None,
            ram_peak_gb=max((r['ram_peak_gb'] or 0) for r in res), jobs=len(res), ok=len(ok), wall_first=res[0]['wall_s'], wall_median=float(np.median([r['wall_s'] for r in res])), wall_max=max(r['wall_s'] for r in res),
            gpu_peak_gb=max((r['gpu_peak_gb'] or 0) for r in res), host_maxrss_gb=res[-1]['maxrss_children_gb'], gpu=res[0]['gpu'],
            max_rel_vs_packaged=R.max(0).tolist() if len(R) else None, identical=int((R == 0).all(1).sum()) if len(R) else 0)
print('SUMMARY', json.dumps(summ), flush=True); json.dump(dict(summary=summ, jobs=res), open(L + f'out/{OUT}.json', 'w'), indent=1)
