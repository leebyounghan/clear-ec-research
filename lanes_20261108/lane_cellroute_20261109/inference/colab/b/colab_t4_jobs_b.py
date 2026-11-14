"""Real-T4 Grand Challenge style jobs on the PI's Colab T4 (deploy lane, 2026-11-12, D-PKG-004 / D-PKG-006). Run with /content/venv/bin/python.
Layout as Dockerfile.freeze (colab_setup.sh): code /workspace/clear-ec-research, weights /opt/ml/model with the image's links, tier t1 manifest
copied to deploy.json, fp16 copies at the original paths (CLEAR_EC_FP16_LIST / CLEAR_EC_SKIP_BASE as the image's ENV). Every frame is its own
job: a fresh process (colab_gc_wrap.py = gc_entry.py + stage timers) on a fresh /input layout with one .mha and an empty output dir; wall from
process start to exit. Device GPU memory polled with nvidia-smi every 0.5 s (includes the CUDA context). Threads = THREADS (default 2: the
runtime has 2 CPUs; the image sets 4). MODE picks the precision path and the pack_hooks.py the job imports:
  default  the staged pack_hooks.py (17:00), no setting: the container default on a T4 (legacy rule: fp16 below compute capability 8.0)
  others   the current pack_hooks.py: bf16, fp16, fp16b1, mix, mixb1, fp32, fp32b1, emu (CLEAR_EC_PREC / CLEAR_EC_DET_COPIES)
Results rewritten to /content/t4test/out/OUT.json after every job. usage: colab_t4_jobs.py MODE OUT N [TIMEOUT_S] [FIRST]"""
import os, sys, json, time, glob, shutil, subprocess, threading
APP = '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference'; T = '/content/t4test/'
PY = '/content/venv/bin/python'; WRAP = T + 'colab_gc_wrap.py'
MODS = 'deploy_hexkonan_image,hex6_correction,green_point_features_20261027,green_sg2_specialist_20261028,green_s8_specialist_seed_20261028,green_selftrain_20261028'
MODES = dict(default={}, bf16=dict(CLEAR_EC_PREC='bf16'), fp16=dict(CLEAR_EC_PREC='fp16'), fp16b1=dict(CLEAR_EC_PREC='fp16', CLEAR_EC_DET_COPIES='1'),
             mix=dict(CLEAR_EC_PREC='mix', CLEAR_EC_FP32_MODULES=MODS), mixb1=dict(CLEAR_EC_PREC='mix', CLEAR_EC_FP32_MODULES=MODS, CLEAR_EC_DET_COPIES='1'),
             fp32=dict(CLEAR_EC_PREC='fp32'), mixhex6=dict(CLEAR_EC_PREC='mix', CLEAR_EC_FP32_MODULES='hex6_correction', CLEAR_EC_DET_COPIES='1'), mixkonan=dict(CLEAR_EC_PREC='mix', CLEAR_EC_FP32_MODULES='deploy_hexkonan_image', CLEAR_EC_DET_COPIES='1'), fp32b1=dict(CLEAR_EC_PREC='fp32', CLEAR_EC_DET_COPIES='1'), emu=dict(CLEAR_EC_PREC='emu'))
MODE, OUT, N = sys.argv[1], sys.argv[2], int(sys.argv[3]); TO = float(sys.argv[4]) if len(sys.argv) > 4 else 900; FIRST = int(sys.argv[5]) if len(sys.argv) > 5 else 0
TH = os.environ.get('THREADS', '2')
base = {k: v for k, v in os.environ.items() if not k.startswith('CLEAR_EC_') and k not in ()}
base.update(PYTHONPATH='/workspace/clear-ec-research/experiments/phase2_foundation_20261002/runtime', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', CLEAR_EC_UNLOAD='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS=TH, OPENBLAS_NUM_THREADS=TH, MKL_NUM_THREADS=TH, MALLOC_ARENA_MAX='2',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', HF_HUB_DISABLE_PROGRESS_BARS='1',
            CLEAR_EC_FP16_LIST=APP + '/docker/fp16_files.txt', CLEAR_EC_SKIP_BASE=APP + '/docker/base_shapes.json', **MODES[MODE])
frames = sorted(glob.glob(T + 'frames/*.mha'))[FIRST:FIRST + N]; res = []
drop = subprocess.run('sync; echo 3 > /proc/sys/vm/drop_caches', shell=True, capture_output=True, text=True).returncode == 0 if os.environ.get('DROP') == '1' else None


def poll(stop, box):
    while not stop.is_set():
        try: box[0] = max(box[0], int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.split()[0]))
        except Exception: pass
        stop.wait(0.5)


for fp in frames:
    frame = int(os.path.basename(fp).split('_')[1].split('.')[0]); jd = T + f'jobs/{OUT}/job_{frame}/'; shutil.rmtree(jd, ignore_errors=True)
    os.makedirs(jd + 'input/images/corneal-specular-microscopy-image'); os.makedirs(jd + 'output')
    json.dump([{'socket': {'slug': 'corneal-specular-microscopy-image'}, 'image': {'name': f'{frame}.mha'}}], open(jd + 'input/inputs.json', 'w'))
    shutil.copy(fp, jd + f'input/images/corneal-specular-microscopy-image/{frame}.mha')
    stop, box = threading.Event(), [0]; th = threading.Thread(target=poll, args=(stop, box), daemon=True); th.start(); t = time.time()
    try:
        p = subprocess.run([PY, WRAP], env=dict(base, GC_INPUT=jd + 'input', GC_OUTPUT=jd + 'output'), capture_output=True, text=True, timeout=TO); rc, so, se = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc, so, se = 'timeout', (e.stdout or b'').decode() if isinstance(e.stdout, bytes) else (e.stdout or ''), ''
    wall = time.time() - t; stop.set(); th.join()
    info = next((json.loads(l) for l in reversed(so.splitlines()) if l.startswith('{')), {})
    st = next((json.loads(l[7:]) for l in reversed(so.splitlines()) if l.startswith('STAGES ')), None)
    res.append(dict(frame=frame, mode=MODE, rc=rc, wall_s=round(wall, 1), entry_seconds=info.get('seconds'), setup_s=info.get('setup_s'), entry_mode=info.get('mode'),
                    errors=info.get('errors'), output=info.get('output'), gpu=info.get('gpu'), gpu_alloc_peak_gb=info.get('gpu_peak_gb'), gpu_device_peak_mib=box[0],
                    ram_peak_gb=info.get('ram_peak_gb'), hooks=info.get('hooks'), stage_s=st, threads=TH, drop_caches=drop if not res else None,
                    stderr_tail=se[-600:] if rc != 0 else ''))
    print(json.dumps({k: res[-1][k] for k in ('frame', 'mode', 'rc', 'wall_s', 'entry_mode', 'output', 'gpu_alloc_peak_gb', 'gpu_device_peak_mib', 'ram_peak_gb')}), flush=True)
    json.dump(dict(mode=MODE, env={k: v for k, v in base.items() if k.startswith(('CLEAR_EC_', 'OMP'))}, jobs=res), open(T + f'out/{OUT}.json', 'w'), indent=1)
print('DONE', OUT, flush=True)
