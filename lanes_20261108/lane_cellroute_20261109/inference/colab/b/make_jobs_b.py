"""coordinator 2026-11-13 (D-PKG-007): derive colab_t4_jobs_b.py from the deploy lane's colab_t4_jobs.py for the tier-p package on Colab session
b9204d: the fixed image environment (PYTHONPATH, allocator setting, CLEAR_EC_UNLOAD=1: Dockerfile.freeze ENV), the staged pack_hooks.py as is
(no swap: the p context holds the current one), and narrow mixed-precision modes with one detector copy."""
import sys
C = '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference/colab/'
s = open(C + 'colab_t4_jobs.py').read()
def rep(a, b):
    global s
    assert a in s, a[:60]; s = s.replace(a, b)
rep("k not in ('PYTHONPATH',)", "k not in ()")
rep("base.update(PYTHONUNBUFFERED='1',", "base.update(PYTHONPATH='/workspace/clear-ec-research/experiments/phase2_foundation_20261002/runtime', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', CLEAR_EC_UNLOAD='1', PYTHONUNBUFFERED='1',")
rep("shutil.copy(APP + '/pack_hooks_staged.py' if MODE == 'default' else T + 'pack_hooks_new.py', APP + '/pack_hooks.py')\n", "")
rep("fp32=dict(CLEAR_EC_PREC='fp32'),", "fp32=dict(CLEAR_EC_PREC='fp32'), mixhex6=dict(CLEAR_EC_PREC='mix', CLEAR_EC_FP32_MODULES='hex6_correction', CLEAR_EC_DET_COPIES='1'), mixkonan=dict(CLEAR_EC_PREC='mix', CLEAR_EC_FP32_MODULES='deploy_hexkonan_image', CLEAR_EC_DET_COPIES='1'),")
open(C + 'b/colab_t4_jobs_b.py', 'w').write(s); print('ok', len(s))
