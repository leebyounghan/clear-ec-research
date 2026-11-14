import subprocess
code = r'''
import sys, os, traceback
os.chdir("/workspace/clear-ec-research"); sys.path.insert(0, "experiments/phase2_foundation_20261002/runtime")
try:
    import dinov3.hub.backbones as b; print("OK", b.__file__)
except Exception:
    traceback.print_exc()
'''
r = subprocess.run(['/content/venv/bin/python', '-c', code], capture_output=True, text=True); print(r.stdout[-3000:], r.stderr[-3000:])
