"""Cellpose-SAM settings screen on a few frames (HEX lane, 2026-11-08): coverage and interior cells per setting."""
import sys, os, time
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_hex_20261108/source')
os.chdir('/workspace/clear-ec-research')
import cv2
from cellpose import models
FR = [int(q) for q in sys.argv[1].split(',')]; sys.argv = ['seg_hex', 'unused.npz', '0']
import seg_hex as S   # reuses feats() and the image memmap
model = models.CellposeModel(gpu=True, pretrained_model='cpsam_models/cpsam_v2')
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
SET = {'default': dict(), 'loose': dict(flow_threshold=0.8, cellprob_threshold=-2.0), 'looser': dict(flow_threshold=1.2, cellprob_threshold=-4.0)}
for i in FR:
    img = np.asarray(S.MM[i]); ci = clahe.apply(img)
    for nm, kw in SET.items():
        for tag, im in (('raw', img), ('clahe', ci)):
            if nm == 'looser' and tag == 'raw': continue
            t = time.time(); M = np.asarray(model.eval([im], batch_size=8, **kw)[0][0], np.int32); f = S.feats(M)
            print(i, nm, tag, 'h6 %.3f n_int %d cov %.3f cells %d  %.0fs' % (f[0], f[3] if np.isfinite(f[3]) else -1, f[6], f[7] if np.isfinite(f[7]) else -1, time.time() - t), flush=True)
print('done', flush=True)
