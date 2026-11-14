"""Vigueras-Guillen TVST CNN-Edge + CNN-ROI on all 8,954 full frames (HEX lane, 2026-11-11). Frames are resized to the
authors' pixel size (972 x 1296 at 0.7716 um/px -> 720 x 960 at 1.042 um/px; their Topcon images are 1.038 um/px, INTER_AREA),
scaled by 1/255 as in their notebook, and run through CNN-Edge; CNN-ROI takes the float edge map. Output per chunk of 64
frames (in index order): OUT/cNNN.npz with frames, edge and roi (uint8, round(prob * 255), as the authors save them).
fp32 throughout (TF32 off, as in the port check). Resumable. usage: vig_infer.py OUT_DIR [lo hi]"""
import os, sys, time
import numpy as np
import cv2
import torch
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
sys.path.insert(0, LANE + '/source')
import vig_net as V

if __name__ == '__main__':
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); NF, CH = 8954, 64; nch = (NF + CH - 1) // CH
    lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0; hi = int(sys.argv[3]) if len(sys.argv) > 3 else nch
    MM = np.memmap('/workspace/clear-ec-research/datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(NF, 972, 1296))
    We, Wr = V.load('tvst_edge'), V.load('tvst_roi'); t0 = time.time(); nd = 0
    for c in range(lo, hi):
        path = os.path.join(OUT, f'c{c:03d}.npz')
        if os.path.exists(path): continue
        fr = np.arange(c * CH, min(NF, (c + 1) * CH))
        X = np.stack([cv2.resize(np.asarray(MM[i]), (960, 720), interpolation=cv2.INTER_AREA) for i in fr]).astype(np.float32) / 255
        E = V.run(We, X, batch=8); Rr = V.run(Wr, E, batch=8)
        tmp = path + '.tmp.npz'
        np.savez_compressed(tmp, frames=fr, edge=np.round(E * 255).astype(np.uint8), roi=np.round(Rr * 255).astype(np.uint8)); os.replace(tmp, path)
        nd += len(fr); print(f'chunk {c} done, {nd} frames, {(time.time() - t0) / nd:.3f} s/frame', flush=True)
    print('done', flush=True)
