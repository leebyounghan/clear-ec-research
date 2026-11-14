"""port diagnostics 2 (HEX lane, 2026-11-11): transposed-kernel flip, flip-TTA, linear fit, where the residual sits"""
import sys
import numpy as np, torch, torch.nn.functional as F, cv2
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_hex_20261108/source'); import vig_net as V
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
im = cv2.imread(V.EX + 'Intn.tif', cv2.IMREAD_UNCHANGED).astype(np.float32) / 255; ref = cv2.imread(V.EX + 'NN_Edges.tif', cv2.IMREAD_UNCHANGED).astype(float) / 255
W = V.load('tvst_edge'); x = torch.from_numpy(im[None, None]).cuda()
def score(P, tag): print(f'{tag:26s} corr {np.corrcoef(P.ravel(), ref.ravel())[0, 1]:.5f} maxabs {np.abs(P - ref).max() * 255:6.1f} mean {np.abs(P - ref).mean() * 255:6.2f}', flush=True)
with torch.no_grad():
    P = V.forward(W, x)[0].cpu().numpy(); score(P, 'port')
    ct = F.conv_transpose2d
    F.conv_transpose2d = lambda t, w, *a, **k: ct(t, torch.flip(w, (2, 3)), *a, **k); score(V.forward(W, x)[0].cpu().numpy(), 'flip transposed kernels'); F.conv_transpose2d = ct
    c2 = F.conv2d; pf0 = F.pad
    F.conv2d = lambda t, w, *a, **k: c2(t, torch.flip(w, (2, 3)) if w.shape[-1] == 4 else w, *a, **k)
    F.pad = (lambda pf: (lambda t, p, *a, **k: pf(t, (2, 1, 2, 1) if tuple(p) == (1, 2, 1, 2) else p, *a, **k)))(F.pad)
    score(V.forward(W, x)[0].cpu().numpy(), 'flip 4x4 kernels + pad'); F.conv2d = c2; F.pad = pf0
with torch.no_grad():
    tta = np.mean([V.forward(W, torch.flip(x, d) if d else x)[0].flip(tuple(i - 2 for i in d)).cpu().numpy() if d else V.forward(W, x)[0].cpu().numpy() for d in ((), (3,), (2,), (2, 3))], 0)
score(tta, 'flip TTA mean of 4')
a, b = np.polyfit(P.ravel(), ref.ravel(), 1); print('linear fit ref = a P + b:', round(a, 4), round(b, 4), 'ref max', ref.max(), 'P max', round(float(P.max()), 4), 'ref min', ref.min(), 'P min', round(float(P.min()), 6))
D = np.abs(P - ref); H, Wd = D.shape
for nm, m in (('outer 16 px', np.pad(np.zeros((H - 32, Wd - 32), bool), 16, constant_values=True)), ('inner', ~np.pad(np.zeros((H - 32, Wd - 32), bool), 16, constant_values=True))):
    print(nm, 'mean abs', round(float(D[m].mean() * 255), 2), 'frac > 25/255', round(float((D[m] > 0.1).mean()), 4))
R8 = cv2.imread(V.EX + 'NN_ROI.tif', cv2.IMREAD_UNCHANGED) >= 128
print('inside ROI mean abs', round(float(D[R8].mean() * 255), 2), 'outside', round(float(D[~R8].mean() * 255), 2))
print('ref histogram (8 bins)', np.histogram(ref, 8, (0, 1))[0].tolist(), 'port', np.histogram(P, 8, (0, 1))[0].tolist())
