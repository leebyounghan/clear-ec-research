"""port diagnostics vs the authors' example (HEX lane, 2026-11-11): variants of padding / BN statistics / output head, and
the alignment of the residual; nothing here is used for our frames"""
import sys, json
import numpy as np, torch, torch.nn.functional as F, cv2
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_hex_20261108/source'); import vig_net as V
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
im = cv2.imread(V.EX + 'Intn.tif', cv2.IMREAD_UNCHANGED).astype(np.float32) / 255; ref = cv2.imread(V.EX + 'NN_Edges.tif', cv2.IMREAD_UNCHANGED).astype(float) / 255
W = V.load('tvst_edge')
bn_layers = sorted({k.rsplit('.', 1)[0] for k in W if k.endswith('.moving_stddev')})
d = max(float((W[n + '.moving_variance'] - (W[n + '.moving_stddev'] ** 2 - 1.001e-5).clamp_min(0)).abs().max()) for n in bn_layers)
print('max |moving_variance - (moving_stddev^2 - eps)|', d)
d2 = max(float((W[n + '.renorm_mean'] - W[n + '.moving_mean']).abs().max()) for n in bn_layers); print('max |renorm_mean - moving_mean|', d2)
def score(P, tag):
    c = np.corrcoef(P.ravel(), ref.ravel())[0, 1]; best = max(((np.corrcoef(np.roll(P, (dy, dx), (0, 1))[4:-4, 4:-4].ravel(), ref[4:-4, 4:-4].ravel())[0, 1], dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)))
    print(f'{tag:28s} corr {c:.5f} maxabs {np.abs(P - ref).max() * 255:6.1f} mean {np.abs(P - ref).mean() * 255:6.2f}  best shift corr {best[0]:.5f} at {best[1:]}', flush=True)
x = torch.from_numpy(im[None, None]).cuda()
with torch.no_grad():
    score(V.forward(W, x)[0].cpu().numpy(), 'port')
    old = V._bn
    V._bn = lambda x, W, n, eps=1.001e-5: (x - W[n + '.moving_mean'][:, None, None]) / W[n + '.moving_stddev'][:, None, None] * W[n + '.gamma'][:, None, None] + W[n + '.beta'][:, None, None]
    score(V.forward(W, x)[0].cpu().numpy(), 'bn moving_stddev'); V._bn = old
    V._bn = lambda x, W, n, eps=1.001e-5: (x - W[n + '.renorm_mean'][:, None, None]) / W[n + '.renorm_stddev'][:, None, None] * W[n + '.gamma'][:, None, None] + W[n + '.beta'][:, None, None]
    score(V.forward(W, x)[0].cpu().numpy(), 'bn renorm stats'); V._bn = old
    padf = F.pad
    F.pad = lambda t, p, *a, **k: padf(t, (2, 1, 2, 1) if tuple(p) == (1, 2, 1, 2) else p, *a, **k)
    score(V.forward(W, x)[0].cpu().numpy(), 'pad 2-before'); F.pad = padf
    xt = torch.from_numpy(np.ascontiguousarray(im.T)[None, None]).cuda(); score(V.forward(W, xt)[0].cpu().numpy().T, 'transposed input')
    score(V.forward(W, torch.flip(x, (3,)))[0].flip(1).cpu().numpy(), 'flipped input (control)')
