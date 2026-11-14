"""How much does a trained recipe A member move under perturbations inside the real range, against the out-of-range blur and noise of the earlier response test? (lane GPU 0, 2026-11-06)
Trained reference: fold 0 seed 46, terminal epoch. 240 validation frames. Reads median log change per metric, and MAPE on the same frames. Inference only."""
import sys, types
import numpy as np, torch, cv2
sys.path[:0] = ['experiments/lane_gpu0_20261106/source', 'experiments/meta_ceiling_20261103/source', 'experiments/phase2_pooling_20261020/source']
import phase2_pooling_regression_20261019 as R
import recipeA_measured_aug2_20261106 as W
b = R.b
RT = 'experiments/phase2_pooling_replication_20261026/f0_s46_std'
R.ARGS = types.SimpleNamespace(pooling='std')
H, W_ = 972, 1296
import json
m = json.load(open(RT + '/manifest.json'))
net, _ = R.build_cpu(46)
ck = torch.load(RT + '/runs/fold0/cpdino_vitb/epoch_20.pt', map_location='cpu', weights_only=True)
net.load_state_dict(ck['model']); net = net.cuda().eval(); del ck
mu = torch.tensor(m['log_mean']).cuda(); sg = torch.tensor(m['log_std']).cuda()
with np.load(b.META, allow_pickle=True) as z:
    ids, y = z['ids'].astype(str), z['Y'].astype(np.float64)
va = np.array(m['validation_indices']); tr = np.array(m['train_indices'])
pick = np.sort(np.random.default_rng(7).choice(va, 240, replace=False))
raw = np.memmap(b.RAW, mode='r', dtype=np.uint8, shape=(len(ids), H, W_))
s = W.stats()
by = {v: np.sort(tr[s['src'][tr] == v]) for v in (0, 1)}
cv2.setNumThreads(2)
p99_sig = {v: float(np.percentile(s['sigma'][s['src'] == v], 99)) for v in (0, 1)}


def norm(a):
    a = cv2.createCLAHE(2.0, (W_ // 32, H // 32)).apply(a)
    lo, hi = np.percentile(a, (1, 99))
    return np.clip((a.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)[None]


def u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def defocus(a, i, sb, rng):
    x = cv2.GaussianBlur(a.astype(np.float32), (0, 0), sb)
    k = cv2.getGaussianKernel(int(2 * np.ceil(4 * sb) + 1), sb); f = float((k ** 2).sum())
    return u8(x + rng.standard_normal(a.shape, dtype=np.float32) * float(s['sigma'][i]) * np.sqrt(max(0, 1 - f * f)))


def gain_only(a, g):
    return u8(a.astype(np.float32) * g)


VAR = {
    'none': lambda a, i, rng: a,
    'blur 1 no refill (earlier test)': lambda a, i, rng: u8(cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1.0)),
    'blur 2.5 no refill (earlier test)': lambda a, i, rng: u8(cv2.GaussianBlur(a.astype(np.float32), (0, 0), 2.5)),
    'defocus 1 refill': lambda a, i, rng: defocus(a, i, 1.0, rng),
    'defocus 1.5 refill': lambda a, i, rng: defocus(a, i, 1.5, rng),
    'noise 6 (earlier test)': lambda a, i, rng: u8(a.astype(np.float32) + rng.standard_normal(a.shape, dtype=np.float32) * 6),
    'noise up to source p99': lambda a, i, rng: W.transfer(a, i, int(by[int(s['src'][i])][np.argmax(s['sigma'][by[int(s['src'][i])]])]), int(s['src'][i]), rng, ('noise',)),
    'gain 0.7': lambda a, i, rng: gain_only(a, 0.7),
    'gain 1.3': lambda a, i, rng: gain_only(a, 1.3),
    'measured transfer, random donor': lambda a, i, rng: W.transfer(a, i, int(rng.choice(by[int(s['src'][i])])), int(s['src'][i]), rng),
}
PRED = {}
for name, fn in VAR.items():
    out = []; batch = []
    for i in pick:
        rng = np.random.default_rng(int(i) + 1000)
        batch.append(norm(fn(np.asarray(raw[i]), int(i), rng)))
        if len(batch) == 8 or i == pick[-1]:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                z = net(b.inputs(torch.from_numpy(np.stack(batch)), R.ARCH)).float()
            out.append(torch.exp(z * sg + mu).cpu().numpy()); batch = []
    PRED[name] = np.concatenate(out)
    d = np.log(PRED[name]) - np.log(PRED['none'])
    e = (np.abs(PRED[name] - y[pick]) / y[pick]).mean(0) * 100
    print(name.ljust(34), 'median log change % CD CV HEX', np.round(np.median(d, 0) * 100, 2).tolist(), 'abs', np.round(np.median(np.abs(d), 0) * 100, 2).tolist(), 'MAPE', np.round(e, 3).tolist(), 'mean', round(float(e.mean()), 3), flush=True)
