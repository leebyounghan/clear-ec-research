"""CPU check of the image path with fold-0 pre-only detectors on 2 fold-0 frames: peaks vs the stored TTA peaks, features vs cellroute_feats."""
import sys, time
import numpy as np
import torch
torch.set_num_threads(16)
torch.Tensor.cuda = lambda self, *a, **k: self  # CPU-only test pod: the shared input helper calls .cuda(); deploy code is unchanged
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/source'); sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import deploy_hexkonan_image as DI
import deploy_hexkonan as D
from scipy.optimize import linear_sum_assignment
ROOT = '/workspace/clear-ec-research/'; GT = ROOT + 'experiments/green_track_20261028/external_srep/'
MM = np.memmap(ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
hk = DI.HexKonan([GT + 'fold0/sg2_pre.pt', GT + 's47/fold0/sg2_pre.pt', GT + 's48/fold0/sg2_pre.pt'], ROOT + 'experiments/lane_cellroute_20261109/out/deploy_hexkonan_params.json', device='cpu')
def stored(tag, r):
    z = np.load(ROOT + f'experiments/lane_cv_20261108/out/tta_peaks/{tag}/fold0/fold0_sg2_peaks.npz', allow_pickle=True); ix = list(z['indices']); k = ix.index(r); return z['peaks'][z['offsets'][k]:z['offsets'][k + 1]].astype(float), ix
_, ix = stored('s46_tta4', 0); zs = np.load(ROOT + 'experiments/lane_cellroute_20261109/feats/seg_centroids.npz'); _p, _o, _i = zs['pts'], zs['offsets'], zs['indices']
zf = np.load(ROOT + 'experiments/lane_cellroute_20261109/feats/cellroute_feats.npz'); NM = list(zf['names']); FF = zf['F'][:, [NM.index(c) for c in D.NAMES]]
for r in ix[:2]:
    t = time.time(); pk3, pk = hk.peaks(MM[r]); dt = time.time() - t
    for name, mine, tag in (('s46', pk[0], 's46_tta4'), ('s47', pk[1], 's47_tta4'), ('s48', pk[2], 's48_tta4'), ('mean3', pk3, 'seeds3_tta4')):
        ref, _ = stored(tag, r); a, b = mine[mine[:, 2] >= 0.35, :2], ref[ref[:, 2] >= 0.35, :2]
        Dm = np.linalg.norm(a[:, None] - b[None], axis=2); D2 = np.where(Dm <= 2, Dm, 1e6); u, v = linear_sum_assignment(D2); ok = D2[u, v] < 1e5
        print(f'frame {r} {name}: peaks>=0.35 mine {len(a)} stored {len(b)}; matched within 2 px {int(ok.sum())}; mean |d| {Dm[u[ok], v[ok]].mean():.3f} px', flush=True)
    k = list(_i).index(r); F = D.features(pk3, pk[0], pk[1], pk[2], _p[_o[k]:_o[k + 1]])
    print(f'frame {r}: {dt:.0f} s CPU for 12 forwards; features mine {np.round(F, 4).tolist()}\n          stored {np.round(FF[r], 4).tolist()}', flush=True)
