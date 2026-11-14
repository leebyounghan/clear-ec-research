import sys, json, numpy as np, torch
torch.set_num_threads(16)
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/source')
import deploy_hexkonan_image as DI, deploy_hexkonan as D
R = '/workspace/clear-ec-research/'; GT = R + 'experiments/green_track_20261028/external_srep/'
MM = np.memmap(R + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
zs = np.load(R + 'experiments/lane_cellroute_20261109/feats/seg_centroids.npz'); _p, _o, _i = zs['pts'], zs['offsets'], zs['indices']; k = list(_i).index(6124)
zf = np.load(R + 'experiments/lane_cellroute_20261109/feats/cellroute_feats.npz'); NM = list(zf['names']); FF = zf['F'][:, [NM.index(c) for c in D.NAMES]]
hk = DI.HexKonan([GT + 'fold0/sg2_pre.pt', GT + 's47/fold0/sg2_pre.pt', GT + 's48/fold0/sg2_pre.pt'], R + 'experiments/lane_cellroute_20261109/out/deploy_hexkonan_params.json', device='cpu')
pk3, pk = hk.peaks(np.asarray(MM[6124])); F = D.features(pk3, pk[0], pk[1], pk[2], _p[_o[k]:_o[k + 1]])
for n, a, b in zip(D.NAMES, F, FF[6124]): print(f'{n:16s} cpu {a:.4f} stored {b:.4f} rel {abs(a / b - 1):.3f}')
hi = pk3[pk3[:, 2] >= 0.5, :2]; print('windows cpu', D.windows(hi, 380, 300), 'n peaks>=0.5', len(hi))
