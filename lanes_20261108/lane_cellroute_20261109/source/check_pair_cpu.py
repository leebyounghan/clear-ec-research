"""Cell-route lane (2026-11-10): which sg2pre_s47/s48 pair? Cell-route features from the full-data detectors with the green-track pair (10-30) and
with the cell-route duplicate (11-09), sg2pre_s46 shared, on 4 random frames (CPU), against each other and against the OOF features."""
import sys, json
import numpy as np, torch
torch.set_num_threads(16)
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/source')
import deploy_hexkonan_image as DI, deploy_hexkonan as D
R = '/workspace/clear-ec-research/'; GT = R + 'experiments/green_track_20261028/deploy/detectors/'; CR = R + 'experiments/lane_cellroute_20261109/deploy/detectors/'
MM = np.memmap(R + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
zs = np.load(R + 'experiments/lane_cellroute_20261109/feats/seg_centroids.npz'); _p, _o, _i = zs['pts'], zs['offsets'], zs['indices']; pos = {int(d): k for k, d in enumerate(_i)}
zf = np.load(R + 'experiments/lane_cellroute_20261109/feats/cellroute_feats.npz'); NM = list(zf['names']); FF = zf['F'][:, [NM.index(c) for c in D.NAMES]]
P = R + 'experiments/lane_cellroute_20261109/out/deploy_hexkonan_params.json'
hg = DI.HexKonan([GT + 'sg2pre_s46.pt', GT + 'sg2pre_s47.pt', GT + 'sg2pre_s48.pt'], P, device='cpu'); hc = DI.HexKonan([GT + 'sg2pre_s46.pt', CR + 'sg2pre_s47.pt', CR + 'sg2pre_s48.pt'], P, device='cpu')
out = []
for r in np.random.default_rng(3).choice(8954, 4, replace=False):
    seg = _p[_o[pos[int(r)]]:_o[pos[int(r)] + 1]]; img = np.asarray(MM[r])
    fg = D.features(*(lambda q: (q[0], q[1][0], q[1][1], q[1][2]))(hg.peaks(img)), seg); fc = D.features(*(lambda q: (q[0], q[1][0], q[1][1], q[1][2]))(hc.peaks(img)), seg)
    rel = lambda a, b: float(np.nanmax(np.abs(a / b - 1)))
    o = dict(frame=int(r), green_vs_cellroute=rel(fg, fc), green_vs_oof=rel(fg, FF[r]), cellroute_vs_oof=rel(fc, FF[r])); out.append(o); print(json.dumps(o), flush=True)
print('median over frames:', json.dumps({k: float(np.median([o[k] for o in out])) for k in out[0] if k != 'frame'}))
