import sys, json, numpy as np
sys.path.insert(0, "experiments/eval_harness_20261031/source")
import eval_oof_20261031 as E
from scipy.spatial import Delaunay
b = E.load_base(); pos = {int(i): k for k, i in enumerate(b["idx"])}; y = b["y"]
rows, dens, nnd = [], [], []
for f in range(5):
    z = np.load(f"experiments/green_track_20261028/external_srep/fold{f}/fold{f}_sg2_peaks.npz", allow_pickle=True)
    P, off, idx = z["peaks"], z["offsets"], z["indices"]
    for n, i in enumerate(idx):
        p = P[off[n]:off[n+1]]; p = p[p[:, 2] >= 0.35]
        if len(p) < 30: continue
        # central 600x800 window to avoid the vignette: density of peaks per px^2
        c = p[(np.abs(p[:, 0] - 486) < 300) & (np.abs(p[:, 1] - 648) < 400)] if p[:, 0].max() < 1000 else p
        if len(c) < 30: continue
        tri = Delaunay(c[:, :2]); e = np.concatenate([tri.simplices[:, [0, 1]], tri.simplices[:, [1, 2]], tri.simplices[:, [0, 2]]])
        L = np.linalg.norm(c[e[:, 0], :2] - c[e[:, 1], :2], axis=1)
        rows.append(pos[int(i)]); dens.append(len(c) / (600 * 800)); nnd.append(np.median(L))
rows = np.array(rows); dens = np.array(dens); nnd = np.array(nnd)
cd = y[rows, 0]; ok = cd > 0
# px^2 per cell implied by labels: 1e6 um^2/mm^2 / CD = um^2 per cell ; detected px^2 per cell = 1/dens
r = (1 / dens[ok]) / (1e6 / cd[ok])   # px^2 per um^2
print(json.dumps(dict(n=int(ok.sum()), px2_per_um2_median=float(np.median(r)), um_per_px=float(np.median(r) ** -0.5),
      iqr=[float(np.percentile(r, 25)), float(np.percentile(r, 75))], corr_logdens_logcd=float(np.corrcoef(np.log(dens[ok]), np.log(cd[ok]))[0, 1]),
      median_neighbor_px=float(np.median(nnd)), coords_max=[float(P[:, 0].max()), float(P[:, 1].max())])))
