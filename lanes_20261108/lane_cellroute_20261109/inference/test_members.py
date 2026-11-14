"""Member parity on fold-0 validation frames (CPU): image -> member log predictions with fold-0 checkpoints vs the stored per-checkpoint predictions.
Test harness only (reads stored prediction files to compare; the inference modules never do)."""
import sys, time, json
import numpy as np, torch
torch.set_num_threads(16)
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference')
import members as M
E = 'experiments/'; MM = np.memmap(M.ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
RA46 = E + 'phase2_pooling_replication_20261026/f0_s46_std/runs/fold0/cpdino_vitb/'
DV46 = E + 'lane_gpu3_20261106/runs/dinov3_b16_f0_s46/runs/fold0/dinov3_b16/'
VL46 = E + 'lane_backbone_20261108/runs/vitl16_avg_f0_s46/raw/runs/fold0/dinov3_l16t/'
def stored(path, key='pred'):
    z = np.load(path); return {int(i): v for i, v in zip(z['indices'], z[key])}
st = {'A_tta_e20': stored(RA46 + 'predictions_tta4.npz', 'pred_tta4'), 'A_e08': stored(RA46 + 'predictions_epoch_08.npz'), 'D_e16': stored(DV46 + 'predictions_epoch_16.npz'), 'V_tta_e12': stored(VL46 + 'predictions_final_tta.npz')}
frames = sorted(st['A_tta_e20'])[:: max(1, len(st['A_tta_e20']) // 3)][:3]; out = []
for r in frames:
    x = M.preprocess(MM[r]); t0 = time.time()
    got = {'A_tta_e20': np.exp(M.read_logs(x, [RA46 + 'epoch_20.pt'], 'cpdino_recipeA', True)[0]), 'A_e08': np.exp(M.read_logs(x, [RA46 + 'epoch_08.pt'], 'cpdino_recipeA', False)[0]),
           'D_e16': np.exp(M.read_logs(x, [DV46 + 'epoch_16.pt'], 'dinov3_b16', False)[0]), 'V_tta_e12': np.exp(M.read_logs(x, [VL46 + 'final.pt'], 'dinov3_l16t', True)[0])}
    row = {k: np.round(np.abs(got[k] / st[k][r] - 1), 5).tolist() for k in got}; row['frame'] = int(r); row['seconds'] = round(time.time() - t0, 1); out.append(row)
    print(json.dumps(row), flush=True)
print('max rel diff per member (CD, CV, HEX):', json.dumps({k: np.max([o[k] for o in out], 0).tolist() for k in st}), flush=True)
