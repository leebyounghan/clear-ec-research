"""Zero-shot read of a synthetic-pretrained DINOv3-B (backbone lane, 2026-11-11; coordinator): the synthpre3 fold-0 model (fold-0 SSL5,
then synthetic regression; never saw a real label) predicts (CD, area CV, HEX) on all real frames of the cache with recipe A
preprocessing (CLAHE + percentile, no flips, no TTA). Writes data/synth_zeroshot_f0pre_20261111.npz {ids, idx, pred (N, 3): CD, area CV,
HEX in synthetic units}. Run from /workspace/clear-ec-research with one visible GPU."""
import argparse, sys
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, 'experiments/lane_backbone_20261108/source')
import family_bb_20261108 as F  # noqa: E402
R, b = F.R, F.b
CK = 'experiments/lane_backbone_20261108/runs/synthpre3_f0_dinov3b16_s46/pretrain.pt'
OUT = 'experiments/lane_backbone_20261108/data/synth_zeroshot_f0pre_20261111.npz'
ck = torch.load(CK, map_location='cpu', weights_only=False)
net = F.build_net(argparse.Namespace(family='dinov3_b16'), 46, None)
net[0].tokens.load_state_dict(ck['tokens'], strict=True); net[1:].load_state_dict(ck['head'], strict=True); net.eval()
mu = torch.tensor(ck['log_mean'], device='cuda', dtype=torch.float32); sigma = torch.tensor(ck['log_std'], device='cuda', dtype=torch.float32)
with np.load(b.META, allow_pickle=True) as z:
    ids, Y = z['ids'].astype(str), z['Y'].astype(np.float32)
n = len(ids); rows = np.arange(n)
R.ARGS = argparse.Namespace(seed=46)
dl = DataLoader(R.Frames(rows, np.nan_to_num(Y), n, 0, False, 46), batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
P = []
with torch.inference_mode():
    for x, _, ix in dl:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z = net(F.inputs(x, 'dinov3_b16')).float()[:, :3]
        P.append(torch.exp(z * sigma + mu).cpu().numpy())
P = np.concatenate(P).astype(np.float32)
np.savez_compressed(OUT, ids=ids, idx=rows, pred=P, columns=np.array(['CD_synth', 'CVarea_synth', 'HEX_synth']), model=CK)
print('saved', OUT, P.shape, np.percentile(P, [1, 50, 99], axis=0).round(3).tolist(), flush=True)
