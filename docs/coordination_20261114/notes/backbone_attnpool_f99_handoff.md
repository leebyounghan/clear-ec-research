# Attention-pool DINOv3-B full-data build (fold 99, seed 46): VERIFIED 2026-11-11 10:52 EST (backbone lane)

Root: `experiments/lane_backbone_20261108/runs/attnpool_f99_dinov3b16_s46/raw/runs/fold99/dinov3_b16/`
Build: bb_runner_attnpool_20261111.py --family dinov3_b16 --fold 99 --seed 46 --attn-pool --epochs 12 --micro 8 --ema 0.998,0.999 --swa-start 6 --tta-final --keep-epochs 8-12; split = CD lane fulldata_regression_split.npz (train 8952, the two HEX-zero rows held); SSL = lane_cd arms/dinov3_b16_f99_s46/ssl/last.pt. Finished 15:51:34 UTC (2602 s).

HEX column read (as confirmed by the HEX lane): raw track, epochs 8, 9, 10, 11, 12, single view, no TTA, mean of log(pred[:, 2]).
Raw checkpoints (epoch 12 is final.pt; the runner names the last epoch final.pt):

| file | epoch | sha256 | bytes |
|---|---|---|---|
| epoch_08.pt | 8 | 069331d2fa0d34633aef0b4f35f7f920d922eb01d89054d7bb7087408f7f1848 | 353781413 |
| epoch_09.pt | 9 | f87bb3fb6286abbd34627b8785c3b261f56b2fb567a2052ab958b51a3a3a8e64 | 353781413 |
| epoch_10.pt | 10 | 79c8e4aa7dce9275e6e3eacf81998402b7b8aed1008f3de4946647375a72c664 | 353781413 |
| epoch_11.pt | 11 | 36fde73696242bb60343370031111ff55699c1bdfbef412925261e7bb8d5acd6 | 353781413 |
| final.pt | 12 | d29631486c0311a42806b7a04e32bf738b141e54e029763a2f9e5248edfec51a | 353768179 |

Other tracks (final.pt, epoch 12): ema998 b2eb9622bf02a61fff49c7f36dd0de0e6fad70f2dec7d3c0f50c84b5e4a87d5c, ema999 3949468b7a359291876c9cdf8f6b7f5282d42506c5563822b970b4de298c5184, swa6 8f07d91976ab010d58b68cefcbcfd22d6ccb4a23d466aefc057a13794b0c8e01 (under `.../attnpool_f99_dinov3b16_s46/<track>/runs/fold99/dinov3_b16/`).

Verification (`source/verify_attnpool_f99_20261111.py`, `reads/verify_attnpool_f99_20261111.json`): every checkpoint loads strict=True into the runner's own AttnPool network; recomputed predictions on the two held rows equal the stored ones (max relative difference 0.0 for epochs 8-11 and final single view, and for final four-flip TTA on all four tracks).
Sanity on 256 fixed training rows (seed 1111), f99 (in-sample) vs the attention-pool OOF members (each row from the fold run that held it out), same read: r 0.9865 / 0.9852 / 0.9856 (CD / CV / HEX); mean log ratio f99 over OOF +0.0069 / +0.0114 / -0.0087; r with the label f99 0.894 / 0.710 / 0.627 vs OOF 0.869 / 0.675 / 0.600 (in-sample, as expected).

Inference call (the runner's own network and evaluate(); family_infer's generic build does not create the AttnPool head):
```python
import sys, argparse, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
sys.path.insert(0, 'experiments/lane_backbone_20261108/source')   # run from /workspace/clear-ec-research
import bb_runner_attnpool_20261111 as B
F, R = B.F, B.R; B.FAMILY = 'dinov3_b16'; R.ARGS = argparse.Namespace(seed=46)
net = F.build_net(argparse.Namespace(family='dinov3_b16'), 46, 'experiments/lane_cd_20261108/arms/dinov3_b16_f99_s46/ssl/last.pt')
dim = net[1].normalized_shape[0]; torch.manual_seed(48)
net[0] = B.AttnPool(net[0].tokens, dim).cuda(); net[1] = nn.LayerNorm(4 * dim).cuda(); net[2] = nn.Linear(4 * dim, 128).cuda()
D = 'experiments/lane_backbone_20261108/runs/attnpool_f99_dinov3b16_s46/raw/runs/fold99/dinov3_b16/'
ids, y, tr, va = B.arrays(99); n = len(ids)            # rows = cache indices to score
dl = DataLoader(R.Frames(rows, y, n, 0, False, 46), batch_size=8, shuffle=False, num_workers=2)
logs = []
for name in ['epoch_08.pt', 'epoch_09.pt', 'epoch_10.pt', 'epoch_11.pt', 'final.pt']:
    ck = torch.load(D + name, map_location='cuda', weights_only=False); net.load_state_dict(ck['model'], strict=True)
    mu = torch.tensor(ck['log_mean'], device='cuda'); sigma = torch.tensor(ck['log_std'], device='cuda')
    logs.append(np.log(B.evaluate(net, dl, mu, sigma, rows, tta=False)))   # (len(rows), 3): CD, CV, HEX
hex_column = np.mean(logs, 0)[:, 2]
```
For test images outside the cache, the same preprocessing applies (R.Frames: CLAHE + 1-99 percentile on the 972 x 1296 frame); the evaluate() call handles padding and normalisation.
