#!/usr/bin/env python3
"""CV lane copy of the CD lane TTA script (writes into lane_cv_20261108/out/member_tta). Four-flip TTA over the kept epochs 16-20 of a finished family run (CD lane, 2026-11-08). Inference only, validation rows of that fold.
Reads the run (any lane, read-only); writes only into experiments/lane_cd_20261108/tta/: OUTNAME_tta4_e16_20.npz (indices, ids, target,
pred = exp of the mean log prediction over 5 epochs x 4 flips) and OUTNAME_tta4_epoch_XX.npz. One loader worker.
usage (from /workspace/clear-ec-research, one visible GPU): family_tta_20261108.py RUNROOT FAMILY FOLD SEED OUTNAME"""
import sys, argparse
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, 'experiments/lane_cv_20261108/source')
import family_cv_20261108 as Fm
root, fam, fold, seed, outname = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
OUT = Path('experiments/lane_cv_20261108/out/member_tta'); OUT.mkdir(exist_ok=True)
run = root / 'runs' / f'fold{fold}' / fam
assert (run / 'complete.json').exists(), run
out_all = OUT / f'{outname}_tta4_e16_20.npz'
if out_all.exists(): print('exists', out_all); sys.exit(0)
ids, y, tr, va = Fm.arrays(fold); n = len(ids)
a = argparse.Namespace(family=fam, micro=0)
Fm.R.ARGS = argparse.Namespace(seed=seed)
micro = Fm.FAMILIES[fam]['micro']
dl = DataLoader(Fm.R.Frames(va, y, n, 0, False, seed), batch_size=micro, shuffle=False, num_workers=1, pin_memory=True, prefetch_factor=4)
logs = []
for e in range(16, 21):
    ck = torch.load(run / f'epoch_{e:02d}.pt', map_location='cpu', weights_only=True)
    net = Fm.build_net(a, seed, None); net.load_state_dict(ck['model'], strict=True); net.eval()
    mu = torch.tensor(ck['log_mean'], device='cuda', dtype=torch.float32); sigma = torch.tensor(ck['log_std'], device='cuda', dtype=torch.float32)
    preds, seen = [], []
    with torch.inference_mode():
        for x, _, ix in dl:
            zs = []
            for hf in (False, True):
                for vf in (False, True):
                    xx = x
                    if hf: xx = torch.flip(xx, (-1,))
                    if vf: xx = torch.flip(xx, (-2,))
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        zs.append(net(Fm.inputs(xx, fam)).float())
            z = torch.stack(zs).mean(0); preds.append((z * sigma + mu).cpu().numpy()); seen.append(ix.numpy())
    np.testing.assert_array_equal(np.concatenate(seen), va)
    lp = np.concatenate(preds); logs.append(lp)
    np.savez_compressed(OUT / f'{outname}_tta4_epoch_{e:02d}.npz', indices=va, ids=ids[va], target=y[va], pred=np.exp(lp))
    print('epoch', e, Fm.safe_metrics(np.exp(lp), y[va]), flush=True)
    del net; torch.cuda.empty_cache()
P = np.exp(np.mean(logs, 0)); np.savez_compressed(out_all, indices=va, ids=ids[va], target=y[va], pred=P)
print('tta4 e16-20', Fm.safe_metrics(P, y[va]), flush=True)
