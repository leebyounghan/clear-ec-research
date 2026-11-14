"""Image-level inference for models of the lane GPU 3 family runner (HEX lane, 2026-11-10), e.g. the full-data CV-weighted
DINOv3 (HEX chain step cvw3_fix) and the CV lane's DINOv3 members. Same path as the runner's validation read: R.Frames on the raw
frame (CLAHE, 1-99 % scaling, family crop/pad), net(inputs(x)), exp(z * sigma + mu) with mu/sigma = the checkpoint's training
log mean / std. Output: per image the mean over the given checkpoints of log prediction (N x 3: CD, CV, HEX), as the chain reads
members (exp of the mean log over epochs).
usage: family_infer.py --ckpts 'DIR/epoch_{08..12}.pt' --images X.dat --n M --out OUT.npz
       family_infer.py --verify RUN_DIR --epochs 16,17,18,19,20 --k 16     (compares with the run's stored validation predictions)"""
import os, sys, glob, argparse, json
from pathlib import Path
import numpy as np
os.chdir('/workspace/clear-ec-research'); sys.path.insert(0, 'experiments/lane_hex_20261108/source')
import torch
from torch.utils.data import DataLoader
import family_cvw3_copy as FR


def predict(ckpts, raw, rows, total, micro=8):
    FR.b.RAW = Path(raw); FR.R.ARGS = argparse.Namespace(seed=46); out = []
    for c in ckpts:
        ck = torch.load(c, map_location='cpu', weights_only=False); a = argparse.Namespace(family=ck['family'])
        net = FR.build_net(a, ck.get('seed', 46), None); net.load_state_dict(ck['model'], strict=True)
        mu = torch.tensor(ck['log_mean'], device='cuda', dtype=torch.float32); sg = torch.tensor(ck['log_std'], device='cuda', dtype=torch.float32)
        dl = DataLoader(FR.R.Frames(np.asarray(rows), np.ones((total, 3), np.float32), total, 0, False, ck.get('seed', 46)), batch_size=micro, shuffle=False, num_workers=3, pin_memory=True)
        out.append(np.log(FR.evaluate(net, ck['family'], dl, mu, sg, np.asarray(rows)))); del net; torch.cuda.empty_cache(); print('ckpt', c, flush=True)
    return np.mean(out, 0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--ckpts'); ap.add_argument('--images'); ap.add_argument('--n', type=int); ap.add_argument('--out')
    ap.add_argument('--verify'); ap.add_argument('--epochs', default='16,17,18,19,20'); ap.add_argument('--k', type=int, default=16); a = ap.parse_args()
    if a.verify:
        eps = [int(e) for e in a.epochs.split(',')]; d = Path(a.verify); z0 = np.load(d / f'predictions_epoch_{eps[0]:02d}.npz'); va = z0['indices']
        rows = np.sort(np.random.default_rng(0).choice(va, a.k, replace=False)); total = 8954
        P = predict([str(d / f'epoch_{e:02d}.pt') for e in eps], 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', rows, total)
        ref = np.mean([np.log(np.load(d / f'predictions_epoch_{e:02d}.npz')['pred']) for e in eps], 0); pos = {int(i): k for k, i in enumerate(va)}; R_ = ref[[pos[int(r)] for r in rows]]
        rep = dict(rows=int(a.k), max_abs_log_diff=float(np.abs(P - R_).max()), per_column=np.abs(P - R_).max(0).tolist()); print(json.dumps(rep), flush=True)
        json.dump(rep, open('experiments/lane_hex_20261108/logs/family_infer_verify.json', 'w'))
    else:
        ck = sorted(glob.glob(a.ckpts)) if '*' in a.ckpts else a.ckpts.split(','); P = predict(ck, a.images, np.arange(a.n), a.n)
        np.savez_compressed(a.out, logpred=P.astype(np.float32), ckpts=np.array(ck)); print('done', a.out, flush=True)
