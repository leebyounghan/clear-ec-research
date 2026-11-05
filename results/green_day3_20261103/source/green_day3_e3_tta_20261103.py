"""E3: four-flip TTA predictions of the box-pooling regressor on its held fold (GPU), in the recipe A predictions_tta4 format.
    CUDA_VISIBLE_DEVICES=k python ... --fold F
"""
import argparse, json, sys, time
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
ROOT = Path('experiments/green_day3_20261103/e3_boxpool')


def main(a):
    run_root = ROOT/f'f{a.fold}_s46_stdbox'
    sys.path.insert(0, str(run_root/'source'))
    import recipeA_box_pool_backbone_20261103 as W          # applies the std_box pooling, Frames box channel and inputs patches
    import phase2_foundation_preflight_20261002 as preflight
    import phase2_foundation_train_20261002 as b
    import phase2_pooling_20261019 as pooling
    sys.path.insert(0, 'experiments/green_track_20261028/source')
    import green_selftrain_20261028 as ST
    run = run_root/'runs'/f'fold{a.fold}'/'cpdino_vitb'
    ck = torch.load(run/'epoch_20.pt', map_location='cpu', weights_only=False)
    net, multiple, details = preflight.build('cpdino_vitb', False, native_convnext_norm=True)
    net[0] = pooling.SpatialPoolEncoder(net[0].original, 'std_box')
    net.load_state_dict(ck['model'], strict=True); net = net.cuda().eval()
    mu = np.array(ck['log_mean'], float); sigma = np.array(ck['log_std'], float)
    ann, binding, ids, y, tr, va = ST.green(a.fold)
    dl = DataLoader(b.Frames(va, y, len(ids)), batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    FLIPS = ((False, False), (True, False), (False, True), (True, True))
    zs, seen = [], []; t0 = time.time()
    with torch.inference_mode():
        for k, (x, t, ix) in enumerate(dl):
            zv = []
            for h, v in FLIPS:
                xf = x
                if h: xf = torch.flip(xf, (-1,))
                if v: xf = torch.flip(xf, (-2,))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z = net(b.inputs(xf, 'cpdino_vitb')).float()
                zv.append(z.cpu().numpy())
            zs.append(np.stack(zv, 1)); seen.append(ix.numpy())
            if k % 100 == 0:
                print(json.dumps(dict(msg='tta', fold=a.fold, batch=k, of=len(dl), seconds=round(time.time()-t0, 1))), flush=True)
    zs = np.concatenate(zs); seen = np.concatenate(seen); np.testing.assert_array_equal(seen, va)
    pred_tta4 = np.exp(zs.mean(1)*sigma+mu); pred_single = np.exp(zs[:, 0]*sigma+mu)
    with np.load(run/'predictions.npz') as z:
        np.testing.assert_array_equal(z['indices'], va); stored = z['pred'].astype(float)
    print('single-view vs audit predictions max rel diff: %.2e' % float((np.abs(stored-pred_single)/stored).max()), flush=True)
    np.savez_compressed(run/'predictions_tta4.npz', indices=va, ids=ids[va], target=y[va], z_views=zs.astype(np.float32), pred_single=pred_single, pred_tta4=pred_tta4, flips=np.array(FLIPS), log_mean=mu, log_std=sigma)
    print(json.dumps(dict(msg='tta done', fold=a.fold, n=int(len(va)), seconds=round(time.time()-t0, 1))), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--fold', type=int, required=True); main(ap.parse_args())
