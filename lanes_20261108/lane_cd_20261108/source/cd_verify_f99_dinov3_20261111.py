"""Verify the full-data DINOv3-B/16 members (fold 99, seeds 46 = arms/dinov3_b16_f99_s46b, 47 = arms/dinov3_b16_f99_s47): sha256 of
epoch_16..20.pt, strict load into the family runner's network, single-view prediction on the held rows (the two HEX-zero rows) vs the stored
predictions_epoch_XX.npz, and the deploy form (four-flip TTA per epoch, mean log over the 20 forwards) on the held rows as a reference value
for deploy parity, plus the two-seed mean. Run from /workspace/clear-ec-research with one visible GPU (inference only)."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, 'experiments/lane_cd_20261108/source')
import family_cd_20261108 as Fm  # noqa: E402
FAM = 'dinov3_b16'
ROOTS = {46: Path('experiments/lane_cd_20261108/arms/dinov3_b16_f99_s46b'), 47: Path('experiments/lane_cd_20261108/arms/dinov3_b16_f99_s47')}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


ids, y, tr, va = Fm.arrays(99); n = len(ids)
out = dict(held_rows=va.tolist(), held_ids=ids[va].tolist()); tta_log = {}
for s, root in ROOTS.items():
    run = root / 'runs' / 'fold99' / FAM
    assert (run / 'complete.json').exists(), run
    Fm.R.ARGS = argparse.Namespace(seed=s)
    dl = DataLoader(Fm.R.Frames(va, y, n, 0, False, s), batch_size=Fm.FAMILIES[FAM]['micro'], shuffle=False, num_workers=1)
    rec = {}; logs = []
    for e in range(16, 21):
        ck = torch.load(run / f'epoch_{e:02d}.pt', map_location='cpu', weights_only=True)
        net = Fm.build_net(argparse.Namespace(family=FAM, micro=0), s, None); net.load_state_dict(ck['model'], strict=True); net.eval()
        mu = torch.tensor(ck['log_mean'], device='cuda', dtype=torch.float32); sigma = torch.tensor(ck['log_std'], device='cuda', dtype=torch.float32)
        single, flips, seen = [], [], []
        with torch.inference_mode():
            for x, _, ix in dl:
                zs = []
                for hf in (False, True):
                    for vf in (False, True):
                        xx = x
                        if hf: xx = torch.flip(xx, (-1,))
                        if vf: xx = torch.flip(xx, (-2,))
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            zs.append(net(Fm.inputs(xx, FAM)).float())
                single.append((zs[0] * sigma + mu).cpu().numpy()); flips.append((torch.stack(zs).mean(0) * sigma + mu).cpu().numpy()); seen.append(ix.numpy())
        np.testing.assert_array_equal(np.concatenate(seen), va)
        p = np.exp(np.concatenate(single)); logs.append(np.concatenate(flips))
        with np.load(run / f'predictions_epoch_{e:02d}.npz') as z:
            assert (z['indices'] == va).all(); q = z['pred'].astype(np.float64)
        rec[f'epoch_{e:02d}.pt'] = dict(sha256=sha(run / f'epoch_{e:02d}.pt'), max_rel_diff_single_vs_stored=float(np.abs(p / q - 1).max()),
                                        cd_single=[round(float(v), 2) for v in p[:, 0]], cd_stored=[round(float(v), 2) for v in q[:, 0]])
        print(json.dumps({f's{s}': {f'epoch_{e:02d}.pt': rec[f'epoch_{e:02d}.pt']}}), flush=True)
        del net; torch.cuda.empty_cache()
    tta_log[s] = np.mean(logs, 0)
    rec['deploy_D_held_cd'] = [round(float(v), 3) for v in np.exp(tta_log[s])[:, 0]]
    out[f's{s}'] = rec
out['deploy_D_seedmean_held_cd'] = [round(float(v), 3) for v in np.exp(0.5 * tta_log[46] + 0.5 * tta_log[47])[:, 0]]
print(json.dumps({k: out[k] for k in ('held_ids', 'deploy_D_seedmean_held_cd')}), {s: out[f's{s}']['deploy_D_held_cd'] for s in ROOTS}, flush=True)
Path('experiments/lane_cd_20261108/reads/verify_f99_dinov3_20261111.json').write_text(json.dumps(out, indent=1))
print('saved reads/verify_f99_dinov3_20261111.json')
