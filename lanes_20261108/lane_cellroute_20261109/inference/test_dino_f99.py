"""Read check of the CD lane's full-data DINOv3-B s47 (second seed of D after the 2026-11-11 seed audit) (cell-route lane). Test harness, CPU.
For epochs 16-20: sha256 (to compare with the CD lane's list), strict load into members.py's dinov3_b16 net, my single-view read vs the trainer's
stored predictions_epoch_XX on its held-out frames; then the deploy D read (4 flips x epochs 16-20 for s46b and s47, seeds averaged in log space)
on those frames, next to s46b alone. usage: test_dino_f99.py THREADS"""
import sys, json, time, hashlib
import numpy as np, torch
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference')
import members as M
torch.set_num_threads(int(sys.argv[1]))
E = 'experiments/'; R = E + 'lane_cd_20261108/arms/dinov3_b16_f99_{}/runs/fold99/dinov3_b16/'
MM = np.memmap(M.ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
out = dict(sha256={}, single={}); t0 = time.time(); frames = None
for e in range(16, 21):
    p = R.format('s47') + f'epoch_{e}.pt'; h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 24), b''): h.update(b)
    out['sha256'][e] = h.hexdigest(); st = np.load(R.format('s47') + f'predictions_epoch_{e}.npz'); m = M.Member(p, 'dinov3_b16'); frames = [int(i) for i in st['indices']]
    for j, d in enumerate(frames):
        mine = np.exp(m.logpred(M.preprocess(np.asarray(MM[d])), False))
        out['single'][f'e{e}_{d}'] = dict(mine=mine.tolist(), stored=st['pred'][j].tolist(), rel=np.abs(mine / st['pred'][j] - 1).tolist())
    print('epoch', e, round(time.time() - t0), out['sha256'][e][:8], json.dumps({k: [round(x, 5) for x in v['rel']] for k, v in out['single'].items() if k.startswith(f'e{e}_')}), flush=True)
ck = {s: [R.format(s) + f'epoch_{e}.pt' for e in range(16, 21)] for s in ('s46b', 's47')}; out['deploy_D'] = {}
for d in frames:
    M.clear_cache(); x = M.preprocess(np.asarray(MM[d]))
    out['deploy_D'][d] = dict(two_seed=float(M.dino_cd(x, [ck['s46b'], ck['s47']])[0]), s46b=float(M.dino_cd(x, ck['s46b'])[0]), s47=float(M.dino_cd(x, ck['s47'])[0]))
print('SUMMARY', json.dumps(dict(sha256={e: v[:8] for e, v in out['sha256'].items()}, max_rel_cd=max(v['rel'][0] for v in out['single'].values()),
                                 max_rel_all3=max(max(v['rel']) for v in out['single'].values()), deploy_D=out['deploy_D'])), flush=True)
json.dump(out, open(E + 'lane_cellroute_20261109/out/dino_f99_s47_smoke.json', 'w'), indent=1)
