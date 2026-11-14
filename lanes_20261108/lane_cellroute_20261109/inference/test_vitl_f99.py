"""Smoke of the full-data ViT-L s46 half (cell-route lane, 2026-11-10). Test harness, CPU. Checks: (1) sha256 against the CD lane's list; (2) strict load into
members.py's dinov3_l16t net; (3) my read reproduces the trainer's own stored predictions on its 2 held-out frames (predictions_epoch_08..11 single view,
predictions_final_tta = TTA4 at final.pt); (4) the s46 half of V on 8 fold-0 frames vs stored fold-0 V (different weights: sanity only, not parity).
usage: test_vitl_f99.py SEED THREADS"""
import sys, json, hashlib, time
import numpy as np, torch
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference')
import members as M
seed = sys.argv[1]; torch.set_num_threads(int(sys.argv[2]))
E = 'experiments/'; D = E + f'lane_cd_20261108/arms/vitl16_f99_{seed}/raw/runs/fold99/dinov3_l16t/'
SHA = {'s46': dict(epoch_08='dec28272129e1b6115f7899bab876a4c95e206310c49840013aad4f3bd421079', epoch_09='223249f231622f0a1efa12d3781926ea66ddda3c0e4d8558790acf98df96468c',
                   epoch_10='a08a7d18573a9072e6b925b16c48ce113588959fecea5177bd6b9090ad040667', epoch_11='a3c72175e9bd871b07f93c0f2fa506fe74e93b5d01d30a655e7e0ef1f028efbe',
                   final='fbb0eca88712774c5b48d3e4dbd44db95efb7ad9977d49ac53e03fd06c5a0dc1'),
       's47': dict(epoch_08='fcdaa0bac47fbb23d6617d3fbed8b891f714257e1a0fe49c3ad4eab48aa21456', epoch_09='6ea893ed62cfd288cced2ecbaa26b467efd16513cf5d63d94e342b760486ecfd',
                   epoch_10='de22b99667316c49d601c1771b6846b49a36d925d0a022759f5a0deb36388942', epoch_11='1b5d22929f99de4cdfa4a14b07e3840c1649a6d44527d7e455c55221ea2e7b95',
                   final='d43c3a73742d02eed7118bc76ea2ce232b168956b8046697265c82f563ab6b4b')}.get(seed, {})
out = dict(seed=seed, sha={}, held={}, fold0={})
for n in ('epoch_08', 'epoch_09', 'epoch_10', 'epoch_11', 'final'):
    h = hashlib.sha256()
    with open(D + n + '.pt', 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    out['sha'][n] = dict(sha=h.hexdigest(), match=(SHA.get(n) == h.hexdigest()) if SHA else None)
print('SHA', json.dumps(out['sha']), flush=True)
MM = np.memmap(M.ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
ref = np.load(D + 'predictions_epoch_08.npz'); hidx = [int(i) for i in ref['indices']]
zc = np.load(E + 'lane_cd_20261108/reads/freeze_cd_oof_20261109.npz'); r0 = np.where(zc['fold'] == 0)[0]; rows = r0[np.linspace(0, len(r0) - 1, 8).astype(int)]
fidx = [int(zc['idx'][k]) for k in rows]
X = {d: M.preprocess(np.asarray(MM[d])) for d in hidx + fidx}
single = {d: [] for d in X}; t0 = time.time()
for e in range(8, 12):
    m = M.Member(D + f'epoch_{e:02d}.pt', 'dinov3_l16t')
    st = np.load(D + f'predictions_epoch_{e:02d}.npz')
    for d in X:
        L = m.logpred(X[d], False); single[d].append(L)
        if d in hidx: out['held'][f'e{e:02d}_{d}'] = dict(mine=np.exp(L).tolist(), stored=st['pred'][hidx.index(d)].tolist(), rel=np.abs(np.exp(L) / st['pred'][hidx.index(d)] - 1).tolist())
    print('epoch', e, round(time.time() - t0), flush=True)
m = M.Member(D + 'final.pt', 'dinov3_l16t'); st = np.load(D + 'predictions_final_tta.npz'); tta = {}
for d in X:
    tta[d] = m.logpred(X[d], True)
    if d in hidx: out['held'][f'final_tta_{d}'] = dict(mine=np.exp(tta[d]).tolist(), stored=st['pred'][hidx.index(d)].tolist(), rel=np.abs(np.exp(tta[d]) / st['pred'][hidx.index(d)] - 1).tolist())
for k, d in zip(rows, fidx):
    half = float(np.exp(0.5 * np.mean(single[d], 0)[0] + 0.5 * tta[d][0]))
    out['fold0'][d] = dict(s46_half_cd=half, stored_fold0_V=float(zc['V'][k]), ratio=half / float(zc['V'][k]))
held_max = max(max(v['rel'][:1]) for v in out['held'].values()); held_max3 = max(max(v['rel']) for v in out['held'].values())
rat = np.array([v['ratio'] for v in out['fold0'].values()])
out['summary'] = dict(sha_all_match=all(v['match'] for v in out['sha'].values()) if SHA else None, held_max_rel_cd=held_max, held_max_rel_all3=held_max3,
                      fold0_ratio_median=float(np.median(rat)), fold0_ratio_min=float(rat.min()), fold0_ratio_max=float(rat.max()), seconds=round(time.time() - t0))
print('SUMMARY', json.dumps(out['summary']), flush=True)
json.dump(out, open(E + f'lane_cellroute_20261109/out/vitl_f99_{seed}_smoke.json', 'w'), indent=1)
