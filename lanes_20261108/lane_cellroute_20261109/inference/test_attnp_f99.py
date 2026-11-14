"""Read check of the backbone lane's full-data attention-pool DINOv3-B (attnp5 of the HEX lane's 39-column ridge) (cell-route lane, 2026-11-11). GPU.
(1) sha256 of the 5 raw checkpoints vs the hand-over list; (2) members.attnpool_hex with one checkpoint vs the trainer's stored predictions_epoch_XX
(final.pt = epoch 12) on its held rows; (3) attnp5 (all 5) on those rows; (4) the SSL weights loaded at build (ATTNPOOL_SSL) vs not: attnp5 must be
bit-identical, since every strict checkpoint load overwrites them. usage: test_attnp_f99.py"""
import sys, os, json, hashlib
import numpy as np, torch
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference')
import members as M
E = 'experiments/'; D = E + 'lane_backbone_20261108/runs/attnpool_f99_dinov3b16_s46/raw/runs/fold99/dinov3_b16/'
SHA = {'epoch_08.pt': '069331d2fa0d34633aef0b4f35f7f920d922eb01d89054d7bb7087408f7f1848', 'epoch_09.pt': 'f87bb3fb6286abbd34627b8785c3b261f56b2fb567a2052ab958b51a3a3a8e64',
       'epoch_10.pt': '79c8e4aa7dce9275e6e3eacf81998402b7b8aed1008f3de4946647375a72c664', 'epoch_11.pt': '36fde73696242bb60343370031111ff55699c1bdfbef412925261e7bb8d5acd6',
       'final.pt': 'd29631486c0311a42806b7a04e32bf738b141e54e029763a2f9e5248edfec51a'}
out = dict(sha256_match={}, single={})
for n, h in SHA.items():
    d = hashlib.sha256()
    with open(D + n, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 24), b''): d.update(b)
    out['sha256_match'][n] = d.hexdigest() == h
MM = np.memmap(M.ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
ck = [D + n for n in SHA]; ep = [8, 9, 10, 11, 12]
for n, e in zip(SHA, ep):
    st = np.load(D + f'predictions_epoch_{e:02d}.npz'); frames = [int(i) for i in st['indices']]
    for j, d in enumerate(frames):
        M.clear_cache(); x = M.preprocess(np.asarray(MM[d])); mine = M.attnpool_hex(x, [D + n])
        out['single'][f'{n}_{d}'] = dict(mine=mine.tolist(), stored=st['pred'][j].tolist(), rel=np.abs(mine / st['pred'][j] - 1).tolist())
out['attnp5'] = {}
for d in frames:
    M.clear_cache(); x = M.preprocess(np.asarray(MM[d])); out['attnp5'][d] = float(M.attnpool_hex(x, ck)[2])
os.environ['ATTNPOOL_SSL'] = E + 'lane_cd_20261108/arms/dinov3_b16_f99_s46/ssl/last.pt'; M.Member._cache.pop('attnpool', None)
out['ssl_identical'] = {}
for d in frames:
    M.clear_cache(); x = M.preprocess(np.asarray(MM[d])); out['ssl_identical'][d] = float(M.attnpool_hex(x, ck)[2]) == out['attnp5'][d]
print('SUMMARY', json.dumps(dict(sha256_match=out['sha256_match'], max_rel_hex=max(v['rel'][2] for v in out['single'].values()),
                                 max_rel_all3=max(max(v['rel']) for v in out['single'].values()), attnp5=out['attnp5'], ssl_identical=out['ssl_identical'])), flush=True)
json.dump(out, open(E + 'lane_cellroute_20261109/out/attnp_f99_smoke.json', 'w'), indent=1)
