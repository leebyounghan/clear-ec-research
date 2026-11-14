"""Deploy D on GPU (production path: members.dino_cd, bf16 autocast) on the CD lane's two held images, vs the CD lane's reference values
(message 11-11 05:30 EST: s46 1107.292 / 1555.547, s47 1057.343 / 1481.693, seed mean 1082.03 / 1518.171). Also full sha256 vs the CD lane's list. usage: test_dino_f99_gpu.py"""
import sys, json, hashlib
import numpy as np, torch
torch.set_num_threads(4)
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference')
import members as M
E = 'experiments/'; R = E + 'lane_cd_20261108/arms/dinov3_b16_f99_{}/runs/fold99/dinov3_b16/'
REF = {'s46b': (1107.292, 1555.547), 's47': (1057.343, 1481.693), 'two_seed': (1082.03, 1518.171)}
SHA = {16: '99b4e536b60e26a330b1906576856d0d94f8ac844d84efd100cf6bc79876ff2b', 17: 'a8a685eb447fa0db208d69d3e1c8216afe72f30d1afd2f803727cc3539ae05e2',
       18: '51878c275aaf87d93c039d50c1f43936f6358d74419beb1bc53981a3ad183af7', 19: '82cfec94ffcf9cbf407290cd389cf0801b43a002c147e5ff73c2ab06fc23d9cf',
       20: 'd3452baaaf946bb2f9834b05997c2545678ae9bc67884e4bdadc0788707d6431'}
sha_ok = {}
for e, h in SHA.items():
    d = hashlib.sha256()
    with open(R.format('s47') + f'epoch_{e}.pt', 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 24), b''): d.update(b)
    sha_ok[e] = d.hexdigest() == h
zc = np.load(E + 'lane_cd_20261108/reads/freeze_cd_oof_20261111.npz', allow_pickle=True); ids = {int(i): str(s) for i, s in zip(zc['idx'], zc['ids'])}
st = np.load(R.format('s47') + 'predictions_epoch_20.npz'); frames = [int(i) for i in st['indices']]
MM = np.memmap(M.ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
ck = {s: [R.format(s) + f'epoch_{e}.pt' for e in range(16, 21)] for s in ('s46b', 's47')}; out = dict(sha256_match=sha_ok, cuda=torch.cuda.is_available(), frames={})
for j, d in enumerate(frames):
    M.clear_cache(); x = M.preprocess(np.asarray(MM[d]))
    v = dict(two_seed=float(M.dino_cd(x, [ck['s46b'], ck['s47']])[0]), s46b=float(M.dino_cd(x, ck['s46b'])[0]), s47=float(M.dino_cd(x, ck['s47'])[0]))
    out['frames'][d] = dict(id=ids.get(d), **{k: dict(mine=v[k], ref=REF[k][j], rel=v[k] / REF[k][j] - 1) for k in v})
print('SUMMARY', json.dumps(out), flush=True); json.dump(out, open(E + 'lane_cellroute_20261109/out/dino_f99_s47_gpu.json', 'w'), indent=1)
