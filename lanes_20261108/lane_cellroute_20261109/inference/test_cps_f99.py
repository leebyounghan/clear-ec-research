"""Read check of the HEX lane's full-data CPSAM 512 (mtcps_c512_f99, input cps5 of hex_apply) (cell-route lane, 2026-11-10). Test harness, CPU.
For every epoch file on disk: strict load into members.py's cpsam_c512 net, then my single-view read vs the trainer's stored predictions on its 2 held-out
frames (predictions_epoch_XX). usage: test_cps_f99.py THREADS"""
import sys, os, json, time
import numpy as np, torch
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference')
import members as M
torch.set_num_threads(int(sys.argv[1]))
E = 'experiments/'; D = E + 'lane_hex_20261108/deploy/mtcps_c512_f99/runs/fold99/cpsam_c512/'
MM = np.memmap(M.ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
out = {}; t0 = time.time()
for e in range(8, 13):
    if not os.path.exists(D + f'epoch_{e:02d}.pt'): continue
    st = np.load(D + f'predictions_epoch_{e:02d}.npz'); m = M.Member(D + f'epoch_{e:02d}.pt', 'cpsam_c512')
    for j, d in enumerate(int(i) for i in st['indices']):
        mine = np.exp(m.logpred(M.preprocess(np.asarray(MM[d])), False))
        out[f'e{e:02d}_{d}'] = dict(mine=mine.tolist(), stored=st['pred'][j].tolist(), rel=np.abs(mine / st['pred'][j] - 1).tolist())
    print('epoch', e, round(time.time() - t0), json.dumps({k: [round(x, 5) for x in v['rel']] for k, v in out.items() if k.startswith(f'e{e:02d}')}), flush=True)
print('SUMMARY', json.dumps(dict(epochs=sorted({k[:3] for k in out}), max_rel_hex=max(v['rel'][2] for v in out.values()), max_rel_all3=max(max(v['rel']) for v in out.values()))), flush=True)
json.dump(out, open(E + 'lane_cellroute_20261109/out/cps_f99_smoke.json', 'w'), indent=1)
