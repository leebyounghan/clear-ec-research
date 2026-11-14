"""Do single-image peaks differ from the stored (batch-of-4) training peaks because of batch-dependent bf16 numerics? fold-0 pre-only s46 detector,
4 fold-0 validation frames: stored external_srep/preonly/fold0 peaks vs (a) det_peaks_image.peaks_for with batch 4 (training path), (b) the same path with
batch 1, (c) deploy_hexkonan_image identity-flip heat (batch 1)."""
import sys, numpy as np, torch
from pathlib import Path
sys.path[:0] = ['/workspace/clear-ec-research/experiments/lane_cellroute_20261109/source', '/workspace/clear-ec-research/experiments/lane_hex_20261108/source']
import os; os.chdir('/workspace/clear-ec-research')
import det_peaks_image as DP
import deploy_hexkonan_image as DI
G, SG, ST = DP.G, DP.SG, DP.ST
raw = 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat'; z = np.load('experiments/green_track_20261028/external_srep/preonly/fold0/fold0_sg2_peaks.npz', allow_pickle=True)
pos = {int(i): k for k, i in enumerate(z['indices'])}; rows = np.sort(np.random.default_rng(0).choice(list(pos), 4, replace=False))
enc, head = DP.load('full', 'experiments/green_track_20261028/external_srep/fold0/sg2_pre.pt', enc_init=ST.ENCODERS[0]); total = 8954
def cmp(P, O, label):
    for k, r in enumerate(rows):
        a = P[O[k]:O[k + 1]]; j = pos[int(r)]; b = z['peaks'][z['offsets'][j]:z['offsets'][j + 1]]
        same = a.shape == b.shape and (len(a) == 0 or np.abs(np.sort(a, 0) - np.sort(b, 0)).max() < 1e-4)
        print(label, int(r), 'mine', len(a), 'stored', len(b), 'identical' if same else 'DIFFERENT', flush=True)
P, O = DP.peaks_for(enc, head, raw, rows, total); cmp(P, O, 'batch4 training path')
pk = []; of = [0]
for r in rows:
    P1, O1 = DP.peaks_for(enc, head, raw, np.array([r]), total); pk.append(P1); of.append(of[-1] + len(P1))
cmp(np.concatenate(pk), np.array(of), 'batch1 training path')
MM = np.memmap(raw, dtype=np.uint8, mode='r', shape=(total, 972, 1296)); pk = []; of = [0]
for r in rows:
    q = SG.peaks2(DI.heat(enc.cuda(), head, DI.preprocess(np.asarray(MM[r])))[0], SG.frame_support(), 0.2).astype(np.float32); pk.append(q); of.append(of[-1] + len(q))
cmp(np.concatenate(pk), np.array(of), 'entry identity flip')
# (d) the image repeated 4 times as one batch (same kernel shapes as the batch-4 training path), first copy's map
pk = []; of = [0]
for r in rows:
    x = DI.preprocess(np.asarray(MM[r])); x4 = x.repeat(4, 1, 1, 1)
    q = SG.peaks2(DI.heat(enc.cuda(), head, x4)[0], SG.frame_support(), 0.2).astype(np.float32); pk.append(q); of.append(of[-1] + len(q))
cmp(np.concatenate(pk), np.array(of), 'entry batch of 4 copies')
