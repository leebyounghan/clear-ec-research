"""Detector peaks for unseen images (HEX lane, 2026-11-10): runs the full-data detectors of the HEX chain on a raw-frame memmap
and writes their peaks in the per-fold peak-file format (<root>/<detector>/fold0/fold0_sg2_peaks.npz, indices 0..M-1), the
input of hex_test_features.py. Same input pipeline (green_track Frames: CLAHE 2.0 / 40x30 tiles, 1-99 % scaling), same
heat map (SG.heat) and peak extraction (SG.peaks2, threshold 0.2, NMS 12 px) as the training peak files.
  --set pre3 | all16   detectors to run (feat_fuse.SETS names); pre3 feeds prior, ridge k3 and the graph networks, all16 adds
                       the 13 detectors of ridgeev_fall16
  --verify K           runs the fold-0 checkpoints of the three detector families on K fold-0 validation frames and compares
                       their peaks with the stored fold-0 peak files (must match)
usage: det_peaks_image.py --images X.dat --n M --out ROOT [--set all16] | --verify 12"""
import os, sys, argparse, json
from pathlib import Path
import numpy as np
import torch
os.chdir('/workspace/clear-ec-research')
sys.path.insert(0, 'experiments/green_track_20261028/source'); sys.path.insert(0, 'experiments/lane_hex_20261108/source')
import green_point_features_20261027 as G
import green_sg2_specialist_20261028 as SG
import green_selftrain_20261028 as ST
GD = 'experiments/green_track_20261028/deploy'; HD = 'experiments/lane_hex_20261108/deploy'
FULL = 'experiments/fulldata_recipeA_20261028/regression_std_s46/runs/fold99/cpdino_vitb/epoch_20.pt'
# detector name (feat_fuse.SETS) -> (kind, checkpoint, seed); kind 'full' = encoder+head state, 'frozen' = head on the FULL encoder
FULLDATA = {'external_srep/preonly': ('full', f'{GD}/detectors/sg2pre_s46.pt'), 'external_srep/s47_preonly': ('full', f'{GD}/detectors/sg2pre_s47.pt'),
            'external_srep/s48_preonly': ('full', f'{GD}/detectors/sg2pre_s48.pt'), 'sg2_specialist': ('full', f'{GD}/detectors/sg2_s46.pt'),
            **{f'sg2_specialist_s{s}': ('full', f'{GD}/detectors/sg2_s{s}.pt') for s in range(47, 51)},
            'sg2_specialist_s51': ('full', f'{HD}/detectors/sg2_specialist_s51/detectors/sg2_s51.pt'), 'sg2_specialist_s52': ('full', f'{HD}/detectors/sg2_specialist_s52/detectors/sg2_s52.pt'),
            'sg2_sigma3p5': ('full', f'{HD}/detectors/sg2_sigma3p5/detectors/sg2_s46.pt'), 'sg2_sigma5': ('full', f'{HD}/detectors/sg2_sigma5/detectors/sg2_s46.pt'),
            **{f'frozen_heads/{k}': ('frozen', f'{HD}/frozen_heads/{k}/fold0/head.pt') for k in ('sg2f_s46', 'sg2f_s47', 'sg2f_s48', 'sg2pref_s46')}}


def load(kind, path, enc_init=None):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if kind == 'full':
        encoder = G.load_encoder(enc_init or ck['init_encoder']); encoder.load_state_dict(ck['encoder'], strict=True)
    else:
        encoder = G.load_encoder(enc_init or FULL)
    head = SG.Head(seed=ck.get('seed', 46)).cuda(); head.load_state_dict(ck['head']); head.eval(); return encoder.eval(), head


def peaks_for(encoder, head, raw_path, rows, total):
    from torch.utils.data import DataLoader
    G.b.RAW = Path(raw_path); y = np.ones((total, 3), np.float32); support = SG.frame_support(); pk_all, of = [], [0]
    dl = DataLoader(G.b.Frames(np.asarray(rows), y, total), batch_size=4, shuffle=False, num_workers=3, pin_memory=True)
    for x, _, ix in dl:
        for m in SG.heat(encoder, head, x):
            pk = SG.peaks2(m, support, SG.STORE_THRESHOLD).astype(np.float32); pk_all.append(pk); of.append(of[-1] + len(pk))
    return (np.concatenate(pk_all) if pk_all else np.zeros((0, 3), np.float32)), np.array(of)


def write(root, det, peaks, offsets, M):
    d = Path(root) / det / 'fold0'; d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / 'fold0_sg2_peaks.npz', peaks=peaks, offsets=offsets, indices=np.arange(M), store_threshold=SG.STORE_THRESHOLD, nms_radius=SG.NMS2)


def verify(K):
    ann, binding, ids, y, tr, va = ST.green(0); rows = np.sort(np.random.default_rng(0).choice(va, K, replace=False)); total = len(ids); rep = {}
    raw = 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat'
    for nm, kind, ck, stored in (('preonly', 'full', 'experiments/green_track_20261028/external_srep/fold0/sg2_pre.pt', 'experiments/green_track_20261028/external_srep/preonly/fold0/fold0_sg2_peaks.npz'),
                                  ('specialist', 'full', 'experiments/green_track_20261028/sg2_specialist/fold0/specialist_sg2.pt', 'experiments/green_track_20261028/sg2_specialist/fold0/fold0_sg2_peaks.npz'),
                                  ('frozen sg2f_s46', 'frozen', 'experiments/green_track_20261028/frozen_heads/sg2f_s46/fold0/head.pt', 'experiments/green_track_20261028/frozen_heads/sg2f_s46/fold0/fold0_sg2_peaks.npz')):
        enc, head = load(kind, ck, enc_init=ST.ENCODERS[0]); P, O = peaks_for(enc, head, raw, rows, total)
        z = np.load(stored, allow_pickle=True); pos = {int(i): k for k, i in enumerate(z['indices'])}; worst = 0.0; cnt = 0
        for k, r in enumerate(rows):
            a = P[O[k]:O[k + 1]]; j = pos[int(r)]; b = z['peaks'][z['offsets'][j]:z['offsets'][j + 1]]
            if a.shape != b.shape: cnt += 1; continue
            if len(a): worst = max(worst, float(np.abs(np.sort(a, 0) - np.sort(b, 0)).max()))
        rep[nm] = dict(frames=int(K), frames_with_other_peak_count=cnt, max_abs_diff=worst); print(nm, rep[nm], flush=True)
    json.dump(rep, open('experiments/lane_hex_20261108/logs/det_peaks_verify.json', 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--images'); ap.add_argument('--n', type=int); ap.add_argument('--out'); ap.add_argument('--set', default='all16'); ap.add_argument('--verify', type=int, default=0); a = ap.parse_args()
    assert torch.cuda.is_available(), 'needs a GPU'
    if a.verify: verify(a.verify); sys.exit(0)
    sys.argv = ['feat_fuse.py', '/dev/null', a.set]; import feat_fuse as FF
    for det in FF.SETS[a.set]:
        kind, ck = FULLDATA[det]; enc, head = load(kind, ck); P, O = peaks_for(enc, head, a.images, np.arange(a.n), a.n); write(a.out, det, P, O, a.n); print('peaks', det, int(O[-1]), flush=True)
        del enc, head; torch.cuda.empty_cache()
    print('done', a.out, flush=True)
