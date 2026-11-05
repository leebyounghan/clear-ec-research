"""Green day 3 / E11: push the pre-only detector without challenge labels.  See e11_nolabel/protocol.json.
  pseudo        teacher (shipped pre-only) peaks on real frames outside fold 4 (GPU; --rows-from/--rows-to for splitting)
  train-student noisy student from the teacher on gated pseudo-labels (GPU)
  train-mixed   SREP + synthetic v5 mixed training from the fold-4 encoder (GPU)
  score25       --ckpt PATH --tag NAME : all-25 scoring vs shipped with the E11 gate (GPU)
"""
import argparse, glob, json, os, sys, time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.spatial import cKDTree, Delaunay
sys.path.insert(0, 'experiments/green_track_20261028/source'); sys.path.insert(0, 'experiments/green_day3_20261103/source'); sys.path.insert(0, 'experiments/green_day2_20261101/source')
os.environ.setdefault('E10_DATA', 'experiments/meta_ceiling_20261103/synth_data/v5')
import green_point_features_20261027 as G
import green_selftrain_20261028 as ST
import green_sg2_specialist_20261028 as SG
import green_external_srep_20261030 as E
import green_day2_cd_features_20261101 as D
import green_day3_e10_synth_20261103 as E10
ROOT = Path('experiments/green_day3_20261103'); OUT = ROOT/'e11_nolabel'; OUT.mkdir(parents=True, exist_ok=True)
HGT, WID = 972, 1296; TH_POS, TH_NEG, FAR_PX, GATE_N = 0.6, 0.2, 16.0, 100
log = E10.log


def fold_rows():
    with np.load('experiments/recipeA_analysis_20261028/seed46_oof_final.npz') as z:
        idx, fold = z['indices'], z['fold']
    o = np.argsort(idx); return np.where(fold[o] != 4)[0]


def pseudo(a):
    enc, head = E10.load_detector(E10.SHIPPED, 'cuda'); mm = np.memmap(D.RAW, dtype=np.uint8, mode='r', shape=(8954, HGT, WID))
    rows = fold_rows(); rows = rows[(rows >= a.rows_from) & (rows < a.rows_to)]
    R_, O_, P_ = [], [0], []; t0 = time.time()
    for k, r in enumerate(rows):
        det = E10.detect(enc, head, E10.prep(np.asarray(mm[r])), 'cuda', threshold=TH_NEG)
        R_.append(int(r)); P_.append(det.astype(np.float32)); O_.append(O_[-1]+len(det))
        if k % 500 == 0:
            log('pseudo', k=k, of=len(rows), seconds=round(time.time()-t0, 1))
    np.savez_compressed(OUT/f'teacher_peaks_{a.rows_from:05d}.npz', rows=np.array(R_), offsets=np.array(O_), peaks=np.concatenate(P_), th_store=TH_NEG)
    log('pseudo done', n=len(rows), seconds=round(time.time()-t0, 1))


def load_pseudo():
    rows, pk = [], {}
    for p in sorted(glob.glob(str(OUT/'teacher_peaks_*.npz'))):
        with np.load(p) as z:
            R, O, P = z['rows'], z['offsets'], z['peaks']  # 2026-11-04 fix: read each member once (NpzFile re-reads on every access; per-row z['peaks'] slices kept a full copy per row -> OOM)
        for i, r in enumerate(R):
            pk[int(r)] = P[O[i]:O[i+1]]; rows.append(int(r))
    return sorted(rows), pk


def pseudo_target(peaks):
    pos = peaks[peaks[:, 2] >= TH_POS, :2].astype(np.float64)
    t, valid = SG.target2(pos, [0, 0, WID-1, HGT-1]) if len(pos) else (np.zeros(SG.GRID2, np.float32), np.ones(SG.GRID2, bool))
    low = peaks[peaks[:, 2] >= TH_NEG, :2]
    occ = np.zeros(SG.GRID2, np.uint8)
    if len(low):
        gx = np.clip(((low[:, 0]-SG.OFFSET2)/SG.STRIDE2).round().astype(int), 0, SG.GRID2[1]-1); gy = np.clip(((low[:, 1]-SG.OFFSET2)/SG.STRIDE2).round().astype(int), 0, SG.GRID2[0]-1); occ[gy, gx] = 1
    dist = cv2.distanceTransform((1-occ).astype(np.uint8), cv2.DIST_L2, 3)*SG.STRIDE2
    support = valid & ((t >= 0.05) | (dist > FAR_PX))
    return t.astype(np.float32), support


class PseudoSet(torch.utils.data.Dataset):
    def __init__(self, rows, pk): self.rows, self.pk, self.mm = rows, pk, None
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        if self.mm is None: self.mm = np.memmap(D.RAW, dtype=np.uint8, mode='r', shape=(8954, HGT, WID))
        r = self.rows[i]; t, sup = pseudo_target(self.pk[r]); return torch.from_numpy(E10.prep(np.asarray(self.mm[r])))[None], torch.from_numpy(t), torch.from_numpy(sup)


def photometric(x, rng):
    g = float(rng.uniform(0.7, 1.4)); c = float(rng.uniform(0.8, 1.25)); b = float(rng.uniform(-0.08, 0.08))
    m = x.mean(dim=(-2, -1), keepdim=True); x = ((x.clamp(0, 1)**g-m)*c+m+b)
    if rng.random() < 0.3:
        s = float(rng.uniform(0.5, 1.2)); k = int(2*round(2*s)+1); x = torch.nn.functional.avg_pool2d(x, 1) if k < 3 else _gauss(x, s, k)
    x = x+torch.randn_like(x)*float(rng.uniform(0, 0.03)); return x.clamp(0, 1)


def _gauss(x, s, k):
    ax = torch.arange(k, dtype=torch.float32, device=x.device)-(k-1)/2; w = torch.exp(-ax**2/(2*s*s)); w = w/w.sum()
    x = F.conv2d(F.pad(x, (k//2, k//2, 0, 0), mode='reflect'), w.view(1, 1, 1, k)); return F.conv2d(F.pad(x, (0, 0, k//2, k//2), mode='reflect'), w.view(1, 1, k, 1))


def train_student(a):
    assert torch.cuda.device_count() == 1; torch.set_num_threads(4)
    rows, pk = load_pseudo(); gated = [r for r in rows if (pk[r][:, 2] >= TH_POS).sum() >= GATE_N]
    log('student data', rows=len(rows), gated=len(gated), median_pos=float(np.median([(pk[r][:, 2] >= TH_POS).sum() for r in gated])))
    ck = torch.load(E10.SHIPPED, map_location='cpu', weights_only=False)
    encoder = G.load_encoder(ST.ENCODERS[4]); encoder.load_state_dict(ck['encoder'], strict=True); encoder = encoder.train().requires_grad_(True)
    head = SG.Head(seed=int(ck.get('seed', 46))).cuda(); head.load_state_dict(ck['head'])
    opt = torch.optim.AdamW([{'params': encoder.parameters(), 'lr': 1e-5}, {'params': head.parameters(), 'lr': 3e-4}], weight_decay=1e-4)
    dl = torch.utils.data.DataLoader(PseudoSet(gated, pk), batch_size=2, shuffle=True, num_workers=2, pin_memory=True, drop_last=True, persistent_workers=True)
    rng = np.random.default_rng(46); step = 0; losses = []; t0 = time.time()
    while step < a.updates:
        for x, t, sup in dl:
            f = int(rng.integers(4)); h, v = G.FLIPS[f]
            if h: x, t, sup = x.flip(-1), t.flip(-1), sup.flip(-1)
            if v: x, t, sup = x.flip(-2), t.flip(-2), sup.flip(-2)
            x = photometric(x.cuda(non_blocking=True), rng); t = t.cuda(non_blocking=True); sup = sup.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                tok = encoder.forward_features(G.b.inputs(x, 'cpdino_vitb'))['x_norm_patchtokens']
            logits = head(tok); loss = sum(F.binary_cross_entropy_with_logits(logits[i][sup[i]], t[i][sup[i]]) for i in range(len(t)))/len(t)
            loss.backward(); torch.nn.utils.clip_grad_norm_(list(encoder.parameters())+list(head.parameters()), 1.0, error_if_nonfinite=True); opt.step(); losses.append(float(loss)); step += 1
            if step % 200 == 0 or step == 1: log('train-student', step=step, of=a.updates, loss=round(float(loss), 4), seconds=round(time.time()-t0, 1))
            if step >= a.updates: break
    encoder.eval().requires_grad_(False); head.eval()
    torch.save(dict(encoder=encoder.state_dict(), head=head.state_dict(), kind='sg2', features='last', seed=46, threshold=0.35, init='deploy sg2pre_s46', updates=a.updates, mode='E11a noisy student from the pre-only teacher on real frames outside fold 4'), OUT/'detector_student.pt')
    (OUT/'train_student.json').write_text(json.dumps(dict(created_utc=G.now(), updates=a.updates, gated_rows=len(gated), loss_last200=float(np.mean(losses[-200:])), seconds=round(time.time()-t0, 1)), indent=1)); log('train-student done', loss_last200=float(np.mean(losses[-200:])))


def train_mixed(a):
    assert torch.cuda.device_count() == 1; torch.set_num_threads(4)
    imgs, pts, masks, shape = E.load_cache(E.SCALE_NOMINAL); n = len(imgs)
    metas = E10.synth_items(0); syn = torch.utils.data.DataLoader(E10.SynthSet(metas), batch_size=1, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    encoder = G.load_encoder(ST.ENCODERS[4]).train().requires_grad_(True); head = SG.Head(seed=46).cuda()
    opt = torch.optim.AdamW([{'params': encoder.parameters(), 'lr': 1e-5}, {'params': head.parameters(), 'lr': 1e-3}], weight_decay=1e-4)
    rng = np.random.default_rng(46); cache = {}; losses = []; t0 = time.time(); it = iter(syn); gh, gw = (shape[0]+20)//8, (shape[1]+16)//8
    for step in range(a.updates):
        idx = rng.choice(n, 2, replace=False); flips = rng.integers(4, size=2); xs, ts, ss = [], [], []
        for i, f in zip(idx, flips):
            h, v = G.FLIPS[f]; key = (int(i), int(f))
            if key not in cache:
                p, m = E.flip_pts(pts[i], masks[i], shape, h, v); t, sup = E.rasterise(p, m, shape, 2, 0.5); cache[key] = (torch.from_numpy(t).cuda(), torch.from_numpy(sup).cuda())
                if len(cache) > 400: cache.pop(next(iter(cache)))
            x = torch.from_numpy(np.asarray(imgs[i], np.float32))[None, None]; xs.append(G.flip_image(x, f)); ts.append(cache[key][0]); ss.append(cache[key][1])
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            tok = encoder.forward_features(G.b.inputs(torch.cat(xs), 'cpdino_vitb'))['x_norm_patchtokens']
        logits = F.pixel_shuffle(head.lin(tok.float()).reshape(2, gh, gw, 16).permute(0, 3, 1, 2), 4).squeeze(1)
        l_srep = sum(F.binary_cross_entropy_with_logits(logits[j][:ts[j].shape[0], :ts[j].shape[1]][ss[j]], ts[j][ss[j]]) for j in range(2))/2
        try: xsyn, tsyn, ssyn = next(it)
        except StopIteration: it = iter(syn); xsyn, tsyn, ssyn = next(it)
        f = int(rng.integers(4)); h, v = G.FLIPS[f]
        if h: xsyn, tsyn, ssyn = xsyn.flip(-1), tsyn.flip(-1), ssyn.flip(-1)
        if v: xsyn, tsyn, ssyn = xsyn.flip(-2), tsyn.flip(-2), ssyn.flip(-2)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            tok2 = encoder.forward_features(G.b.inputs(xsyn.cuda(non_blocking=True), 'cpdino_vitb'))['x_norm_patchtokens']
        lg2 = head(tok2)[0]; tsyn = tsyn[0].cuda(); ssyn = ssyn[0].cuda(); l_syn = F.binary_cross_entropy_with_logits(lg2[ssyn], tsyn[ssyn])
        loss = (l_srep+l_syn)/2; loss.backward(); torch.nn.utils.clip_grad_norm_(list(encoder.parameters())+list(head.parameters()), 1.0, error_if_nonfinite=True); opt.step(); losses.append([float(l_srep), float(l_syn)])
        if step % 200 == 0: log('train-mixed', step=step, of=a.updates, l_srep=round(float(l_srep), 4), l_syn=round(float(l_syn), 4), seconds=round(time.time()-t0, 1))
    encoder.eval().requires_grad_(False); head.eval()
    torch.save(dict(encoder=encoder.state_dict(), head=head.state_dict(), kind='sg2', features='last', seed=46, threshold=0.35, init_encoder=str(ST.ENCODERS[4]), updates=a.updates, mode='E11b SREP + synthetic v5 mixed training'), OUT/'detector_mixed.pt')
    (OUT/'train_mixed.json').write_text(json.dumps(dict(created_utc=G.now(), updates=a.updates, n_srep=int(n), n_synth=len(metas), loss_last200=dict(srep=float(np.mean([l[0] for l in losses[-200:]])), synth=float(np.mean([l[1] for l in losses[-200:]]))), seconds=round(time.time()-t0, 1)), indent=1)); log('train-mixed done')


def score25(a):
    torch.set_num_threads(4)
    ann = json.loads(Path('datasets/clear_ec_official/green_eval/annotations.json').read_text()); binding = json.loads(Path('experiments/green_point_probe_20261023/data_binding.json').read_text()); train_ids = set(binding['train_ids'])
    with np.load('datasets/clear_ec_official/cache_native_meta_512.npz', allow_pickle=True) as z:
        ids = z['ids'].astype(str)
    pos = {n: i for i, n in enumerate(ids)}; mm = np.memmap(D.RAW, dtype=np.uint8, mode='r', shape=(8954, HGT, WID))
    res = dict(created_utc=G.now(), tag=a.tag, ckpt=a.ckpt, detectors={})
    for name, path in ((a.tag, Path(a.ckpt)), ('shipped_pre_s46', E10.SHIPPED)):
        enc, head = E10.load_detector(path, 'cuda'); rows = {}
        for n_, e in ann.items():
            r = pos[n_]; img = np.asarray(mm[r]); det = E10.detect(enc, head, E10.prep(img), 'cuda')
            gt = np.asarray(e['markers_mha'], float); hull = Delaunay(gt); sp = float(np.median(cKDTree(gt).query(gt, k=2)[0][:, 1]))
            if len(det):
                dmin, _ = cKDTree(gt).query(det[:, :2]); inside = (hull.find_simplex(det[:, :2]) >= 0) | (dmin <= 0.5*sp)
            else: inside = np.zeros(0, bool)
            m = E10.measure(gt, det[inside, :2].astype(np.float64), None)
            if m is None: continue
            m.update(val=n_ not in train_ids); rows[n_] = m
        agg = lambda sel: None if not sel else {k: float(np.nanmean([r[k] for r in sel])) for k in ('miss', 'fp', 'jitter', 'f1_12px')} | dict(n=len(sel))
        res['detectors'][name] = dict(val5=agg([v for v in rows.values() if v['val']]), all25=agg(list(rows.values())), per_image=rows)
        log('score25', detector=name, val5=res['detectors'][name]['val5'], all25=res['detectors'][name]['all25'])
    v, s = res['detectors'][a.tag], res['detectors']['shipped_pre_s46']
    g1 = (v['all25']['miss'] <= 0.12 or v['all25']['jitter'] <= 0.19) and v['all25']['fp'] <= 0.18
    g2 = all(v['val5'][k] <= s['val5'][k]+0.02 for k in ('miss', 'fp', 'jitter'))
    res['gate'] = dict(all25_rule=bool(g1), val5_not_worse=bool(g2), passed=bool(g1 and g2), rule='all-25 miss <= 0.12 or jitter <= 0.19, FP <= 0.18; val-5 miss/FP/jitter within +0.02 of shipped')
    (OUT/f'score25_{a.tag}.json').write_text(json.dumps(res, indent=1, default=float)); log('score25 done', tag=a.tag, gate=res['gate'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['pseudo', 'train-student', 'train-mixed', 'score25']); ap.add_argument('--rows-from', type=int, default=0); ap.add_argument('--rows-to', type=int, default=8954)
    ap.add_argument('--updates', type=int, default=4000); ap.add_argument('--ckpt', default=''); ap.add_argument('--tag', default='student')
    a = ap.parse_args(); {'pseudo': pseudo, 'train-student': train_student, 'train-mixed': train_mixed, 'score25': score25}[a.command](a)
