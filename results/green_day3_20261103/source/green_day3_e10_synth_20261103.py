"""Green day 3 / E10 (route B): detector trained on synthetic endothelium only.  See e10_synth/protocol.json.
  domain-check [--limit N] [--cpu]   shipped pre-only s46 on synthetic images vs exact centres
  train [--updates 4000]             synthetic-only stride-2 detector on the fold-4 encoder
  score25 [--ckpt PATH]              detector(s) on the 25 annotated real images vs expert points
"""
import argparse, glob, json, os, sys, time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.spatial import cKDTree, Delaunay
sys.path.insert(0, 'experiments/green_track_20261028/source'); sys.path.insert(0, 'experiments/green_day3_20261103/source'); sys.path.insert(0, 'experiments/green_day2_20261101/source')
import green_point_features_20261027 as G
import green_selftrain_20261028 as ST
import green_sg2_specialist_20261028 as SG
import phase2_foundation_preflight_20261002 as preflight
import green_day2_cd_features_20261101 as D
ROOT = Path('experiments/green_day3_20261103'); OUT = ROOT/'e10_synth'; OUT.mkdir(parents=True, exist_ok=True)
SYN = Path(os.environ.get('E10_DATA', 'experiments/meta_ceiling_20261103/synth_data/v1')); HGT, WID = 972, 1296
SHIPPED = Path('experiments/green_track_20261028/deploy/detectors/sg2pre_s46.pt')
MARGIN = 8


def log(msg, **kw):
    print(json.dumps(dict(utc=G.now(), msg=msg, **kw)), flush=True)


def prep(a_u8):
    c = cv2.createCLAHE(2.0, (WID//32, HGT//32)).apply(np.ascontiguousarray(a_u8)); lo, hi = np.percentile(c, (1, 99))
    return np.clip((c.astype(np.float32)-lo)/(hi-lo+1e-6), 0, 1)


def inputs_dev(x, device):
    x = x.to(device).expand(-1, 3, -1, -1); return F.pad(x, (0, 16, 0, 20), mode='replicate').contiguous()


def load_detector(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    net, _, _ = preflight.build('cpdino_vitb', False, native_convnext_norm=True); enc = net[0].original
    enc.load_state_dict(ck['encoder'], strict=True); enc = enc.eval().requires_grad_(False).to(device)
    head = SG.Head(seed=int(ck.get('seed', 46))); head.load_state_dict(ck['head']); head = head.eval().to(device)
    return enc, head


def detect(enc, head, img01, device, threshold=0.35):
    x = torch.from_numpy(img01)[None, None]
    with torch.no_grad():
        if device == 'cuda':
            with torch.autocast('cuda', dtype=torch.bfloat16):
                tok = enc.forward_features(inputs_dev(x, device))['x_norm_patchtokens']
        else:
            tok = enc.forward_features(inputs_dev(x, device))['x_norm_patchtokens']
        m = head(tok).sigmoid()[0].float().cpu().numpy()
    return SG.peaks2(m, SG.frame_support(), threshold)


def measure(gt, det, tissue_half=None):
    """miss, FP, jitter (x spacing), F1 at 12 px; gt and det restricted to tissue and an inner margin."""
    keep = (gt[:, 0] >= MARGIN) & (gt[:, 0] <= WID-1-MARGIN) & (gt[:, 1] >= MARGIN) & (gt[:, 1] <= HGT-1-MARGIN)
    kd = (det[:, 0] >= MARGIN) & (det[:, 0] <= WID-1-MARGIN) & (det[:, 1] >= MARGIN) & (det[:, 1] <= HGT-1-MARGIN)
    if tissue_half is not None:
        gi = tissue_half[np.clip((gt[:, 1]/2).astype(int), 0, tissue_half.shape[0]-1), np.clip((gt[:, 0]/2).astype(int), 0, tissue_half.shape[1]-1)]; keep &= gi
        di = tissue_half[np.clip((det[:, 1]/2).astype(int), 0, tissue_half.shape[0]-1), np.clip((det[:, 0]/2).astype(int), 0, tissue_half.shape[1]-1)]; kd &= di
    g = gt[keep]; d = det[kd]
    if len(g) < 10:
        return None
    sp = float(np.median(cKDTree(g).query(g, k=2)[0][:, 1]))
    out = dict(n_gt=int(len(g)), n_det=int(len(d)), spacing=sp)
    if len(d) == 0:
        out.update(miss=1.0, fp=0.0, jitter=np.nan, f1_12px=0.0); return out
    for r, key in ((0.5*sp, 'sp'), (12.0, '12')):
        dist, j = cKDTree(d).query(g); used = set(); tp = []
        for dd, jj in sorted(zip(dist, j)):
            if dd <= r and jj not in used: used.add(jj); tp.append(dd)
        if key == 'sp':
            out['miss'] = 1-len(tp)/len(g); out['fp'] = (len(d)-len(tp))/len(g); out['jitter'] = float(np.median(tp)/sp) if tp else np.nan
        else:
            p = len(tp)/len(d); rc = len(tp)/len(g); out['f1_12px'] = 2*p*rc/(p+rc) if p+rc else 0.0
    return out


def local_contrast(img01, k=25):
    x = img01.astype(np.float32); m = cv2.blur(x, (k, k)); m2 = cv2.blur(x*x, (k, k)); return np.sqrt(np.maximum(m2-m*m, 0))


def measure_cells(gt, det, img01, tissue_half=None):
    """Per-cell records for the contrast breakdown: GT cells (contrast, matched, matched distance / spacing) and detections (contrast, is_fp)."""
    keep = (gt[:, 0] >= MARGIN) & (gt[:, 0] <= WID-1-MARGIN) & (gt[:, 1] >= MARGIN) & (gt[:, 1] <= HGT-1-MARGIN)
    kd = (det[:, 0] >= MARGIN) & (det[:, 0] <= WID-1-MARGIN) & (det[:, 1] >= MARGIN) & (det[:, 1] <= HGT-1-MARGIN)
    if tissue_half is not None:
        keep &= tissue_half[np.clip((gt[:, 1]/2).astype(int), 0, tissue_half.shape[0]-1), np.clip((gt[:, 0]/2).astype(int), 0, tissue_half.shape[1]-1)]
        kd &= tissue_half[np.clip((det[:, 1]/2).astype(int), 0, tissue_half.shape[0]-1), np.clip((det[:, 0]/2).astype(int), 0, tissue_half.shape[1]-1)]
    g = gt[keep]; d = det[kd]
    if len(g) < 10:
        return None
    lc = local_contrast(img01); cg = lc[np.clip(g[:, 1].astype(int), 0, HGT-1), np.clip(g[:, 0].astype(int), 0, WID-1)]
    cd = lc[np.clip(d[:, 1].astype(int), 0, HGT-1), np.clip(d[:, 0].astype(int), 0, WID-1)] if len(d) else np.zeros(0)
    sp = float(np.median(cKDTree(g).query(g, k=2)[0][:, 1])); matched = np.zeros(len(g), bool); mdist = np.full(len(g), np.nan); dmatched = np.zeros(len(d), bool)
    if len(d):
        dist, j = cKDTree(d).query(g); used = set()
        for dd, jj, gi in sorted(zip(dist, j, range(len(g)))):
            if dd <= 0.5*sp and jj not in used: used.add(jj); matched[gi] = True; mdist[gi] = dd/sp; dmatched[jj] = True
    return dict(gt_contrast=cg, matched=matched, mdist=mdist, det_contrast=cd, det_matched=dmatched, spacing=sp)


def fp_geometry(det, det_matched, instance, tissue_half):
    """Where unmatched detections sit, from the exact instance map: junction (>= 3 labels in a 7x7 neighbourhood),
    border / chord (2 labels), interior (1 label, inside a cell), near_frame (within 30 px of black)."""
    out = dict(junction=0, border_or_chord=0, interior=0, near_frame=0, background=0, n_fp=0, in_large_cell=0, n_large_cells=0, n_cells=0)
    if len(det) == 0: return out
    dist_black = cv2.distanceTransform(np.pad(tissue_half.astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]*2.0
    areas = np.bincount(instance.ravel().astype(np.int64)); areas[0] = 0; med = np.median(areas[areas > 0]) if (areas > 0).any() else 1
    large = areas > 1.6*med; out['n_large_cells'] = int(large.sum()); out['n_cells'] = int((areas > 0).sum())
    for (x, y), m in zip(det, det_matched):
        if m: continue
        out['n_fp'] += 1; xi, yi = int(round(x)), int(round(y))
        if dist_black[min(max(yi//2, 0), tissue_half.shape[0]-1), min(max(xi//2, 0), tissue_half.shape[1]-1)] < 30: out['near_frame'] += 1; continue
        patch = instance[max(0, yi-3):yi+4, max(0, xi-3):xi+4]; labs = np.unique(patch); labs = labs[labs > 0]
        lab_here = int(instance[min(max(yi, 0), HGT-1), min(max(xi, 0), WID-1)])
        if lab_here > 0 and large[lab_here]: out['in_large_cell'] += 1
        if len(labs) >= 3: out['junction'] += 1
        elif len(labs) == 2: out['border_or_chord'] += 1
        elif len(labs) == 1: out['interior'] += 1
        else: out['background'] += 1
    return out


def contrast_breakdown(cells):
    gc = np.concatenate([c['gt_contrast'] for c in cells]); mt = np.concatenate([c['matched'] for c in cells]); md = np.concatenate([c['mdist'] for c in cells])
    dc = np.concatenate([c['det_contrast'] for c in cells]); dm = np.concatenate([c['det_matched'] for c in cells])
    q = np.quantile(gc, [1/3, 2/3]); tg = np.digitize(gc, q); td = np.digitize(dc, q)
    out = dict(contrast_tercile_edges=q.tolist(), n_gt=int(len(gc)), n_det=int(len(dc)))
    for t, name in enumerate(('low', 'mid', 'high')):
        out[name] = dict(gt_cells=int((tg == t).sum()), miss=float(1-mt[tg == t].mean()) if (tg == t).any() else None, jitter=float(np.nanmedian(md[(tg == t) & mt])) if ((tg == t) & mt).any() else None,
                         det=int((td == t).sum()), fp_of_dets=float(1-dm[td == t].mean()) if (td == t).any() else None, fp_rate_vs_gt=float((~dm[td == t]).sum()/max(1, (tg == t).sum())))
    return out


def synth_items(limit):
    metas = sorted(glob.glob(str(SYN/'meta'/'*.npz')))
    return metas[:limit] if limit else metas


def domain_check(a):
    device = 'cpu' if a.cpu else 'cuda'
    torch.set_num_threads(4 if a.cpu else 4)
    enc, head = load_detector(SHIPPED, device)
    rows = []; cells = []; dets = {}; fpgeo = {}; t0 = time.time()
    for k, mp in enumerate(synth_items(a.limit)):
        z = np.load(mp, allow_pickle=True); img = cv2.imread(mp.replace('/meta/', '/images/').replace('.npz', '.png'), cv2.IMREAD_UNCHANGED)
        tis, _ = D.tissue_mask(img[::2, ::2]); det = detect(enc, head, prep(img), device)
        img01 = prep(img); m = measure(z['centres'].astype(np.float64), det[:, :2].astype(np.float64), tis)
        if m is None: continue
        mc = measure_cells(z['centres'].astype(np.float64), det[:, :2].astype(np.float64), img01, tis); cells.append(mc); dets[os.path.basename(mp)] = det.astype(np.float32)
        if mc is not None and 'instance' in z.files:
            kd = (det[:, 0] >= MARGIN) & (det[:, 0] <= WID-1-MARGIN) & (det[:, 1] >= MARGIN) & (det[:, 1] <= HGT-1-MARGIN)
            kd &= tis[np.clip((det[:, 1]/2).astype(int), 0, tis.shape[0]-1), np.clip((det[:, 0]/2).astype(int), 0, tis.shape[1]-1)]
            fg = fp_geometry(det[kd, :2].astype(np.float64), mc['det_matched'], z['instance'], tis)
            for kk, vv in fg.items(): fpgeo[kk] = fpgeo.get(kk, 0)+vv
        prm = json.loads(str(z['params'])); m.update(framed=bool(prm.get('framed')), cd=float(z['labels'][0]), image=os.path.basename(mp)); rows.append(m)
        if k % 10 == 0 or a.cpu:
            log('domain', k=k, image=m['image'], miss=round(m['miss'], 3), fp=round(m['fp'], 3), jitter=round(m['jitter'], 3) if np.isfinite(m['jitter']) else None, f1=round(m['f1_12px'], 3), n_det=m['n_det'], n_gt=m['n_gt'], framed=m['framed'], seconds=round(time.time()-t0, 1))
    def agg(sel):
        return None if not sel else {k: float(np.nanmean([r[k] for r in sel])) for k in ('miss', 'fp', 'jitter', 'f1_12px', 'n_gt', 'n_det')} | dict(n=len(sel))
    cells = [c for c in cells if c is not None]
    res = dict(created_utc=G.now(), device=device, n=len(rows), all=agg(rows), framed=agg([r for r in rows if r['framed']]), unframed=agg([r for r in rows if not r['framed']]), by_local_contrast=contrast_breakdown(cells) if cells else None, fp_geometry=fpgeo,
               real_val5_reference=dict(miss=0.293, fp=0.188, jitter=0.195, f1=0.644, source='sg2 fine-tuned; pre-only s46: 0.166 / 0.166 / 0.222'), rows=rows)
    tag = ('_cpu' if a.cpu else '')+(('_'+a.tag) if a.tag else '')
    res['data'] = str(SYN)
    (OUT/f'domain_check{tag}.json').write_text(json.dumps(res, indent=1, default=float)); np.savez_compressed(OUT/f'domain_check{tag}_detections.npz', **dets)
    if res['by_local_contrast']: log('by local contrast', **{k: v for k, v in res['by_local_contrast'].items() if k in ('low', 'mid', 'high')})
    log('fp geometry', **fpgeo); log('domain check done', **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in res['all'].items()})


class SynthSet(torch.utils.data.Dataset):
    def __init__(self, metas): self.metas = metas
    def __len__(self): return len(self.metas)
    def __getitem__(self, i):
        mp = self.metas[i]; z = np.load(mp, allow_pickle=True); img = cv2.imread(mp.replace('/meta/', '/images/').replace('.npz', '.png'), cv2.IMREAD_UNCHANGED)
        tis, _ = D.tissue_mask(img[::2, ::2]); c = z['centres'].astype(np.float64)
        gi = tis[np.clip((c[:, 1]/2).astype(int), 0, tis.shape[0]-1), np.clip((c[:, 0]/2).astype(int), 0, tis.shape[1]-1)]
        t, sup = SG.target2(c[gi], [0, 0, WID-1, HGT-1])
        return torch.from_numpy(prep(img))[None], torch.from_numpy(t), torch.from_numpy(sup)


def train(a):
    assert torch.cuda.device_count() == 1; torch.set_num_threads(4)
    metas = synth_items(0); log('train start', n_images=len(metas))
    encoder = G.load_encoder(ST.ENCODERS[4]).train().requires_grad_(True); head = SG.Head(seed=46).cuda()
    opt = torch.optim.AdamW([{'params': encoder.parameters(), 'lr': 1e-5}, {'params': head.parameters(), 'lr': 1e-3}], weight_decay=1e-4)
    dl = torch.utils.data.DataLoader(SynthSet(metas), batch_size=2, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    rng = np.random.default_rng(46); step = 0; losses = []; t0 = time.time()
    while step < a.updates:
        for x, t, sup in dl:
            f = int(rng.integers(4)); h, v = G.FLIPS[f]
            if h: x, t, sup = x.flip(-1), t.flip(-1), sup.flip(-1)
            if v: x, t, sup = x.flip(-2), t.flip(-2), sup.flip(-2)
            x = x.cuda(non_blocking=True); t = t.cuda(non_blocking=True); sup = sup.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                tok = encoder.forward_features(G.b.inputs(x, 'cpdino_vitb'))['x_norm_patchtokens']
            logits = head(tok)
            loss = sum(F.binary_cross_entropy_with_logits(logits[i][sup[i]], t[i][sup[i]]) for i in range(len(t)))/len(t)
            loss.backward(); torch.nn.utils.clip_grad_norm_(list(encoder.parameters())+list(head.parameters()), 1.0, error_if_nonfinite=True); opt.step()
            losses.append(float(loss)); step += 1
            if step % 200 == 0 or step == 1:
                log('train', step=step, of=a.updates, loss=round(float(loss), 4), seconds=round(time.time()-t0, 1))
            if step >= a.updates: break
    encoder.eval().requires_grad_(False); head.eval()
    torch.save(dict(encoder=encoder.state_dict(), head=head.state_dict(), kind='sg2', features='last', seed=46, threshold=0.35, init_encoder=str(ST.ENCODERS[4]), updates=a.updates, mode='synthetic-only (route B), no real point labels'), OUT/'detector_synth.pt')
    (OUT/'train.json').write_text(json.dumps(dict(created_utc=G.now(), updates=a.updates, n_images=len(metas), loss_last200=float(np.mean(losses[-200:])), seconds=round(time.time()-t0, 1)), indent=1)); log('train done', loss_last200=float(np.mean(losses[-200:])), seconds=round(time.time()-t0, 1))


def score25(a):
    device = 'cuda'; torch.set_num_threads(4)
    ann = json.loads(Path('datasets/clear_ec_official/green_eval/annotations.json').read_text()); binding = json.loads(Path('experiments/green_point_probe_20261023/data_binding.json').read_text()); train_ids = set(binding['train_ids'])
    with np.load('datasets/clear_ec_official/cache_native_meta_512.npz', allow_pickle=True) as z:
        ids = z['ids'].astype(str)
    pos = {n: i for i, n in enumerate(ids)}; mm = np.memmap(D.RAW, dtype=np.uint8, mode='r', shape=(8954, HGT, WID))
    dets = {'synth': Path(a.ckpt), 'shipped_pre_s46': SHIPPED}
    res = dict(created_utc=G.now(), detectors={})
    for name, path in dets.items():
        enc, head = load_detector(path, device); rows = {}
        for n, e in ann.items():
            r = pos[n]; img = np.asarray(mm[r]); tis, _ = D.tissue_mask(img[::2, ::2]); det = detect(enc, head, prep(img), device)
            gt = np.asarray(e['markers_mha'], float); hull = Delaunay(gt); sp = float(np.median(cKDTree(gt).query(gt, k=2)[0][:, 1]))
            dmin, _ = cKDTree(gt).query(det[:, :2]) if len(det) else (np.zeros(0), None)
            inside = (hull.find_simplex(det[:, :2]) >= 0) | (dmin <= 0.5*sp) if len(det) else np.zeros(0, bool)
            m = measure(gt, det[inside, :2].astype(np.float64), None)
            if m is None: continue
            m.update(val=n not in train_ids); rows[n] = m
        def agg(sel):
            return None if not sel else {k: float(np.nanmean([r[k] for r in sel])) for k in ('miss', 'fp', 'jitter', 'f1_12px')} | dict(n=len(sel))
        res['detectors'][name] = dict(val5=agg([v for v in rows.values() if v['val']]), all25=agg(list(rows.values())), per_image=rows)
        log('score25', detector=name, val5=res['detectors'][name]['val5'], all25=res['detectors'][name]['all25'])
    s = res['detectors']['synth']['all25']; res['gate'] = dict(passed=bool((s['miss'] <= 0.15 or s['jitter'] <= 0.17) and s['fp'] <= 0.30), rule='all-25: miss <= 0.15 or jitter <= 0.17, with FP <= 0.30', e4_break_even=dict(miss=0.08, jitter=0.17, fp=0.14))
    (OUT/'score25.json').write_text(json.dumps(res, indent=1, default=float)); log('score25 done', gate=res['gate'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['domain-check', 'train', 'score25']); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--cpu', action='store_true'); ap.add_argument('--updates', type=int, default=4000); ap.add_argument('--ckpt', default=str(OUT/'detector_synth.pt')); ap.add_argument('--tag', default='')
    a = ap.parse_args(); {'domain-check': domain_check, 'train': train, 'score25': score25}[a.command](a)
