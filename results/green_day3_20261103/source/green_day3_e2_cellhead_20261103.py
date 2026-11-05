"""Green day 3 / E2 (pre-registered 2026-11-03): learned per-cell geometry from local appearance.
  labels   (CPU, 4 workers)  expert-point Voronoi labels (hull-bounded cells): srep Topcon 307 images + the 20 train-green
                             images; 96x96 patches around each labelled point; the 5 validation-green images kept apart.
  train    (GPU)             small CNN: P(six-sided) + log area; flips/rotations/intensity jitter; green patches x5.
  infer    (GPU)             per fold-clean OOF detector points (sg2_s46 compact, sg2pre_s46 full frame): P(six), log area.
  evaluate (CPU)             per-image hex_learned / cv_learned; LOFO on B for HEX and CV, stratified; member test in the
                             active ten (HEX); sanity on the 5 validation-green images.
"""
import argparse, json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
import multiprocessing as mp
from pathlib import Path
import numpy as np
import cv2
cv2.setNumThreads(1)
from scipy.spatial import Voronoi, Delaunay

sys.path.insert(0, 'experiments/green_track_20261028/source'); sys.path.insert(0, 'experiments/eval_harness_20261031/source'); sys.path.insert(0, 'experiments/green_day2_20261101/source'); sys.path.insert(0, 'experiments/green_day3_20261103/source')
import cpu_pool
ROOT = Path('experiments/green_day3_20261103'); OUT = ROOT/'e2_cellhead'
GT = Path('experiments/green_track_20261028')
RAW = Path('datasets/clear_ec_fullframe_20260904/img_full_raw.dat'); N_TOTAL, HGT, WID = 8954, 972, 1296
ANN = Path('datasets/clear_ec_official/green_eval/annotations.json'); BINDING = Path('experiments/green_point_probe_20261023/data_binding.json')
META = Path('datasets/clear_ec_official/cache_native_meta_512.npz')
PATCH = 96; HALF = PATCH//2
FAMILIES = {'sg2_s46': GT/'sg2_specialist', 'sg2pre_s46': GT/'external_srep/preonly'}
BOXES = ROOT/'e3_boxpool/cluster_boxes.npz'


def log(msg, **kw):
    print(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), msg=msg, **kw)), flush=True)


def poly_area(v):
    x, y = v[:, 0], v[:, 1]; return 0.5*abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def cell_labels(pts):
    """Hull-bounded Voronoi cells of expert points: (point index, side count, area px^2)."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    if len(pts) < 8:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    vor = Voronoi(pts); tri = Delaunay(pts); idx, sides, areas = [], [], []
    for i in range(len(pts)):
        r = vor.regions[vor.point_region[i]]
        if -1 in r or len(r) == 0:
            continue
        v = vor.vertices[r]
        if (tri.find_simplex(v) < 0).any():
            continue
        idx.append(i); sides.append(len(r)); areas.append(poly_area(v))
    return np.array(idx, int), np.array(sides, int), np.array(areas, float)


def prep_frame(a_u8):
    """CLAHE + 1/99 percentile scaling exactly as the regression / detector frame pipeline (b.Frames)."""
    c = cv2.createCLAHE(2.0, (WID//32, HGT//32)).apply(np.ascontiguousarray(a_u8))
    lo, hi = np.percentile(c, (1, 99))
    return np.clip((c.astype(np.float32)-lo)/(hi-lo+1e-6), 0, 1)


def crops(img, pts):
    """96x96 patches (uint8) centred on pts from a float [0,1] image, zero-padded at the borders."""
    pad = np.pad(img, HALF, mode='constant')
    out = np.zeros((len(pts), PATCH, PATCH), np.uint8)
    for k, (x, y) in enumerate(pts):
        cx, cy = int(round(x))+HALF, int(round(y))+HALF
        out[k] = np.clip(pad[cy-HALF:cy+HALF, cx-HALF:cx+HALF]*255, 0, 255).astype(np.uint8)
    return out


def labels(a):
    import green_external_srep_20261030 as E
    OUT.mkdir(parents=True, exist_ok=True)
    imgs, pts, masks, shape = E.load_cache(E.SCALE_NOMINAL)
    P, S, A, SRC, IMG = [], [], [], [], []
    for i in range(len(imgs)):
        idx, sides, areas = cell_labels(pts[i])
        if len(idx) == 0:
            continue
        im = np.asarray(imgs[i], np.float32)
        P.append(crops(im, pts[i][idx])); S.append(sides); A.append(areas); SRC.append(np.zeros(len(idx), np.int8)); IMG.append(np.full(len(idx), i, np.int32))
    log('srep labels', images=len(imgs), cells=int(sum(len(s) for s in S)), img_mean=float(np.mean([np.asarray(x, np.float32).mean() for x in imgs[:20]])))
    ann = json.loads(ANN.read_text()); binding = json.loads(BINDING.read_text())
    with np.load(META, allow_pickle=True) as z:
        ids = z['ids'].astype(str)
    pos = {n: i for i, n in enumerate(ids)}
    mm = np.memmap(RAW, dtype=np.uint8, mode='r', shape=(N_TOTAL, HGT, WID))
    train_ids = set(binding['train_ids']); val = {}
    gm = []
    for name, e in ann.items():
        r = pos[name]; im = prep_frame(mm[r]); gm.append(im.mean())
        p = np.asarray(e['markers_mha'], float); idx, sides, areas = cell_labels(p)
        if name in train_ids:
            P.append(crops(im, p[idx])); S.append(sides); A.append(areas); SRC.append(np.ones(len(idx), np.int8)); IMG.append(np.full(len(idx), 1000+r, np.int32))
        else:
            val[name] = dict(patches=crops(im, p[idx]), sides=sides, areas=areas, row=r)
    log('green labels', train_images=len(train_ids), val_images=len(val), green_cells=int(sum(len(s) for s, src in zip(S, SRC) if src[0] == 1)), img_mean=float(np.mean(gm)))
    np.savez_compressed(OUT/'labels.npz', patches=np.concatenate(P), sides=np.concatenate(S), area=np.concatenate(A), source=np.concatenate(SRC), image=np.concatenate(IMG))
    np.savez_compressed(OUT/'labels_val5.npz', names=np.array(list(val)), **{f'{k}_{n}': v[k] for n, v in val.items() for k in ('patches', 'sides', 'areas')}, rows=np.array([v['row'] for v in val.values()]))
    log('labels done', total=int(sum(len(s) for s in S)))


def build_net():
    import torch.nn as nn
    def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
    return nn.Sequential(blk(1, 32), blk(32, 32), nn.MaxPool2d(2), blk(32, 64), blk(64, 64), nn.MaxPool2d(2), blk(64, 128), blk(128, 128), nn.MaxPool2d(2),
                         blk(128, 256), nn.MaxPool2d(2), blk(256, 256), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(128, 2))


def augment(x, g):
    """x: (B,1,96,96) float tensor on GPU; random flips / rot90 / intensity jitter / small shift."""
    import torch
    B = x.shape[0]
    if torch.rand(1, generator=g).item() < 0.5: x = x.flip(-1)
    if torch.rand(1, generator=g).item() < 0.5: x = x.flip(-2)
    k = int(torch.randint(0, 4, (1,), generator=g)); x = torch.rot90(x, k, (-2, -1))
    gain = (0.8+0.4*torch.rand(B, 1, 1, 1, generator=g)).to(x.device); bias = (0.1*(torch.rand(B, 1, 1, 1, generator=g)-0.5)).to(x.device)
    x = (x*gain+bias).clamp(0, 1)
    dx, dy = [int(v) for v in torch.randint(-4, 5, (2,), generator=g)]
    return torch.roll(x, (dy, dx), (-2, -1))


def train(a):
    import torch, torch.nn.functional as F
    torch.manual_seed(46); g = torch.Generator().manual_seed(46)
    z = np.load(OUT/'labels.npz'); X = z['patches']; S = z['sides']; A = z['area']; SRC = z['source']; IM = z['image']
    la = np.log(A); m_la, s_la = float(la.mean()), float(la.std())
    rng = np.random.default_rng(46); srep_imgs = np.unique(IM[SRC == 0]); hold = set(rng.choice(srep_imgs, max(1, len(srep_imgs)//10), replace=False).tolist())
    is_val = np.array([(src == 0 and im in hold) for src, im in zip(SRC, IM)])
    tr = np.where(~is_val)[0]; va = np.where(is_val)[0]
    tr = np.concatenate([tr]+[tr[SRC[tr] == 1]]*4)                       # green x5
    net = build_net().cuda(); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xg = torch.from_numpy(X).cuda(); y6 = torch.from_numpy((S == 6).astype(np.float32)).cuda(); ya = torch.from_numpy(((la-m_la)/s_la).astype(np.float32)).cuda()
    steps = a.epochs*(len(tr)//a.batch); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    t0 = time.time(); step = 0; hist = []
    for ep in range(a.epochs):
        net.train(); perm = rng.permutation(tr); tot = np.zeros(3)
        for i in range(0, len(perm)-a.batch+1, a.batch):
            b = torch.from_numpy(perm[i:i+a.batch]).cuda()
            x = augment(Xg[b].float().unsqueeze(1)/255, g)
            out = net(x); l6 = F.binary_cross_entropy_with_logits(out[:, 0], y6[b]); larea = (out[:, 1]-ya[b]).abs().mean()
            loss = l6+larea; opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
            tot += [float(l6), float(larea), 1]
        net.eval()
        with torch.no_grad():
            pv = torch.cat([net(Xg[torch.from_numpy(va[i:i+1024]).cuda()].float().unsqueeze(1)/255) for i in range(0, len(va), 1024)])
            acc = float(((pv[:, 0] > 0).float() == y6[torch.from_numpy(va).cuda()]).float().mean()); r = float(np.corrcoef(pv[:, 1].cpu().numpy(), ya[torch.from_numpy(va).cuda()].cpu().numpy())[0, 1])
        hist.append(dict(epoch=ep+1, loss_six=tot[0]/tot[2], loss_area=tot[1]/tot[2], val_six_acc=acc, val_area_r=r))
        log('train', epoch=ep+1, loss_six=round(tot[0]/tot[2], 4), loss_area=round(tot[1]/tot[2], 4), val_six_acc=round(acc, 4), val_area_r=round(r, 4), seconds=round(time.time()-t0, 1))
    # sanity on the 5 validation-green images (expert points)
    zv = np.load(OUT/'labels_val5.npz', allow_pickle=True); san = {}
    with torch.no_grad():
        for n in zv['names']:
            xp = torch.from_numpy(zv[f'patches_{n}']).cuda().float().unsqueeze(1)/255; o = net(xp).cpu().numpy()
            s6 = zv[f'sides_{n}'] == 6; ar = np.log(zv[f'areas_{n}'])
            san[str(n)] = dict(n=int(len(s6)), six_acc=float(((o[:, 0] > 0) == s6).mean()), six_base_rate=float(s6.mean()), area_r=float(np.corrcoef(o[:, 1], ar)[0, 1]) if len(ar) > 3 else None,
                               hex6_expert=float(s6.mean()), hex6_pred_mean_p=float((1/(1+np.exp(-o[:, 0]))).mean()), cv_expert=float(np.exp(ar).std()/np.exp(ar).mean()), cv_pred=float(np.exp(o[:, 1]*s_la+m_la).std()/np.exp(o[:, 1]*s_la+m_la).mean()))
    torch.save(dict(model=net.state_dict(), m_la=m_la, s_la=s_la, epochs=a.epochs, note='green day 3 E2 per-cell geometry CNN (srep + 20 train-green)'), OUT/'model.pt')
    res = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), n_train=int(len(tr)), n_val_srep=int(len(va)), m_la=m_la, s_la=s_la, history=hist, val5_sanity=san,
               val5_mean_six_acc=float(np.mean([v['six_acc'] for v in san.values()])), val5_mean_area_r=float(np.mean([v['area_r'] for v in san.values() if v['area_r'] is not None])))
    (OUT/'train_result.json').write_text(json.dumps(res, indent=1)); log('train done', val5_six_acc=res['val5_mean_six_acc'], val5_area_r=res['val5_mean_area_r'])


class PatchSet:
    """Per image: CLAHE frame + crops at the detector points (CPU work for the DataLoader workers)."""
    def __init__(self, items):
        self.items = items; self.mm = None
    def __len__(self): return len(self.items)
    def __getitem__(self, k):
        if self.mm is None:
            self.mm = np.memmap(RAW, dtype=np.uint8, mode='r', shape=(N_TOTAL, HGT, WID))
        r, pts = self.items[k]
        im = prep_frame(self.mm[r]); return r, pts.astype(np.float32), crops(im, pts[:, :2])


def infer(a):
    import torch
    from torch.utils.data import DataLoader
    ck = torch.load(OUT/'model.pt', map_location='cpu', weights_only=False); net = build_net().cuda(); net.load_state_dict(ck['model']); net.eval()
    import green_day2_cd_features_20261101 as D
    for fam in a.families:
        pk, seen = D.load_peaks(FAMILIES[fam])
        for f in range(5):
            out = OUT/'infer'/fam; out.mkdir(parents=True, exist_ok=True)
            if (out/f'fold{f}.npz').exists():
                continue
            with np.load(FAMILIES[fam]/f'fold{f}'/f'fold{f}_sg2_peaks.npz') as z:
                rows = z['indices']
            items = []
            for r in rows:
                p = pk[int(r)]; p = p[p[:, 2] >= 0.35] if p is not None else np.zeros((0, 3))
                items.append((int(r), p))
            dl = DataLoader(PatchSet(items), batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda b: b[0], prefetch_factor=4)
            R_, O_, P_ = [], [0], []; preds = []
            t0 = time.time()
            with torch.no_grad():
                for k, (r, pts, X) in enumerate(dl):
                    if len(X):
                        x = torch.from_numpy(X).cuda().float().unsqueeze(1)/255
                        o = torch.cat([net(x[i:i+2048]) for i in range(0, len(x), 2048)]).cpu().numpy()
                    else:
                        o = np.zeros((0, 2), np.float32)
                    R_.append(r); P_.append(pts); preds.append(o.astype(np.float32)); O_.append(O_[-1]+len(pts))
                    if k % 500 == 0:
                        log('infer', family=fam, fold=f, image=k, of=len(items), seconds=round(time.time()-t0, 1))
            np.savez_compressed(out/f'fold{f}.npz', rows=np.array(R_), offsets=np.array(O_), points=np.concatenate(P_) if P_ else np.zeros((0, 3), np.float32), pred=np.concatenate(preds), m_la=ck['m_la'], s_la=ck['s_la'])
            log('infer done', family=fam, fold=f, images=len(items), seconds=round(time.time()-t0, 1))


_EV = {}


def _eval_one(i):
    import green_day2_cd_features_20261101 as D
    rows, off, pts, pred, m_la, s_la, box = (_EV[k] for k in ('rows', 'off', 'pts', 'pred', 'm_la', 's_la', 'box'))
    q = pts[off[i]:off[i+1]]; o = pred[off[i]:off[i+1]]; r = int(rows[i])
    if len(q) < 12:
        return r, None
    p6 = 1/(1+np.exp(-o[:, 0])); area = np.exp(o[:, 1]*s_la+m_la)
    x0, y0, x1, y1 = box[r]; inb = (q[:, 0] >= x0) & (q[:, 0] < x1) & (q[:, 1] >= y0) & (q[:, 1] < y1)
    sel = inb if inb.sum() >= 12 else np.ones(len(q), bool)
    lo, hi = np.percentile(area[sel], (5, 95)); t = sel & (area >= lo) & (area <= hi)
    loa, hia = np.percentile(area, (5, 95)); ta = (area >= loa) & (area <= hia)
    vs = D.voronoi_stats(q[:, :2].astype(np.float64), np.full(len(q), 1e9))
    return r, dict(hex_mean_p=p6[sel].mean(), hex_frac=(p6[sel] > 0.5).mean(), hex_mean_p_all=p6.mean(), cv_learned=(area[t].std()/area[t].mean() if t.sum() >= 8 else np.nan),
                   cv_learned_all=area[ta].std()/area[ta].mean(), n_cells=len(q), hex6_geom=vs['hex6_all'])


def evaluate(a):
    import eval_oof_20261031 as H
    import cd_estimators_20261028 as P
    import green_day2_cd_features_20261101 as D
    base = H.load_base(); y, fold, B, idx = base['y'], base['fold'], base['B'], base['idx']
    yr, Br, fr = np.zeros((N_TOTAL, 3)), np.zeros((N_TOTAL, 3)), np.zeros(N_TOTAL, int); yr[idx] = y; Br[idx] = B; fr[idx] = fold
    st = H.image_stats(base); framed = np.zeros(N_TOTAL, int); framed[idx] = (st['black_frac'] > 0.01)
    box = np.load(BOXES)['box']
    with np.load('experiments/green_day2_20261101/d2b_active_gate/active_family_features.npz') as z:
        seen = z['seen']; hx = {k.replace('hex6_', ''): z[k] for k in z.files if k.startswith('hex6_')}
    ok = ~seen
    feats = {}
    for fam in a.families:
        F_ = {k: np.full(N_TOTAL, np.nan) for k in ('hex_mean_p', 'hex_frac', 'cv_learned', 'cv_learned_all', 'hex_mean_p_all', 'n_cells', 'hex6_geom')}
        for f in range(5):
            p = OUT/'infer'/fam/f'fold{f}.npz'
            if not p.exists():
                continue
            with np.load(p) as z:
                rows, off, pts, pred, m_la, s_la = z['rows'], z['offsets'], z['points'], z['pred'], float(z['m_la']), float(z['s_la'])
            _EV.update(rows=rows, off=off, pts=pts, pred=pred, m_la=m_la, s_la=s_la, box=box)
            for r, out in cpu_pool.pmap(_eval_one, range(len(rows)), chunksize=32):
                if out is not None:
                    for k, v in out.items():
                        F_[k][r] = v
        feats[fam] = F_
    np.savez_compressed(OUT/'features.npz', **{f'{fam}_{k}': v for fam, F_ in feats.items() for k, v in F_.items()})
    res = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), hex={}, cv={}, member={})
    D.STRATA_NAMES = {0: 'unframed', 1: 'framed'}
    def run(name, f, t):
        r = P.lofo_blend(yr, Br, fr, ok, np.where(np.isfinite(f) & (f > 0), f, np.nan), t=t); s, _ = D.lofo_strat(yr, Br, fr, ok, f, framed, t=t)
        d = dict(delta_mean_pp=r['overall']['delta_cd_pp']/3, delta_target_pp=r['overall']['delta_cd_pp'], folds_improving=r['overall']['folds_improving'], w=[r[str(h)]['w'] for h in range(5)], est_mape=r['overall']['est_cd_mape'], err_corr=r['overall']['err_corr_mean'],
                 strat_mean_pp=s['overall']['delta_mean_pp'], per_stratum_pp=s['overall']['per_stratum_pp'], dev_pp=s['overall']['dev_folds0to3_pp'], fold4_pp=s['overall']['fold4_pp'])
        print('%-32s %s plain %.4f mean pp (impr %d, est %.2f, r_err %.2f, w %s) | strat %.4f framed %.4f unframed %.4f | dev %.4f f4 %.4f' % (name, 'HEX' if t == 2 else 'CV', d['delta_mean_pp'], d['folds_improving'], d['est_mape'], d['err_corr'], d['w'], d['strat_mean_pp'], d['per_stratum_pp'].get('framed', np.nan), d['per_stratum_pp'].get('unframed', np.nan), d['dev_pp'], d['fold4_pp']), flush=True)
        return d
    for fam, F_ in feats.items():
        for k in ('hex_mean_p', 'hex_frac', 'hex_mean_p_all', 'hex6_geom'):
            res['hex'][f'{fam}_{k}'] = run(f'{fam}_{k}', F_[k], 2)
        for k in ('cv_learned', 'cv_learned_all'):
            res['cv'][f'{fam}_{k}'] = run(f'{fam}_{k}', F_[k], 1)
    # HEX member test in the active ten (deployed features)
    def blend(fams, fe):
        A = np.log(np.stack([fe[k] for k in fams]))
        with np.errstate(all='ignore'):
            f = np.exp(np.nanmean(A, 0))
        r = P.lofo_blend(yr, Br, fr, ok, f, t=2); o_ = r['overall']; return dict(delta_mean_pp=o_['delta_cd_pp']/3, folds_improving=o_['folds_improving'], per_fold_mean_pp=[r[str(h)]['delta_cd_pp']/3 for h in range(5)])
    ACTIVE = ['s8_s46', 's8_s47', 's8_s48', 'sg2_s46', 'sg2_s47', 'sg2_s48', 'sg2_s49', 'sg2_s50', 's8multi_s47', 'sg2pre_s46']
    fe = dict(hx)
    for fam, F_ in feats.items():
        fe[f'learned_{fam}'] = F_['hex_mean_p']
    res['member']['control_ten'] = blend(ACTIVE, fe)
    for fam in feats:
        res['member'][f'ten_plus_learned_{fam}'] = blend(ACTIVE+[f'learned_{fam}'], fe)
    res['member']['ten_plus_all_learned'] = blend(ACTIVE+[f'learned_{fam}' for fam in feats], fe)
    for k, v in res['member'].items():
        print('member %-28s %.4f mean pp impr %d per fold %s' % (k, v['delta_mean_pp'], v['folds_improving'], [round(x, 4) for x in v['per_fold_mean_pp']]), flush=True)
    c = res['member']['control_ten']['delta_mean_pp']
    res['gates'] = {k: dict(gain=v['delta_mean_pp']-c, evidence=bool(v['delta_mean_pp'] <= c-0.005 and v['per_fold_mean_pp'][4] < 0 and v['folds_improving'] >= 4), adoption=bool(v['delta_mean_pp'] <= c-0.02 and v['per_fold_mean_pp'][4] < 0 and v['folds_improving'] >= 4)) for k, v in res['member'].items() if k != 'control_ten'}
    res['gates'].update({f'cv_{k}': dict(evidence=bool(v['delta_mean_pp'] <= -0.005 and v['folds_improving'] >= 4), adoption=bool(v['delta_mean_pp'] <= -0.02 and v['folds_improving'] >= 4)) for k, v in res['cv'].items()})
    tr = json.loads((OUT/'train_result.json').read_text()); res['val5_sanity'] = dict(six_acc=tr['val5_mean_six_acc'], area_r=tr['val5_mean_area_r'], per_image=tr['val5_sanity'])
    (OUT/'result.json').write_text(json.dumps(res, indent=1, default=float)); print('GATES', json.dumps({k: v for k, v in res['gates'].items() if v.get('evidence') or v.get('adoption')}), flush=True); log('evaluate done')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['labels', 'train', 'infer', 'evaluate'])
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--batch', type=int, default=256); ap.add_argument('--families', nargs='*', default=list(FAMILIES))
    a = ap.parse_args(); {'labels': labels, 'train': train, 'infer': infer, 'evaluate': evaluate}[a.command](a)
