"""GREEN self-training pilot (one GPU): teacher = the fold's point specialist; pseudo-points on every
training image of the fold; student = fresh copy of the fold's recipe A encoder + linear head trained
on strong labels (20 train-green images, ROI-masked Gaussian BCE, four exact flips) and pseudo labels
(confident teacher peaks as Gaussian targets, confident teacher negatives as zeros, the uncertain band
ignored).  The 5 validation-green images never enter any loss or threshold choice.

commands
  teacher-infer : single-view teacher probability maps for the fold's TRAIN rows (fp16 npz)
  thresholds    : pseudo-label thresholds from the 20 train-green images only (pre-registered rule)
  student-train : fixed-update mixed-batch training; writes student.pt
  eval          : val-5 F1 at the fixed threshold for teacher and student, student maps on the fold's
                  validation rows (same format as green_point_features extract), calibration-free CD
                  diagnostics teacher vs student on unseen validation rows.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import green_point_features_20261027 as G
import green_point_spatial_20261023 as spatial
import green_point_probe_head_20261023 as probe
from scipy.ndimage import maximum_filter, uniform_filter
from scipy.spatial import cKDTree

ROOT = Path('experiments/green_track_20261028')
ENCODERS = {0: 'experiments/phase2_pooling_replication_20261026/f0_s46_std/runs/fold0/cpdino_vitb/epoch_20.pt',
            1: 'experiments/oof_recipeA_20261027/fold1/regression_std_s46/runs/fold1/cpdino_vitb/epoch_20.pt',
            2: 'experiments/oof_recipeA_20261027/fold2/regression_std_s46_v2/runs/fold2/cpdino_vitb/epoch_20.pt',
            3: 'experiments/oof_recipeA_20261027/fold3/regression_std_s46/runs/fold3/cpdino_vitb/epoch_20.pt',
            4: 'experiments/phase2_pooling_20261020/arms/std/runs/fold4/cpdino_vitb/epoch_20.pt'}
SPLITS = {0: 'experiments/ssl_screen_20260911/preflight/fold0.npz', 1: 'experiments/ssl_screen_20260911/preflight/fold1.npz',
          2: 'experiments/overnight_target_20260912/preflight/fold2.npz', 3: 'experiments/overnight_target_20260912/preflight/fold3.npz',
          4: 'experiments/target_fold4_20260912/preflight/fold4.npz'}
TEACHERS = {f: f'experiments/green_point_features_20261027/specialist/fold{f}_s46/specialist.pt' for f in range(4)}
TEACHERS[4] = 'experiments/green_point_features_20261027/specialist/specialist_fold4.pt'
FRAME_ROI = [0.0, 0.0, G.WIDTH-1.0, G.HEIGHT-1.0]
EVAL_THRESHOLD = 0.35
TAU_NEG = 0.10
PRECISION_TARGET = 0.80
GRID_ROWS, GRID_COLS = 122, 162
WIN = 37


def log(msg, **kw):
    print(json.dumps(dict(utc=G.now(), msg=msg, **kw)), flush=True)


def load_specialist(path, init):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    enc = G.load_encoder(init)
    enc.load_state_dict(ck['encoder'], strict=True)
    head = G.build_head(768, 'linear')
    head.load_state_dict(ck['head'])
    return enc.eval().requires_grad_(False), head.eval().requires_grad_(False)


def maps(encoder, head, x):
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        tok = encoder.forward_features(G.b.inputs(x, 'cpdino_vitb'))['x_norm_patchtokens']
    with torch.no_grad():
        return head(tok.float()).squeeze(-1).sigmoid().reshape(len(x), *G.GRID).cpu().numpy()


def infer_rows(encoder, head, rows, y, n_total, workers=3):
    dl = DataLoader(G.b.Frames(rows, y, n_total), batch_size=4, shuffle=False, num_workers=workers, pin_memory=True)
    probs, seen = [], []
    t0 = time.time()
    for i, (x, _, ix) in enumerate(dl):
        probs.append(maps(encoder, head, x).astype(np.float16))
        seen.append(ix.numpy())
        if i % 200 == 0:
            log('infer', batch=i, of=len(dl), seconds=round(time.time()-t0, 1))
    seen = np.concatenate(seen)
    np.testing.assert_array_equal(seen, rows)
    return np.concatenate(probs)


def green(fold):
    ann = json.loads(G.ANN.read_text())
    binding = json.loads(G.BINDING.read_text())
    with np.load(G.b.META, allow_pickle=True) as z:
        ids, y = z['ids'].astype(str), z['Y'].astype(np.float32)
    with np.load(SPLITS[fold]) as z:
        tr, va = z['train'].copy(), z['validation'].copy()
    return ann, binding, ids, y, tr, va


def rasterise(points, support_fn, grid=G.GRID):
    """Gaussian max-target of points on the grid; support from support_fn(target)."""
    yy, xx = np.indices(grid, dtype=np.float64)
    xx = spatial.OFFSET+spatial.STRIDE*xx
    yy = spatial.OFFSET+spatial.STRIDE*yy
    target = np.zeros(grid, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    for px, py in pts:
        x0, x1 = max(0, int((px-32-spatial.OFFSET)//spatial.STRIDE)), min(grid[1], int((px+32-spatial.OFFSET)//spatial.STRIDE)+2)
        y0, y1 = max(0, int((py-32-spatial.OFFSET)//spatial.STRIDE)), min(grid[0], int((py+32-spatial.OFFSET)//spatial.STRIDE)+2)
        if x1 <= x0 or y1 <= y0:
            continue
        g = np.exp(-((xx[y0:y1, x0:x1]-px)**2+(yy[y0:y1, x0:x1]-py)**2)/(2*spatial.SIGMA**2))
        target[y0:y1, x0:x1] = np.maximum(target[y0:y1, x0:x1], g)
    return target.astype(np.float32), support_fn(target, xx, yy)


def teacher_peaks(prob, threshold):
    """Vectorised peaks on the in-frame grid: local maxima within +-2 grid cells (+-16 px, the probe NMS
    radius) at or above the threshold.  Pixel coordinates of patch centres, deterministic order."""
    p = prob.astype(np.float32)
    peak = (p == maximum_filter(p, size=5, mode='nearest')) & (p >= threshold)
    peak[GRID_ROWS:, :] = False
    peak[:, GRID_COLS:] = False
    yy, xx = np.nonzero(peak)
    return np.stack([spatial.OFFSET+spatial.STRIDE*xx, spatial.OFFSET+spatial.STRIDE*yy], 1).astype(np.float64).reshape(-1, 2)


def pseudo_item(prob, flip, tau_pos, tau_neg):
    """Pseudo target+support for one training image under one of the four exact flips."""
    pts = teacher_peaks(prob, tau_pos)
    h, v = G.FLIPS[flip]
    pts_f, _ = G.flip_annotation(pts if len(pts) else np.zeros((0, 2)), FRAME_ROI, h, v)
    # negatives: grid cells whose centre maps back to a teacher value <= tau_neg
    def support_fn(target, xx, yy):
        bx = (G.WIDTH-1-xx) if h else xx
        by = (G.HEIGHT-1-yy) if v else yy
        ix = np.clip(np.rint((bx-spatial.OFFSET)/spatial.STRIDE).astype(int), 0, G.GRID[1]-1)
        iy = np.clip(np.rint((by-spatial.OFFSET)/spatial.STRIDE).astype(int), 0, G.GRID[0]-1)
        back = prob.astype(np.float32)[iy, ix]
        inframe = (bx <= G.WIDTH-1) & (by <= G.HEIGHT-1)
        return inframe & ((target >= 0.05) | (back <= tau_neg))
    target, support = rasterise(pts_f, support_fn)
    return target, support, len(pts)


def strong_items(ann, names):
    out = {}
    for n in names:
        for f, (h, v) in enumerate(G.FLIPS):
            pts, roi = G.flip_annotation(ann[n]['markers_mha'], ann[n]['roi_box_mha'], h, v)
            t, s = spatial.point_target(pts, roi, G.GRID)
            out[(n, f)] = (t, s)
    return out


def cmd_teacher_infer(a):
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    out = ROOT/a.tag
    out.mkdir(parents=True, exist_ok=True)
    ann, binding, ids, y, tr, va = green(a.fold)
    enc, head = load_specialist(TEACHERS[a.fold], ENCODERS[a.fold])
    log('teacher loaded', teacher=TEACHERS[a.fold], rows=int(len(tr)))
    prob = infer_rows(enc, head, tr, y, len(ids))
    path = out/'teacher_train_maps.npz'
    np.savez_compressed(path, indices=tr, ids=ids[tr], probability=prob,
                        train_green=np.array([n in set(binding['train_ids']) for n in ids[tr]]),
                        validation_green=np.array([n in set(binding['validation_ids']) for n in ids[tr]]))
    G.save(out/'teacher_infer.json', dict(created_utc=G.now(), fold=a.fold, teacher=TEACHERS[a.fold],
           teacher_sha=G.sha(TEACHERS[a.fold]), init_encoder=ENCODERS[a.fold], n=int(len(tr)), maps_sha=G.sha(path),
           views=1, note='validation rows of the fold are not inferred here; the teacher never saw their labels'))
    log('teacher maps written', path=str(path))


def cmd_thresholds(a):
    out = ROOT/a.tag
    ann, binding, ids, y, tr, va = green(a.fold)
    with np.load(out/'teacher_train_maps.npz') as z:
        rows, prob = z['indices'], z['probability']
    lookup = {int(r): i for i, r in enumerate(rows)}
    name_to_row = {n: i for i, n in enumerate(ids)}
    curve = {}
    ths = [round(float(t), 2) for t in np.arange(0.2, 0.951, 0.05)]
    for th in ths:
        rs = []
        for n in binding['train_ids']:
            r = name_to_row[n]
            assert r in lookup, 'train-green image must be a training row of this fold'
            p = prob[lookup[r]].astype(np.float32)
            _, support = spatial.point_target(ann[n]['markers_mha'], ann[n]['roi_box_mha'], G.GRID)
            found = probe.peaks(p, support, threshold=th)
            rs.append(spatial.match_points(found, ann[n]['markers_mha']))
        curve[str(th)] = dict(precision=float(np.mean([r['precision'] for r in rs])), recall=float(np.mean([r['recall'] for r in rs])),
                              f1=float(np.mean([r['f1'] for r in rs])))
    ok = [th for th in ths if curve[str(th)]['precision'] >= PRECISION_TARGET]
    tau_pos = min(ok) if ok else max(ths, key=lambda t: curve[str(t)]['precision'])
    # pseudo-label statistics on all training rows at the chosen thresholds
    counts, neg_frac = [], []
    for i in range(0, len(rows), 25):
        p = prob[i].astype(np.float32)
        counts.append(len(teacher_peaks(p, tau_pos)))
        neg_frac.append(float((p[:GRID_ROWS, :GRID_COLS] <= TAU_NEG).mean()))
    G.save(out/'thresholds.json', dict(created_utc=G.now(), fold=a.fold, rule=f'smallest threshold with mean in-ROI precision >= {PRECISION_TARGET} on the 20 train-green images (teacher saw their labels, so this is optimistic by construction); tau_neg fixed',
           curve=curve, tau_pos=tau_pos, tau_neg=TAU_NEG, fallback_used=not ok,
           pseudo_stats=dict(sampled_rows=len(counts), peaks_mean=float(np.mean(counts)), peaks_min=int(min(counts)), peaks_max=int(max(counts)),
                             negative_fraction_mean=float(np.mean(neg_frac)))))
    log('thresholds', tau_pos=tau_pos, tau_neg=TAU_NEG, peaks_mean=float(np.mean(counts)))


def cmd_student_train(a):
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    out = ROOT/a.tag
    ann, binding, ids, y, tr, va = green(a.fold)
    th = json.loads((out/'thresholds.json').read_text())
    tau_pos, tau_neg = th['tau_pos'], th['tau_neg']
    with np.load(out/'teacher_train_maps.npz') as z:
        rows, prob = z['indices'], z['probability']
    train_green = set(binding['train_ids'])
    pseudo_rows = np.array([i for i, r in enumerate(rows) if ids[r] not in train_green])
    lookup = {n: i for i, n in enumerate(ids)}
    names = binding['train_ids']
    frames = G.b.Frames(np.array([lookup[n] for n in names]), y, len(ids))
    strong_images = {n: frames[i][0][None] for i, n in enumerate(names)}
    strong = strong_items(ann, names)
    pseudo_frames = G.b.Frames(rows, y, len(ids))
    encoder = G.load_encoder(ENCODERS[a.fold]).train().requires_grad_(True)
    head = G.build_head(768, 'linear')
    opt = torch.optim.AdamW([{'params': encoder.parameters(), 'lr': 1e-5}, {'params': head.parameters(), 'lr': 1e-3}], weight_decay=1e-4)
    rng = np.random.default_rng(G.SEED)
    strong_keys = list(strong.keys())
    losses, t0 = [], time.time()
    cache = {}
    for step in range(a.updates):
        sk = strong_keys[rng.integers(len(strong_keys))]
        pi = int(pseudo_rows[rng.integers(len(pseudo_rows))])
        pf = int(rng.integers(4))
        xs = G.flip_image(strong_images[sk[0]], sk[1])
        if pi not in cache:
            cache[pi] = pseudo_frames[pi][0][None]
            if len(cache) > 64:
                cache.pop(next(iter(cache)))
        xp = G.flip_image(cache[pi], pf)
        tp, sp, npk = pseudo_item(prob[pi], pf, tau_pos, tau_neg)
        x = torch.cat([xs, xp])
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            tok = encoder.forward_features(G.b.inputs(x, 'cpdino_vitb'))['x_norm_patchtokens']
        logits = head(tok.float()).squeeze(-1).reshape(2, *G.GRID)
        ts, ss = torch.from_numpy(strong[sk][0]).cuda(), torch.from_numpy(strong[sk][1]).cuda()
        tpp, spp = torch.from_numpy(tp).cuda(), torch.from_numpy(sp).cuda()
        l_strong = F.binary_cross_entropy_with_logits(logits[0][ss], ts[ss])
        l_pseudo = F.binary_cross_entropy_with_logits(logits[1][spp], tpp[spp]) if spp.any() else logits.sum()*0
        loss = (l_strong+a.pseudo_weight*l_pseudo)/(1+a.pseudo_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters())+list(head.parameters()), 1.0, error_if_nonfinite=True)
        opt.step()
        losses.append([float(l_strong), float(l_pseudo)])
        if step % 100 == 0:
            log('train', step=step, strong=float(l_strong), pseudo=float(l_pseudo), peaks=npk, seconds=round(time.time()-t0, 1))
    torch.save(dict(encoder=encoder.state_dict(), head=head.state_dict(), threshold=EVAL_THRESHOLD, init_encoder=ENCODERS[a.fold],
                    tau_pos=tau_pos, tau_neg=tau_neg, updates=a.updates, pseudo_weight=a.pseudo_weight), out/'student.pt')
    G.save(out/'student_train.json', dict(created_utc=G.now(), fold=a.fold, updates=a.updates, pseudo_weight=a.pseudo_weight,
           pseudo_rows=int(len(pseudo_rows)), strong_items=len(strong_keys), tau_pos=tau_pos, tau_neg=tau_neg,
           loss_last100=dict(strong=float(np.mean([l[0] for l in losses[-100:]])), pseudo=float(np.mean([l[1] for l in losses[-100:]]))),
           seconds=round(time.time()-t0, 1), init_encoder=ENCODERS[a.fold]))
    log('student written')


def f1_rows(encoder, head, names, ann, ids, y, threshold):
    lookup = {n: i for i, n in enumerate(ids)}
    rows = []
    for n in names:
        x = G.b.Frames(np.array([lookup[n]]), y, len(ids))[0][0][None]
        p = maps(encoder, head, x)[0]
        _, support = spatial.point_target(ann[n]['markers_mha'], ann[n]['roi_box_mha'], G.GRID)
        found = probe.peaks(p, support, threshold=threshold)
        rows.append(dict(id=n, predicted=len(found), annotated=len(ann[n]['markers_mha']), **spatial.match_points(found, ann[n]['markers_mha'])))
    return dict(rows=rows, mean_f1=float(np.mean([r['f1'] for r in rows])), mean_recall=float(np.mean([r['recall'] for r in rows])),
                mean_precision=float(np.mean([r['precision'] for r in rows])))


def cd_diagnostics(prob_maps, y_cd):
    """Calibration-free: Pearson r between log estimator and log GT CD, for the phase 0 feature set."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cd_estimators_20261028 as P
    feats = [P.features_of(m) for m in prob_maps]
    keys = sorted(feats[0].keys())
    Fm = np.array([[f[k] for k in keys] for f in feats])
    ly = np.log(y_cd)
    return {k: float(np.corrcoef(np.log(Fm[:, j]), ly)[0, 1]) for j, k in enumerate(keys)}


def cmd_eval(a):
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    out = ROOT/a.tag
    ann, binding, ids, y, tr, va = green(a.fold)
    val_names = binding['validation_ids']
    t_enc, t_head = load_specialist(TEACHERS[a.fold], ENCODERS[a.fold])
    teacher_val = f1_rows(t_enc, t_head, val_names, ann, ids, y, EVAL_THRESHOLD)
    del t_enc, t_head
    torch.cuda.empty_cache()
    s_enc, s_head = load_specialist(out/'student.pt', ENCODERS[a.fold])
    student_val = f1_rows(s_enc, s_head, val_names, ann, ids, y, EVAL_THRESHOLD)
    curve = {str(th): f1_rows(s_enc, s_head, val_names, ann, ids, y, th)['mean_f1'] for th in (0.25, 0.3, 0.4, 0.45, 0.5)}
    prob = infer_rows(s_enc, s_head, va, y, len(ids))
    path = out/f'fold{a.fold}_student_maps.npz'
    train_green = set(binding['train_ids'])
    seen = np.array([n in train_green for n in ids[va]])
    np.savez_compressed(path, indices=va, ids=ids[va], probability=prob, point_labels_seen=seen)
    with np.load(Path('experiments/green_point_features_20261027/extract')/f'fold{a.fold}_specialist_s46.npz') as z:
        assert np.array_equal(z['indices'], va)
        teacher_prob = z['probability']
    ok = ~seen
    diag = dict(teacher=cd_diagnostics(teacher_prob[ok], y[va][ok, 0]), student=cd_diagnostics(prob[ok], y[va][ok, 0]))
    best_t = best_s = 'sg_cd_th0.35'
    result = dict(created_utc=G.now(), fold=a.fold, threshold=EVAL_THRESHOLD, teacher_validation=teacher_val, student_validation=student_val,
                  f1_delta=student_val['mean_f1']-teacher_val['mean_f1'], student_threshold_curve_information_only=curve,
                  student_maps=str(path), student_maps_sha=G.sha(path), rows=int(len(va)), rows_unseen=int(ok.sum()),
                  cd_log_correlation=diag, primary_cd_corr=dict(teacher=diag['teacher'][best_t], student=diag['student'][best_s]),
                  decision_rule='extend to folds 0-3 only if f1_delta >= +0.03 at the fixed threshold and r(log sg_cd_th0.35, log GT CD) is higher for the student than the teacher',
                  extend=bool(student_val['mean_f1']-teacher_val['mean_f1'] >= 0.03 and diag['student'][best_s] > diag['teacher'][best_t]))
    G.save(out/'result.json', result)
    log('eval done', teacher_f1=teacher_val['mean_f1'], student_f1=student_val['mean_f1'], extend=result['extend'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['teacher-infer', 'thresholds', 'student-train', 'eval'])
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--updates', type=int, default=3000)
    ap.add_argument('--pseudo-weight', type=float, default=1.0)
    a = ap.parse_args()
    {'teacher-infer': cmd_teacher_infer, 'thresholds': cmd_thresholds, 'student-train': cmd_student_train, 'eval': cmd_eval}[a.command](a)
