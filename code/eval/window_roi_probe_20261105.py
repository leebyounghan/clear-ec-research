"""Can a trained regressor find the technician's box (2026-11-05)? The label comes from one unknown window. If a window's own prediction matches the label better inside the expert box than elsewhere, winner-takes-all training has something to find; if not, the idea is dead before any training.
On the 25 annotated frames, each with the recipe A seed-46 model of its fold (out of fold for the labels of that frame): patch tokens of the frame; the model's own head on the std of tokens inside each box-sized window (51 x 67 tokens, stride 6); window mismatch = sum over metrics of |log prediction - log label| after a per-metric offset that sets the frame-level prediction to its calibrated level. Rank of the window nearest the expert box; IoU with the box of the best-matching window vs random. E9 found the box ranks like a random window on detector statistics (30-38th percentile). Gate: median rank <= 15th percentile and best-window IoU clearly above chance."""
import json, sys
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
import phase2_foundation_train_20261002 as b
import phase2_foundation_preflight_20261002 as preflight
import phase2_pooling_20261019 as pooling
E = C.E; base = E.load_base(); pos = {str(i).strip(): n for n, i in enumerate(base['ids'])}; fold = base['fold']; Y = base['y']; idx = base['idx']
ann = json.load(open('/workspace/clear-ec-research/datasets/clear_ec_official/green_eval/annotations.json')); BOX = (51, 67); STRIDE = 6
def model(f):
    root = C.X / C.RECIPE[46][f]; m = json.loads((root / 'manifest.json').read_text()); ck = torch.load(root / 'runs' / f'fold{f}' / 'cpdino_vitb' / 'epoch_20.pt', map_location='cpu', weights_only=True)
    net, _, _ = preflight.build('cpdino_vitb', False, native_convnext_norm=True); net[0] = pooling.SpatialPoolEncoder(net[0].original, m['recipe']['pooling']); net.load_state_dict(ck['model'], strict=True); return net.cuda().eval().requires_grad_(False), np.array(ck['log_mean'], float), np.array(ck['log_std'], float)
def iou(a, c):
    ix = max(0, min(a[2], c[2]) - max(a[0], c[0])); iy = max(0, min(a[3], c[3]) - max(a[1], c[1])); i = ix * iy; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (c[2] - c[0]) * (c[3] - c[1]) - i)
rows = []; cache = {}
frames = b.Frames(np.array([int(idx[pos[n]]) for n in ann]), Y, 8954)
names = list(ann)
for f in range(5):
    todo = [k for k, n in enumerate(names) if fold[pos[n]] == f]
    if not todo: continue
    net, mu, sigma = model(f); head = net[1:]
    for k in todo:
        n = names[k]; x, _, _ = b.Frames(np.array([int(idx[pos[n]])]), np.zeros((8954, 3), np.float32), 8954)[0]
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]), torch.autocast('cuda', dtype=torch.bfloat16):
            t = net[0].original.forward_features(b.inputs(x[None], 'cpdino_vitb'))['x_norm_patchtokens'].float(); G = t.reshape(1, 124, 164, 768)[0]
            ws, boxes = [], []
            for y0 in range(0, 124 - BOX[0] + 1, STRIDE):
                for x0 in range(0, 164 - BOX[1] + 1, STRIDE):
                    ws.append(G[y0:y0 + BOX[0], x0:x0 + BOX[1]].reshape(-1, 768).std(0)); boxes.append((x0 * 8, y0 * 8, (x0 + BOX[1]) * 8, (y0 + BOX[0]) * 8))
            z = head(torch.stack(ws)).float().cpu().numpy(); z0 = head(t.std(1)).float().cpu().numpy()
        L = z * sigma + mu; L0 = z0 * sigma + mu; y = np.log(np.array([ann[n]['gt']['CD'], ann[n]['gt']['CV'], ann[n]['gt']['HEX']], float))
        cal = np.log(base['B'][pos[n]]) - L0[0]; mis = np.abs(L + cal - y).sum(1); mis_no_cd = np.abs(L + cal - y)[:, 1:].sum(1); ex = ann[n]['roi_box_mha']; I = np.array([iou(c, ex) for c in boxes]); near = int(np.argmax(I))
        rk = float((mis < mis[near]).mean()); best = int(np.argmin(mis)); top5 = np.argsort(mis)[:5]; rows.append(dict(id=n, fold=int(f), n_windows=len(boxes), rank_of_box=round(rk, 3), iou_near=round(float(I[near]), 3), iou_best=round(float(I[best]), 3), iou_top5=round(float(I[top5].mean()), 3), iou_random=round(float(I.mean()), 3), spread=[round(float(v), 3) for v in L.std(0)], frame_vs_y=[round(float(v), 3) for v in (L0[0] + cal - y)], rank_cv_hex=round(float((mis_no_cd < mis_no_cd[near]).mean()), 3))); print(json.dumps(rows[-1]), flush=True)
    del net
s = dict(n=len(rows), median_rank=float(np.median([r['rank_of_box'] for r in rows])), median_rank_cv_hex=float(np.median([r['rank_cv_hex'] for r in rows])), iou_best=float(np.mean([r['iou_best'] for r in rows])), iou_top5=float(np.mean([r['iou_top5'] for r in rows])), iou_random=float(np.mean([r['iou_random'] for r in rows])), share_rank_below_20=float(np.mean([r['rank_of_box'] < 0.2 for r in rows])))
print('SUMMARY', json.dumps(s), flush=True); json.dump(dict(summary=s, rows=rows), open(C.X / 'meta_ceiling_20261103/sweep/window_roi_probe.json', 'w'), indent=1); print('done', flush=True)
