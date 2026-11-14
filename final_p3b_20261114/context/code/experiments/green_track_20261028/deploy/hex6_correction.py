"""Standalone submission-2 hex6 correction.

    from hex6_correction import Hex6Corrector
    corr = Hex6Corrector(deploy_dir)              # loads 9 detectors + blend params once
    corrected = corr(image_u8, base_pred)         # image_u8: (972, 1296) uint8 raw frame; base_pred: (3,) CD, CV, HEX

Pipeline per image: training-time preprocessing (CLAHE 2.0 with a (40, 30) tile grid, 1-99 percentile scaling to
[0, 1]), one forward pass per detector (bf16, SDPA), peaks at threshold 0.35 (stride-8: 3x3 local maxima with
centroid refinement; stride-2: 5x5 local maxima, centroid, NMS 12 px), bounded Voronoi cells -> six-neighbour
fraction hex6_all, mean of log hex6 over detectors with a value, HEX_corr = exp((1-w) log HEX_base + w (a + b feat)).
CD and CV are returned unchanged.  Fallback to base when no detector yields hex6.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter
from scipy.spatial import Delaunay, Voronoi

HERE = Path(__file__).resolve().parent
SRC = HERE.parent/'source'
sys.path.insert(0, str(SRC))
import green_point_features_20261027 as G
import green_sg2_specialist_20261028 as SG
import green_s8_specialist_seed_20261028 as S8
import phase2_foundation_preflight_20261002 as preflight

GRID8, GRID_ROWS8, GRID_COLS8, STRIDE8, OFFSET8 = (124, 164), 122, 162, 8, 3.5


def preprocess(image_u8):
    import cv2
    cv2.setNumThreads(1)
    a = cv2.createCLAHE(2.0, (1296//32, 972//32)).apply(np.asarray(image_u8, dtype=np.uint8))
    lo, hi = np.percentile(a, (1, 99))
    a = np.clip((a.astype(np.float32)-lo)/(hi-lo+1e-6), 0, 1)
    return torch.from_numpy(np.ascontiguousarray(a[None, None]))


def peaks_s8(prob):
    p = prob.astype(np.float32)[:GRID_ROWS8, :GRID_COLS8]
    peak = (p == maximum_filter(p, size=3, mode='nearest')) & (p >= 0.35)
    yy, xx = np.nonzero(peak)
    pts = []
    for iy, ix in zip(yy, xx):
        y0, y1, x0, x1 = max(0, iy-1), min(GRID_ROWS8, iy+2), max(0, ix-1), min(GRID_COLS8, ix+2)
        w = p[y0:y1, x0:x1].astype(np.float64)
        gy, gx = np.mgrid[y0:y1, x0:x1]
        pts.append([(w*gx).sum()/w.sum(), (w*gy).sum()/w.sum()])
    return np.array(pts).reshape(-1, 2)*STRIDE8+OFFSET8


def hex6_all(pts):
    if len(pts) < 20:
        return np.nan
    tri = Delaunay(pts); vor = Voronoi(pts); sides = []
    for i in range(len(pts)):
        r = vor.regions[vor.point_region[i]]
        if -1 in r or len(r) == 0:
            continue
        if (tri.find_simplex(vor.vertices[r]) < 0).any():
            continue
        sides.append(len(r))
    if len(sides) < 12:
        return np.nan
    return float((np.array(sides) == 6).mean())+1e-3


def family_to_file(fam):
    """OOF detector family name -> deployed checkpoint stem."""
    if fam == 'sg2_specialist':
        return 'sg2_s46'
    if fam.startswith('sg2_specialist_s'):
        return 'sg2_' + fam.split('_')[-1]
    if fam == 'external_srep/preonly':
        return 'sg2pre_s46'
    if fam.startswith('external_srep/s') and fam.endswith('_preonly'):
        return 'sg2pre_' + fam.split('/')[1].split('_')[0]
    return fam          # s8_s46, s8_s47, s8_s48, s8multi_s47


class Hex6Corrector:
    def __init__(self, deploy_dir=HERE, device='cuda'):
        deploy_dir = Path(deploy_dir)
        self.params = json.loads((deploy_dir/'hex6_blend_params.json').read_text())
        self.device = device
        self.detectors = []
        for fam in self.params['families']:
            name = family_to_file(fam)
            ck = torch.load(deploy_dir/'detectors'/f'{name}.pt', map_location='cpu', weights_only=False)
            # build the encoder architecture only (no init checkpoint needed in the container), then load the detector's own weights
            net, _, _ = preflight.build('cpdino_vitb', False, native_convnext_norm=True)
            enc = net[0].original
            enc.load_state_dict(ck['encoder'], strict=True)
            enc = enc.eval().requires_grad_(False)
            if ck['kind'] == 'sg2':
                head = SG.Head(seed=ck['seed']).to(device)
            else:
                head = G.build_head(768 if ck['features'] == 'last' else 768*len(G.LAYERS), 'linear', seed=ck['seed'])
            head.load_state_dict(ck['head']); head.eval()
            self.detectors.append((name, ck['kind'], ck['features'], enc.to(device).eval(), head))
        self.support2 = SG.frame_support()

    @torch.no_grad()
    def hex6_per_detector(self, image_u8):
        x = preprocess(image_u8)
        out = {}
        for name, kind, features, enc, head in self.detectors:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                tok = S8.tokens_of(enc, x, features)
            if kind == 'sg2':
                p = head(tok).sigmoid()[0].float().cpu().numpy()
                pk = SG.peaks2(p, self.support2, 0.35)
                pts = pk[:, :2].astype(np.float64)
            else:
                p = head(tok.float()).squeeze(-1).sigmoid().reshape(*GRID8).cpu().numpy()
                pts = peaks_s8(p)
            out[name] = hex6_all(pts)
        return out

    def feature(self, image_u8):
        h = self.hex6_per_detector(image_u8)
        vals = [np.log(v) for v in h.values() if np.isfinite(v)]
        return (float(np.mean(vals)) if vals else None), h

    def __call__(self, image_u8, base_pred):
        base_pred = np.asarray(base_pred, dtype=np.float64).reshape(3)
        feat, per = self.feature(image_u8)
        corrected = base_pred.copy()
        if feat is None or not np.isfinite(base_pred[2]) or base_pred[2] <= 0:
            return corrected, dict(feature=None, fallback=True, per_detector=per)
        a, b, w = self.params['a'], self.params['b'], self.params['w']
        corrected[2] = float(np.exp((1-w)*np.log(base_pred[2])+w*(a+b*feat)))
        return corrected, dict(feature=feat, fallback=False, per_detector=per)


def hex6_correction(image_u8, base_pred, corrector=None):
    corrector = corrector or Hex6Corrector()
    return corrector(image_u8, base_pred)[0]
