"""Image-side wrapper of the cell-route HEX step: raw frame -> 3 pre-only sg2 detectors x 4 exact flips -> peaks -> Konan features -> ridge.
    hk = HexKonan([sg2pre_s46.pt, sg2pre_s47.pt, sg2pre_s48.pt], 'out/deploy_hexkonan_params.json')
    hex_corr, feats = hk(image_u8, seg_pts, hex_base)   # image_u8 (972, 1296) raw; seg_pts (m, 3) Cellpose-SAM centroid x, y, area; hex_base = HEX lane final HEX
Preprocessing = the deployed hex6 corrector's preprocess (CLAHE 2.0, (40, 30) tiles, 1-99 % scaling). Heatmaps as lane_cv tta_infer_20261108.py:
mean of the 4 flips (flips undone on the in-frame cells), per-detector peaks SG.peaks2(conf >= 0.2, NMS 12) and peaks of the mean heatmap of the 3."""
import json, sys
import numpy as np
import torch
ROOT = '/workspace/clear-ec-research/'
sys.path[:0] = [ROOT + 'experiments/green_track_20261028/source', ROOT + 'experiments/green_track_20261028/deploy', ROOT + 'experiments/lane_cellroute_20261109/source']
import green_point_features_20261027 as G
import green_sg2_specialist_20261028 as SG
import phase2_foundation_preflight_20261002 as preflight
from hex6_correction import preprocess
import deploy_hexkonan as D
R2, C2 = SG.ROWS2, SG.COLS2


def load_detector(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    net, _, _ = preflight.build('cpdino_vitb', False, native_convnext_norm=True); enc = net[0].original; enc.load_state_dict(ck['encoder'], strict=True)
    head = SG.Head(seed=ck['seed']); head.load_state_dict(ck['head'])
    return enc.eval().requires_grad_(False).to(device), head.eval().to(device)


def heat(enc, head, x):
    """SG.heat without the hard-coded .cuda(): same padding (0,16,0,20 replicate, 3 channels, no normalisation for cpdino_vitb) and, on GPU, the same
    bf16 autocast; on CPU float32."""
    dev = next(enc.parameters()).device; x = torch.nn.functional.pad(x.to(dev).expand(-1, 3, -1, -1), (0, 16, 0, 20), mode='replicate').contiguous()
    with torch.no_grad():
        if dev.type == 'cuda':
            with torch.autocast('cuda', dtype=torch.bfloat16): tok = enc.forward_features(x)['x_norm_patchtokens']
        else: tok = enc.forward_features(x)['x_norm_patchtokens']
        return head(tok).sigmoid().cpu().numpy()


def tta_heat(enc, head, x, keep_single=None):
    """mean heat map over the 4 exact flips (flips undone on the in-frame cells); keep_single (list) receives the identity-flip full map (the
    single-view map; HEX features and GRAPH / sim graphs use single-view peaks). The 4 flipped views run as ONE batch of 4: the stored training
    peaks were made with batch-4 loaders (tta_infer_20261108, det_peaks_image) and bf16 results depend on the batch shape; a batch of 4 reproduces
    them exactly, a batch of 1 does not (diag_peaks_batch.py, 11-10)."""
    acc = np.zeros(SG.GRID2, np.float32); ms = heat(enc, head, torch.cat([G.flip_image(x, fl) for fl in range(4)], 0))
    for fl, (h, v) in enumerate(G.FLIPS):
        m = ms[fl]; mv = m[:R2, :C2]
        if fl == 0 and keep_single is not None: keep_single.append(m)
        if h: mv = mv[:, ::-1]
        if v: mv = mv[::-1, :]
        acc[:R2, :C2] += mv
    return acc / 4


class HexKonan:
    def __init__(self, detector_paths, params_path, device='cuda'):
        self.device = device; self.dets = [load_detector(p, device) for p in detector_paths]; self.params = json.load(open(params_path)); self.support = SG.frame_support()

    @torch.no_grad()
    def peaks(self, image_u8, single=False):
        """(peaks of the mean TTA map of the 3, [per-detector TTA peaks]) and, with single=True, also [per-detector single-view peaks]."""
        x = preprocess(image_u8).to(self.device); sv = []; maps = [tta_heat(e, h, x, sv) for e, h in self.dets]
        out = (SG.peaks2(np.mean(maps, 0), self.support, 0.2), [SG.peaks2(m, self.support, 0.2) for m in maps])
        return out + ([SG.peaks2(m, self.support, 0.2) for m in sv],) if single else out

    def __call__(self, image_u8, seg_pts, hex_base):
        pk3, pk = self.peaks(image_u8); F = D.features(pk3, pk[0], pk[1], pk[2], seg_pts)
        return float(D.correct(np.array([hex_base], float), F[None], self.params)[0]), F


def load_any(kind, path, device, full_encoder='experiments/fulldata_recipeA_20261028/regression_std_s46/runs/fold99/cpdino_vitb/epoch_20.pt'):
    """HEX lane det_peaks_image.load without .cuda(): kind 'full' = the checkpoint's own encoder + head; 'frozen' = head on the given recipe A
    encoder (FULL for deploy; the fold's recipe A encoder for fold tests)."""
    ck = torch.load(path if path.startswith('/') else ROOT + path, map_location='cpu', weights_only=False)
    net, _, _ = preflight.build('cpdino_vitb', False, native_convnext_norm=True); enc = net[0].original
    if kind == 'full': enc.load_state_dict(ck['encoder'], strict=True)
    else:
        st = torch.load(full_encoder if full_encoder.startswith('/') else ROOT + full_encoder, map_location='cpu', weights_only=True)['model']
        enc.load_state_dict({k[len('0.original.'):]: v for k, v in st.items() if k.startswith('0.original.')}, strict=True)
    head = SG.Head(seed=ck.get('seed', 46)); head.load_state_dict(ck['head'])
    return enc.eval().requires_grad_(False).to(device), head.eval().to(device)


@torch.no_grad()
def single_peaks(enc, head, image_u8):
    """single-view sg2 peaks (SG.heat + SG.peaks2 at 0.2, NMS 12 px), as det_peaks_image.peaks_for; the image runs as a batch of 4 copies so the bf16
    numerics match the batch-4 training peaks exactly (see tta_heat)."""
    return SG.peaks2(heat(enc, head, preprocess(image_u8).repeat(4, 1, 1, 1))[0], SG.frame_support(), 0.2)
