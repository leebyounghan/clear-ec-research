"""New-recipe inference, member layer (cell-route lane, 2026-11-10). Image in, per-member values out. Reads no labels, folds or OOF files.
Every regression member is the families' common head: Sequential(Pooled(tokens, 'std'), LayerNorm, Linear(d,128), GELU, Linear(128,3)) or, for
recipe A, the cpdino SpatialPoolEncoder('std') net; checkpoints carry model, log_mean, log_std. Prediction = exp(z * log_std + log_mean); TTA and
epoch reads are means of z (= log predictions), exactly as the lanes' readers (family_tta_20261108, bb_runner evaluate, hybrid RecipeAMember).
Input: the raw 972x1296 uint8 frame -> CLAHE(2.0, 40x30 tiles) + 1-99 % scaling (identical to the training loader Frames)."""
import os, sys
import numpy as np
import torch
import torch.nn.functional as Fnn
ROOT = os.environ.get('CLEAR_EC_ROOT', '/workspace/clear-ec-research/')
os.chdir(ROOT)
for p in ('experiments/oof_recipeA_20261027/staging_regression_zero', 'experiments/phase2_crossfold_20261015/source', 'experiments/lane_cd_20261108/source',
          'experiments/green_track_20261028/source', 'experiments/lane_cellroute_20261109/source'):
    if p not in sys.path: sys.path.insert(0, p)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
ADT = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if DEV == 'cuda' else None   # bf16 as in training; fp16 where bf16 is absent
FLIPS = ((False, False), (True, False), (False, True), (True, True))
MEAN, STD = (.485, .456, .406), (.229, .224, .225)
FAM = {'cpdino_recipeA': dict(pad=(16, 20), norm=False), 'dinov3_b16': dict(pad=(16, 20), norm=True), 'dinov3_l16t': dict(pad=(16, 20), norm=True),
       'cpsam_c512': dict(pad=(0, 0), norm=False, crop=512)}   # family runner FAMILIES entries


def preprocess(image_u8):
    import cv2
    cv2.setNumThreads(1)
    img = np.asarray(image_u8)
    if img.ndim == 3 and img.shape[0] in (1, 3): img = img[0]
    assert img.shape == (972, 1296) and img.dtype == np.uint8, (img.shape, img.dtype)
    a = cv2.createCLAHE(2.0, (1296 // 32, 972 // 32)).apply(np.ascontiguousarray(img)); lo, hi = np.percentile(a, (1, 99))
    a = np.clip((a.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
    return torch.from_numpy(np.ascontiguousarray(a[None, None]))


def _inputs(x, fam):
    f = FAM[fam]; x = x.to(DEV).expand(-1, 3, -1, -1)
    if f.get('crop'):
        c = f['crop']; top, left = (x.shape[-2] - c) // 2, (x.shape[-1] - c) // 2; x = x[..., top:top + c, left:left + c]
    x = Fnn.pad(x, (0, f['pad'][0], 0, f['pad'][1]), mode='replicate')
    if f['norm']: x = (x - x.new_tensor(MEAN)[None, :, None, None]) / x.new_tensor(STD)[None, :, None, None]
    return x.contiguous()


def _build(fam):
    from torch import nn
    if fam == 'cpdino_recipeA':
        import phase2_foundation_preflight_20261002 as preflight, phase2_pooling_20261019 as pooling
        net, _, _ = preflight.build('cpdino_vitb', False, native_convnext_norm=True); net[0] = pooling.SpatialPoolEncoder(net[0].original, 'std'); return net
    if fam in ('dinov3_b16', 'cpsam_c512'):
        import family_cd_20261108 as Fm
        tok = Fm.build_tokens(fam)
    else:
        import family_bb_cd_20261108 as Fm
        tok = Fm.build_tokens('dinov3_l16t')
    return nn.Sequential(Fm.Pooled(tok, 'std'), nn.LayerNorm(tok.dim), nn.Linear(tok.dim, 128), nn.GELU(), nn.Linear(128, 3))


class Member:
    """One checkpoint. z(x, flips) returns the standardized-log outputs averaged over the given flips (3,)."""
    _cache = {}

    def __init__(self, path, fam):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        if fam not in Member._cache: Member._cache[fam] = _build(fam)
        net = Member._cache[fam]; net.load_state_dict(ck['model'], strict=True); self.net = net.to(DEV).eval().requires_grad_(False)
        self.mu = np.asarray(ck['log_mean'], np.float64); self.sd = np.asarray(ck['log_std'], np.float64); self.fam = fam; self.path = path

    @torch.no_grad()
    def logpred(self, x, tta):
        zs = []
        for h, v in (FLIPS if tta else FLIPS[:1]):
            xf = x
            if h: xf = torch.flip(xf, (-1,))
            if v: xf = torch.flip(xf, (-2,))
            if DEV == 'cuda':
                with torch.autocast('cuda', dtype=ADT): zs.append(self.net(_inputs(xf, self.fam)).float()[0, :3].cpu().numpy())
            else: zs.append(self.net(_inputs(xf, self.fam)).float()[0, :3].numpy())
        return np.mean(zs, 0).astype(np.float64) * self.sd + self.mu      # log prediction (3,)


_LOGCACHE = {}


def read_logs(x, paths, fam, tta):
    """(n_ckpt, 3) log predictions; cached per image tensor so a checkpoint read twice (A and A_cd share epoch 20 TTA) runs once."""
    out = []
    for p in paths:
        k = (x.data_ptr(), p, bool(tta))
        if k not in _LOGCACHE: _LOGCACHE[k] = Member(p, fam).logpred(x, tta)
        out.append(_LOGCACHE[k])
    return np.array(out)


def clear_cache(): _LOGCACHE.clear()


# ---- member read rules (per image). Each returns linear-space (3,) values unless noted. ----
def recipeA_tta(x, seed_ckpts, scalars):
    """A for CV/HEX (hybrid_oof A_recipeA_tta_scalar): arithmetic mean over seeds of exp(TTA4 log at epoch 20), times per-metric scalars."""
    return np.mean([np.exp(read_logs(x, [p], 'cpdino_recipeA', True)[0]) for p in seed_ckpts], 0) * np.asarray(scalars)


def recipeA_cd(x, seed_runs, scalar_cd):
    """A for CD (CD lane): per seed exp(0.5 log TTA4@e20 + 0.5 mean_e log single view e8-19); seeds averaged arithmetically; times the CD scalar.
    seed_runs: list of dict(e20=path, single=[paths e8..e19]). Returns the CD value (scalar)."""
    v = []
    for r in seed_runs:
        t = read_logs(x, [r['e20']], 'cpdino_recipeA', True)[0, 0]; s = read_logs(x, r['single'], 'cpdino_recipeA', False)[:, 0].mean()
        v.append(np.exp(0.5 * t + 0.5 * s))
    return float(np.mean(v) * scalar_cd)


def dino_cd(x, ckpts):
    """D (CD lane): exp(mean over seeds of [mean log over 4 flips x epochs 16-20]). Returns (3,).
    ckpts: one seed's checkpoint list (s46 alone, the member before 11-11; result unchanged) or a list of per-seed lists
    (CD lane seed audit 2026-11-11: D = exp(0.5 logD_s46 + 0.5 logD_s47))."""
    seeds = ckpts if ckpts and isinstance(ckpts[0], (list, tuple)) else [ckpts]
    return np.exp(np.mean([read_logs(x, c, 'dinov3_b16', True).mean(0) for c in seeds], 0))


def dino_cv(x, seed_ckpts):
    """DINO member (CV lane): per seed exp(mean log over epochs 16-20, no TTA); seeds averaged in log space. Returns (3,)."""
    return np.exp(np.mean([read_logs(x, c, 'dinov3_b16', False).mean(0) for c in seed_ckpts], 0))


def vitl_tta_only(x, seeds):
    """V for the packaging tiers (CD lane freeze_cd_params_t1_20261112 'V'): raw track, four-flip TTA at final.pt only, exp(mean over seeds of
    the log). Returns (3,)."""
    return np.exp(np.mean([read_logs(x, [sd['tta']], 'dinov3_l16t', True)[0] for sd in seeds], 0))


def vitl_cd(x, seeds, single_log_stubs=None):
    """V (CD lane, cd_read_20261108 'vitl_s4647_tt'): per seed exp(0.5 mean log single view epochs 8-11 + 0.5 log TTA4 at final.pt / epoch 12),
    then the geometric mean over seeds (s46, s47). seeds: [dict(single=[paths e8-11], tta=path)]. single_log_stubs: per-seed (3,) stored mean
    log when the epoch 8-11 weights were not kept (fold runs keep final.pt only; the full-data builds keep them)."""
    L = []
    for k, sd in enumerate(seeds):
        s = read_logs(x, sd['single'], 'dinov3_l16t', False).mean(0) if sd.get('single') else np.asarray(single_log_stubs[k])
        L.append(0.5 * s + 0.5 * read_logs(x, [sd['tta']], 'dinov3_l16t', True)[0])
    return np.exp(np.mean(L, 0))


ATTN_SRC = '/workspace/clear-ec-research/experiments/lane_backbone_20261108/source'


def _attnpool_net():
    """The backbone lane's attention-pool DINOv3-B network, built as its hand-over note (backbone_attnpool_f99_handoff.md, 2026-11-11):
    bb_runner_attnpool_20261111.AttnPool over the DINOv3-B tokens, LayerNorm(4 dim), Linear(4 dim, 128), GELU, Linear(128, 3). GPU only.
    The SSL weights that build_net can load (env ATTNPOOL_SSL) are overwritten by every strict checkpoint load, so they are not read by default."""
    if 'attnpool' not in Member._cache:
        import argparse, os
        from torch import nn
        if ATTN_SRC not in sys.path: sys.path.insert(0, ATTN_SRC)
        import bb_runner_attnpool_20261111 as B
        F, R = B.F, B.R; B.FAMILY = 'dinov3_b16'; R.ARGS = argparse.Namespace(seed=46)
        net = F.build_net(argparse.Namespace(family='dinov3_b16'), 46, os.environ.get('ATTNPOOL_SSL') or None)
        dim = net[1].normalized_shape[0]; torch.manual_seed(48)
        net[0] = B.AttnPool(net[0].tokens, dim).cuda(); net[1] = nn.LayerNorm(4 * dim).cuda(); net[2] = nn.Linear(4 * dim, 128).cuda()
        Member._cache['attnpool'] = (net.eval(), F)
    return Member._cache['attnpool']


@torch.no_grad()
def attnpool_hex(x, ckpts, fam='dinov3_b16'):
    """attnp5 (HEX lane 39-column joint ridge): exp of the mean over checkpoints (raw epochs 8-11 + final.pt = epoch 12) of the log prediction,
    single view, forward as bb_runner_attnpool_20261111.evaluate (bf16 autocast on GPU, family_bb.inputs). Returns (3,); the ridge reads column 2."""
    net, F = _attnpool_net(); L = []
    for p in ckpts:
        k = (x.data_ptr(), p, 'attnpool')
        if k not in _LOGCACHE:
            ck = torch.load(p, map_location='cpu', weights_only=False); net.load_state_dict(ck['model'], strict=True)
            with torch.autocast('cuda', dtype=torch.bfloat16): z = net(F.inputs(x, 'dinov3_b16')).float()[0, :3].cpu().numpy()
            _LOGCACHE[k] = z.astype(np.float64) * np.asarray(ck['log_std'], np.float64) + np.asarray(ck['log_mean'], np.float64)
        L.append(_LOGCACHE[k])
    return np.exp(np.mean(L, 0))


def family_mean(x, ckpts, fam):
    """exp of the mean log prediction over checkpoints, single view (HEX members cvw3_fix: DINOv3-B CV-weighted epochs 8-12; mtcnn5_fix: CPSAM 512
    epochs 8-12; as hex_apply's inputs dino_cvw3 / cps5, column 2 = HEX). Returns (3,)."""
    return np.exp(read_logs(x, ckpts, fam, False).mean(0))


def orderdino_cv(x, ckpts):
    """orderdino member (CV lane): mean log over epochs 16-20 of the order-loss DINOv3 B/16 s46, single view. Returns (3,) log values."""
    return read_logs(x, ckpts, 'dinov3_b16', False).mean(0)


@torch.no_grad()
def recipeA_tokens(x, ckpt):
    """patch tokens (124*164, 768) of a recipe A checkpoint's encoder, one unflipped view (frozen-head blocks, CV lane token_stats_cv fw):
    replicate pad 16/20, 3 channels, no normalisation; bf16 autocast on GPU as in training, float32 on CPU."""
    import phase2_foundation_preflight_20261002 as preflight
    if ('tok', ckpt) not in Member._cache:
        net, _, _ = preflight.build('cpdino_vitb', False, native_convnext_norm=True); enc = net[0].original
        st = torch.load(ckpt, map_location='cpu', weights_only=True)['model']; enc.load_state_dict({k[len('0.original.'):]: v for k, v in st.items() if k.startswith('0.original.')}, strict=True)
        Member._cache[('tok', ckpt)] = enc.to(DEV).eval().requires_grad_(False)
    enc = Member._cache[('tok', ckpt)]; xi = _inputs(x, 'cpdino_recipeA')
    if DEV == 'cuda':
        with torch.autocast('cuda', dtype=ADT): t = enc.forward_features(xi)['x_norm_patchtokens']
    else: t = enc.forward_features(xi)['x_norm_patchtokens']
    return t[0].float().cpu().numpy()
