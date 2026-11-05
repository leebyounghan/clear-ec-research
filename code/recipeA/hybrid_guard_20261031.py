"""Per-image fault isolation around the verified submission-1 hybrid.

Normal input (uint8, 2-D) takes exactly the verified path: the same calls, in the same order, with the same
arithmetic as hybrid_inference_20261030.predict_hybrid. The module only changes what happens when something
fails on one image, because a crash on a hidden test image cannot be seen or reproduced by us and would fail the
whole evaluation:
  recipe A fails  -> Phase-1 prediction alone
  Phase-1 fails   -> recipe A prediction alone
  both fail       -> MAPE-optimal training constants (CD 2770, CV 0.36, HEX 0.53; weighted medians of the
                     8,954 audited training labels with weights 1/y)
Start-up checks (CUDA, memory-efficient SDPA for 20,336 tokens, model files) stay loud in the entrypoint so a
broken runtime is caught by the platform Try-out instead of silently degrading every case.
Every fallback is printed in the job log ('mode', 'errors', 'notes').
CLEAR_EC_FAULT_INJECT is for the container test only (never set on the platform): 'recipea', 'phase1' or 'both'.
"""
import gc
import os
from pathlib import Path

import numpy as np
import torch

import hybrid_inference_20261030 as H

FALLBACK_CONSTANT = np.array([2770.0, 0.36, 0.53])


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _valid(v):
    v = np.asarray(v)
    return v.shape == (3,) and bool(np.isfinite(v).all()) and bool((v > 0).all())


def coerce_image(image):
    """Identity for the validated case (2-D uint8). Otherwise convert to 2-D uint8 and record what was done."""
    a = np.asarray(image)
    notes = []
    if a.ndim == 3:
        if a.shape[0] == 1:
            a = a[0]
            notes.append('squeezed leading singleton')
        elif a.shape[-1] in (3, 4):
            a = a[..., :3].astype(np.float64).mean(-1)
            notes.append(f'channels-last {image.shape} -> gray')
        elif a.shape[0] in (3, 4):
            a = a[:3].astype(np.float64).mean(0)
            notes.append(f'channels-first {image.shape} -> gray')
        else:
            a = a[a.shape[0] // 2]
            notes.append(f'3-D {image.shape} -> middle slice')
    elif a.ndim != 2:
        raise ValueError(f'unsupported input shape {a.shape}')
    if a.dtype != np.uint8:
        f = np.nan_to_num(a.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = float(f.min()), float(f.max())
        if lo >= 0 and hi <= 255:
            f = np.round(f)
        else:
            f = (f - lo) / (hi - lo + 1e-12) * 255.0
        a = np.clip(f, 0, 255).astype(np.uint8)
        notes.append(f'dtype {np.asarray(image).dtype} -> uint8')
    return np.ascontiguousarray(a), notes


def predict_guarded(image, phase1_model_dir, phase1_config, recipea_members, dtype=torch.float16):
    fault = os.environ.get('CLEAR_EC_FAULT_INJECT', '')
    errors, notes = {}, []
    p1 = ra = None
    try:
        img, notes = coerce_image(image)
    except Exception as e:  # unusable input: constants keep the evaluation alive
        errors['input'] = f'{type(e).__name__}: {e}'[:300]
        img = None
    if img is not None:
        try:
            if fault in ('phase1', 'both'):
                raise RuntimeError('injected phase1 fault (test only)')
            import inference as phase1
            p1, _ = phase1.predict(img, Path(phase1_model_dir), phase1_config, dtype=dtype, return_members=True)
            if not _valid(p1):
                raise ValueError(f'invalid phase1 output {p1}')
        except Exception as e:
            errors['phase1'] = f'{type(e).__name__}: {e}'[:300]
            p1 = None
            _free()
        try:
            if fault in ('recipea', 'both'):
                raise RuntimeError('injected recipe A fault (test only)')
            ra, _ = H.predict_recipea(img, recipea_members, dtype)
            if not _valid(ra):
                raise ValueError(f'invalid recipe A output {ra}')
        except Exception as e:
            errors['recipea'] = f'{type(e).__name__}: {e}'[:300]
            ra = None
            _free()
    if p1 is not None and ra is not None:
        out, mode = H.HYBRID_WEIGHT * ra + (1 - H.HYBRID_WEIGHT) * p1, 'hybrid'
    elif p1 is not None:
        out, mode = np.asarray(p1, dtype=np.float64), 'phase1_only'
    elif ra is not None:
        out, mode = np.asarray(ra, dtype=np.float64), 'recipea_only'
    else:
        out, mode = FALLBACK_CONSTANT.copy(), 'constant'
    if not _valid(out):
        errors['output'] = f'invalid combined output {out}'
        out, mode = FALLBACK_CONSTANT.copy(), 'constant'
    return out, dict(mode=mode, errors=errors, notes=notes, phase1=p1, recipea=ra)
