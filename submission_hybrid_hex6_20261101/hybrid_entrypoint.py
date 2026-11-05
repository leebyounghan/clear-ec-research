"""Grand Challenge entrypoint for the submission-1 hybrid. Same I/O contract as the Phase-1 app.

2026-10-31: per-image failures are isolated by hybrid_guard_20261031 (see its docstring); start-up checks stay loud.
2026-11-01 (submission 2): the hex6 correction (ten detectors, frozen a/b/w) adjusts HEX after the hybrid; the corrector is
loaded at start-up (loud); a per-image hex6 failure keeps the hybrid HEX and is logged.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import SimpleITK as sitk
import torch

sys.path.insert(0, '/opt/app')
import hybrid_inference_20261030 as H
import hybrid_guard_20261031 as G
import hex6_t4_20261101 as X6

RECIPEA_MEMBERS = ('recipea/member_s46.pt', 'recipea/member_s47.pt', 'recipea/member_s48.pt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('/input'))
    parser.add_argument('--output', type=Path, default=Path('/output'))
    parser.add_argument('--model', type=Path, default=Path('/opt/ml/model'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()
    start = time.time()
    import os
    os.chdir('/opt/app')  # frozen recipe A sources use research-tree relative paths (see Dockerfile symlink)
    H.assert_memory_efficient_sdpa()
    corrector = X6.Hex6CorrectorT4(dtype=torch.float16 if args.fp16 else torch.bfloat16)
    sockets = json.loads((args.input / 'inputs.json').read_text())
    if [x['socket']['slug'] for x in sockets] != ['corneal-specular-microscopy-image']:
        raise ValueError('Expected single microscopy input socket')
    paths = list((args.input / 'images/corneal-specular-microscopy-image').glob('*.mha'))
    if len(paths) != 1:
        raise ValueError('Expected exactly one MHA')
    outputs = [args.output / f'{key}.json' for key in ('cell-density', 'coefficient-of-variation', 'hexagonality')]
    if any(p.exists() for p in outputs):
        raise FileExistsError('Refusing to overwrite existing prediction')
    members = [str(args.model / m) for m in RECIPEA_MEMBERS]
    for m in members:
        if not Path(m).exists():
            raise FileNotFoundError(m)
    try:
        image = sitk.GetArrayFromImage(sitk.ReadImage(str(paths[0])))
    except Exception as e:  # unreadable case: the guard falls back to constants and logs it
        image = None
        read_error = f'{type(e).__name__}: {e}'[:300]
    else:
        read_error = None
    if image is None:
        value, parts = G.FALLBACK_CONSTANT.copy(), dict(mode='constant', errors=dict(read=read_error), notes=[],
                                                        phase1=None, recipea=None)
    else:
        value, parts = G.predict_guarded(image, args.model, args.config, members,
                                         torch.float16 if args.fp16 else torch.bfloat16)
    hex6 = dict(applied=False)
    if parts['mode'] != 'constant':
        try:
            if os.environ.get('CLEAR_EC_FAULT_INJECT', '') == 'hex6':
                raise RuntimeError('injected hex6 fault (test only)')
            img, _ = G.coerce_image(image)
            corrected, info = corrector(img, value)
            if not G._valid(corrected):
                raise ValueError(f'invalid hex6 output {corrected}')
            hex6 = dict(applied=not info['fallback'], feature=info['feature'], hex_before=float(value[2]), hex_after=float(corrected[2]))
            value = corrected
        except Exception as e:
            parts['errors']['hex6'] = f'{type(e).__name__}: {e}'[:300]
            torch.cuda.empty_cache()
    args.output.mkdir(parents=True, exist_ok=True)
    for path, v in zip(outputs, value):
        with path.open('x') as handle:
            json.dump(float(v), handle, allow_nan=False)

    def as_list(v):
        return None if v is None else [float(x) for x in v]
    print(json.dumps(dict(seconds=time.time() - start, phase1_members=6, recipea_members=len(members),
                          mode=parts['mode'], errors=parts['errors'], notes=parts['notes'], hex6=hex6,
                          phase1=as_list(parts['phase1']), recipea=as_list(parts['recipea']),
                          output=[float(v) for v in value])), flush=True)


if __name__ == '__main__':
    main()
