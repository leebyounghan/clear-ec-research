"""Grand Challenge entrypoint for the submission-1 hybrid. Same I/O contract as the Phase-1 app."""
import argparse
import json
from pathlib import Path
import sys
import time

import SimpleITK as sitk
import torch

sys.path.insert(0, '/opt/app')
import hybrid_inference_20261030 as H

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
    image = sitk.GetArrayFromImage(sitk.ReadImage(str(paths[0])))
    value, parts = H.predict_hybrid(image, args.model, args.config, members,
                                    torch.float16 if args.fp16 else torch.bfloat16, return_parts=True)
    args.output.mkdir(parents=True, exist_ok=True)
    for path, v in zip(outputs, value):
        with path.open('x') as handle:
            json.dump(float(v), handle, allow_nan=False)
    print(json.dumps(dict(seconds=time.time() - start, phase1_members=6, recipea_members=len(members),
                          phase1=[float(v) for v in parts['phase1']], recipea=[float(v) for v in parts['recipea']],
                          output=[float(v) for v in value])), flush=True)


if __name__ == '__main__':
    main()
