"""Single-MHA convenience wrapper around the unchanged submitted entrypoint."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ('cell-density', 'coefficient-of-variation', 'hexagonality')


def select_device(index, environ):
    visible = environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        return str(index)
    devices = [x.strip() for x in visible.split(',') if x.strip() and x.strip() != '-1']
    if index >= len(devices):
        raise ValueError('--gpu is outside the existing CUDA_VISIBLE_DEVICES allocation.')
    return devices[index]


def validate_paths(image, model, output):
    if not image.is_file() or image.suffix.lower() != '.mha':
        raise ValueError('Provide an existing self-contained .mha image.')
    manifest = json.loads((ROOT / 'provenance/model_archive.json').read_text())
    for name in manifest['members']:
        path = model / name
        if not path.is_file():
            raise FileNotFoundError(f'Missing checkpoint resource: {path}')
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise FileExistsError('Output must be absent or an empty directory; existing results are never replaced.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=ROOT / 'checkpoints/model')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0, help='Index within CUDA_VISIBLE_DEVICES, or physical index if unset')
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error('--gpu must be nonnegative')
    # Reject an output symlink before resolving paths.
    validate_paths(args.image, args.model, args.output)
    image, model, output = args.image.resolve(), args.model.resolve(), args.output.resolve()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=select_device(args.gpu, os.environ), HF_HUB_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               MKL_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, str(ROOT / 'scripts/check_environment.py')], env=env, check=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/verify_release.py')], env=env, check=True)
    # No resizing or pixel decoding: copy the exact MHA into a temporary socket layout.
    with tempfile.TemporaryDirectory(prefix='lamda-input-') as temporary:
        source = Path(temporary)
        image_dir = source / 'images/corneal-specular-microscopy-image'
        image_dir.mkdir(parents=True)
        shutil.copyfile(image, image_dir / 'case.mha')
        (source / 'inputs.json').write_text(json.dumps([
            {'socket': {'slug': 'corneal-specular-microscopy-image'}}]))
        subprocess.run([sys.executable, str(ROOT / 'app/inference.py'), '--input', str(source),
                        '--output', str(output), '--model', str(model), '--config',
                        'candidate6_full_quality.json', '--fp16'], env=env, check=True)
    for name in OUTPUTS:
        value = json.loads((output / f'{name}.json').read_text())
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid scalar output: {name}')
    print(f'Wrote 3 scalar JSON files to {output}')


if __name__ == '__main__':
    main()
