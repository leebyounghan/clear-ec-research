"""Validate the separate Linux inference venv, without reading challenge data."""
import argparse
import importlib.metadata
import json
import platform
import sys

EXPECTED = {
    'torch': '2.11.0+cu126', 'torchvision': '0.26.0+cu126',
    'cellpose': '4.2.1.1', 'numpy': '1.26.4', 'SimpleITK': '2.5.5',
    'opencv-python-headless': '4.11.0.86', 'scipy': '1.15.3',
    'scikit-image': '0.25.2', 'segment-anything': '1.0',
}


def validate(allow_no_gpu=False):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Use Python 3.12 for this inference environment.')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('Supported inference platform: Linux x86_64 + NVIDIA CUDA.')
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('Activate the dedicated .venv; do not install into system Python.')
    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    if versions != EXPECTED:
        raise RuntimeError(f'Dependency mismatch: {versions}; rerun setup_venv.sh in a clean venv.')
    import torch
    import torchvision
    import cellpose.models
    import SimpleITK
    import cv2
    if torch.version.cuda != '12.6':
        raise RuntimeError('Expected CUDA 12.6 PyTorch wheels, not CPU/cu130 wheels.')
    available = torch.cuda.is_available()
    if not available and not allow_no_gpu:
        raise RuntimeError('CUDA is unavailable. Check nvidia-smi, driver compatibility and GPU visibility.')
    device = None
    if available:
        device = torch.cuda.get_device_name(0)
        value = torch.ones((4, 4), device='cuda', dtype=torch.float16)
        if not torch.isfinite(value @ value).all().item():
            raise RuntimeError('FP16 CUDA smoke failed.')
        torch.cuda.synchronize()
    return {'python': platform.python_version(), 'packages': versions, 'cuda': torch.version.cuda,
            'cuda_available': available, 'device': device,
            'scope': 'Environment/import smoke only, not full checkpoint inference parity.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-no-gpu', action='store_true', help='Import-only CI check; not inference approval')
    args = parser.parse_args()
    print(json.dumps(validate(args.allow_no_gpu), indent=2))


if __name__ == '__main__':
    main()
