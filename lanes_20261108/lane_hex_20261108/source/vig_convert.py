"""Convert the Vigueras-Guillen TVST 2020 Keras weights (HDF5) to a tensors-only PyTorch file (HEX lane, 2026-11-11).
The HDF5 files are untrusted data. Rules (coordinator 2026-11-11): python -I; h5py 3.11.0 from lane_hex_20261108/pylib only
(wheel sha256 aa6ae84a14103e8dc19266ef4c3e5d7c00b68f21d07f2966f0ca7bdb6c2761fb, checked against PyPI); datasets only - the tree
is walked link by link, every link must be a hard link (soft or external links abort), no attribute is read, only float
datasets are read; nothing from the authors' repository is imported or run. Dataset path <layer>/.../<weight>:0 -> key
"<layer>.<weight>". Output: {"<layer>.<weight>": tensor} in PyTorch layout (conv and transposed-conv kernels both permuted
(kh, kw, a, b) -> (b, a, kh, kw)), loaded later with torch.load(weights_only=True).
usage: vig_convert.py IN.h5 OUT.pt"""
import sys
import numpy as np
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
sys.path.insert(0, LANE + '/pylib')
import h5py
import torch
assert h5py.__version__ == '3.11.0', h5py.__version__


def walk(g, prefix, out):
    for name in g.keys():
        ln = g.get(name, getlink=True)
        if not isinstance(ln, h5py.HardLink): raise SystemExit(f'refused: {prefix}{name} is a {type(ln).__name__}')
        obj = g[name]
        if isinstance(obj, h5py.Group): walk(obj, f'{prefix}{name}/', out)
        elif isinstance(obj, h5py.Dataset):
            if obj.dtype.kind != 'f' or obj.dtype.hasobject: raise SystemExit(f'refused: {prefix}{name} has dtype {obj.dtype}')
            if obj.external: raise SystemExit(f'refused: {prefix}{name} uses external storage')
            out[f'{prefix}{name}'] = np.asarray(obj[()])
        else: raise SystemExit(f'refused: {prefix}{name} is {type(obj).__name__}')


if __name__ == '__main__':
    raw = {}
    with h5py.File(sys.argv[1], 'r') as f: walk(f, '', raw)
    T = {}
    for path, a in raw.items():
        parts = path.split('/')
        if parts[0] == 'model_weights': parts = parts[1:]
        if parts[0] == 'optimizer_weights': continue
        key = f"{parts[0]}.{parts[-1].split(':')[0]}"; assert key not in T, key
        if a.ndim == 4: a = a.transpose(3, 2, 0, 1)
        T[key] = torch.from_numpy(np.ascontiguousarray(a.astype(np.float32)))
    torch.save(T, sys.argv[2])
    layers = sorted({k.rsplit('.', 1)[0] for k in T}); print(len(T), 'tensors in', len(layers), 'layers;', sum(t.numel() for t in T.values()), 'parameters', flush=True)
    print('non-block layers:', [l for l in layers if not l.startswith(('block_', 'down_', 'up_'))], flush=True)
    print('weight kinds:', sorted({k.rsplit('.', 1)[1] for k in T}), flush=True)
