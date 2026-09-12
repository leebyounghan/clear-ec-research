"""Label-free global/native-local views for central-versus-full-frame ablations."""
from pathlib import Path
import json
import cv2
import numpy as np


def grid_origins(height, width):
    return np.array([(int(r), int(c)) for r in np.linspace(0, height - 256, 4)
                     for c in np.linspace(0, width - 256, 6)], dtype=np.int32)


def focus_score(raw):
    x = raw.astype(np.float32) / 255.0
    saturation = np.mean((x < .02) | (x > .98))
    return float(x.std() * np.sqrt(cv2.Laplacian(x, cv2.CV_32F).var()) * (1 - saturation))


def dihedral(image, variant):
    out = np.rot90(image, variant % 4)
    return np.ascontiguousarray(out[:, ::-1] if variant >= 4 else out)


class ContextViews:
    def __init__(self, central, ids, scope="central", quality="uniform", two_views=True,
                 raw_root="datasets/clear_ec_fullframe_20260904"):
        if scope not in ("central", "full") or quality not in ("uniform", "focus"):
            raise ValueError("invalid view configuration")
        if quality == "focus" and scope != "full":
            raise ValueError("focus ranking is precomputed for the full field only")
        self.scope, self.quality, self.two_views = scope, quality, two_views
        self.clahe = cv2.createCLAHE(2.0, (8, 8))
        self.images = central
        self.global_images = None
        self.origins = None
        self.rank = None
        if scope == "full":
            root = Path(raw_root)
            report = json.loads((root / "fullframe_manifest.json").read_text())
            assert report["central_pixels_exact_match"]
            meta = np.load(root / "fullframe_meta.npz", allow_pickle=True)
            assert np.array_equal(meta["ids"].astype(str), np.asarray(ids).astype(str))
            n, h, w = report["cache_shape"]
            self.images = np.memmap(root / "img_full_raw.dat", dtype=np.uint8, mode="r", shape=(n,h,w))
            prepared = json.loads((root / "context_views_manifest.json").read_text())
            assert prepared["archive_sha256"] == report["archive_sha256"]
            self.global_images = np.memmap(root / "img_global_256.dat", dtype=np.uint8, mode="r", shape=(n,256,256))
            with np.load(root / "context_quality.npz") as data:
                self.origins, scores = data["origins"], data["scores"]
            self.rank = np.argsort(-scores, axis=1, kind="stable")
        self.height, self.width = self.images.shape[1:]

    def normalize(self, raw):
        x = self.clahe.apply(np.ascontiguousarray(raw))
        lo, hi = np.percentile(x, (1,99))
        return np.clip((x.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)

    def global_view(self, index):
        raw = (self.global_images[index] if self.global_images is not None else
               cv2.resize(self.images[index], (256,256), interpolation=cv2.INTER_AREA))
        return self.normalize(raw)

    def bag(self, index, row, col, variant, global_view=None):
        local = self.normalize(self.images[index, row:row+256, col:col+256])
        views = [local]
        if self.two_views:
            views.insert(0, self.global_view(index) if global_view is None else global_view)
        return np.stack([np.stack((dihedral(v, variant),) * 3) for v in views])

    def train(self, index):
        if self.quality == "focus":
            row, col = self.origins[self.rank[index, np.random.randint(8)]]
            row = int(np.clip(row + np.random.randint(-32,33), 0, self.height-256))
            col = int(np.clip(col + np.random.randint(-32,33), 0, self.width-256))
        else:
            row, col = np.random.randint(self.height-256+1), np.random.randint(self.width-256+1)
        return self.bag(index, row, col, np.random.randint(8))

    def evaluate(self, index, d8):
        if self.quality == "focus":
            origins = self.origins[self.rank[index, :5]]
        else:
            h, w = self.height-256, self.width-256
            origins = [(0,0),(0,w),(h,0),(h,w),(h//2,w//2)]
        global_view = self.global_view(index) if self.two_views else None
        return np.stack([self.bag(index, int(r), int(c), variant, global_view)
                         for r,c in origins for variant in range(8 if d8 else 1)])


def prepare(root):
    """Precompute only unlabeled image context and quality, after pixel validation."""
    import fcntl
    root = Path(root)
    with (root / 'context_prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = json.loads((root / "fullframe_manifest.json").read_text())
        n,h,w = report['cache_shape']
        if (root / 'context_views_manifest.json').exists():
            saved = json.loads((root / 'context_views_manifest.json').read_text())
            assert saved['archive_sha256'] == report['archive_sha256']
            assert (root / 'img_global_256.dat').stat().st_size == n*256*256
            return
        images = np.memmap(root/'img_full_raw.dat', dtype=np.uint8, mode='r', shape=(n,h,w))
        global_path = root/'img_global_256.dat'
        if global_path.exists():
            raise FileExistsError('global cache exists without manifest')
        global_tmp = root/'img_global_256.partial.dat'
        globals_out = np.memmap(global_tmp, dtype=np.uint8, mode='w+', shape=(n,256,256))
        origins = grid_origins(h,w)
        scores = np.empty((n,len(origins)), dtype=np.float32)
        cv2.setNumThreads(1)
        for i in range(n):
            globals_out[i] = cv2.resize(images[i], (256,256), interpolation=cv2.INTER_AREA)
            scores[i] = [focus_score(images[i,r:r+256,c:c+256]) for r,c in origins]
            if (i+1)%250 == 0:
                globals_out.flush(); print(f'context quality {i+1}/{n}', flush=True)
        globals_out.flush(); del globals_out
        assert np.isfinite(scores).all()
        global_tmp.rename(global_path)
        np.savez_compressed(root/'context_quality.npz', origins=origins, scores=scores)
        value = dict(archive_sha256=report['archive_sha256'], n=n, grid=origins.tolist(),
                     note='Unlabeled sharpness/contrast heuristic; not ROI ground truth. No labels used.')
        temporary=root/'context_views_manifest.tmp'
        temporary.write_text(json.dumps(value,indent=2)); temporary.rename(root/'context_views_manifest.json')


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='datasets/clear_ec_fullframe_20260904')
    prepare(parser.parse_args().root)
