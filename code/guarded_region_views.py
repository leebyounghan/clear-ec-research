"""Image-only artifact rejection before spatially diverse native crop selection.

These are fixed quality heuristics, NOT cell/ROI ground truth. No labels or
dataset-level fitted thresholds are used. The frozen original policy is intact.
"""
import numpy as np
from native_region_views import NativeRegionViews
from fullframe_views import grid_origins, focus_score

VERSION = "artifact_guard_v1"


def tile_quality(raw):
    if raw.shape != (256, 256) or raw.dtype != np.uint8:
        raise ValueError("Expected native uint8 256-square")
    blocks = raw.reshape(8, 32, 8, 32).transpose(0, 2, 1, 3).reshape(64, 1024)
    lo, hi = np.percentile(blocks, [5, 95], axis=1)
    means = blocks.mean(1)
    clipped = ((blocks <= 5) | (blocks >= 250)).mean(1)
    informative = ((hi-lo >= 12) & (means >= 12) & (means <= 243) & (clipped < .35)).mean()
    saturation = ((raw <= 5) | (raw >= 250)).mean()
    acceptable = informative >= .60 and saturation <= .20
    return float(informative), float(saturation), bool(acceptable)


def choose(origins, scores, occupancy, saturation, acceptable, count=8):
    origins = np.asarray(origins, dtype=np.int32)
    scores, occupancy, saturation = map(np.asarray, (scores, occupancy, saturation))
    acceptable = np.asarray(acceptable, dtype=bool)
    assert len(origins) >= count and np.isfinite(scores).all() and (scores >= 0).all()
    robust = scores * occupancy**2 * (1-saturation)**2
    order = np.argsort(-robust, kind="stable")
    chosen = []
    # Exhaust acceptable tiles first. If fewer than count exist, recordable
    # least-bad fallback preserves the fixed view budget and never silently fails.
    for group in (order[acceptable[order]], order[~acceptable[order]]):
        remaining = list(map(int, group))
        while remaining and len(chosen) < count:
            if not chosen:
                ix = remaining[0]
            else:
                distance = np.linalg.norm(origins[remaining, None]-origins[None, chosen], axis=-1).min(1)
                utility = robust[remaining] * (.5 + .5*np.clip(distance/512., 0, 1))
                ix = remaining[int(np.argmax(utility))]
            chosen.append(ix); remaining.remove(ix)
    assert len(chosen) == count and len(set(chosen)) == count
    return origins[chosen], np.asarray(chosen)


def native_origins(image):
    origins = grid_origins(*image.shape)
    tiles = [image[r:r+256, c:c+256] for r, c in origins]
    quality = [tile_quality(t) for t in tiles]
    return choose(origins, np.asarray([focus_score(t) for t in tiles],np.float32),
                  np.asarray([q[0] for q in quality],np.float32),
                  np.asarray([q[1] for q in quality],np.float32), [q[2] for q in quality])[0]


class GuardedRegionViews(NativeRegionViews):
    def __init__(self, central, ids, selection_path,
                 raw_root="datasets/clear_ec_fullframe_20260904"):
        super().__init__(central, ids, "full_quality", raw_root)
        with np.load(selection_path) as d:
            np.testing.assert_array_equal(d["ids"].astype(str), np.asarray(ids).astype(str))
            assert str(d["version"]) == VERSION
            self.selected = d["selected"].copy()
        assert self.selected.shape == (len(ids), 8, 2)
        assert (self.selected >= 0).all()
        assert (self.selected <= np.array([self.height-256,self.width-256])).all()

    def train_origin(self, index):
        # Exactly the same three RNG calls as original full_quality. Rejected
        # jitter falls back to its unjittered selected tile, not extra random draws.
        base = self.selected[index, np.random.randint(8)]
        proposed = np.clip(base + np.random.randint(-32,33,size=2),
                           [0,0], [self.height-256,self.width-256])
        r,c = proposed
        return proposed if tile_quality(self.images[index,r:r+256,c:c+256])[2] else base
