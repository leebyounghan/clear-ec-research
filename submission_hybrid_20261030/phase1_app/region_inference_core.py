"""Native regional single-image inference, independent of training caches."""
import numpy as np
import torch
from final_candidate_core import LM, LS, normalize
from fullframe_views import grid_origins, focus_score, dihedral
from native_region_views import diverse_origins


def native_views(image, policy):
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 2 or image.dtype != np.uint8 or min(image.shape) < 512:
        raise ValueError("Expected native uint8 grayscale image at least512x512")
    if policy not in ("anchor_quality", "full_quality"):
        raise ValueError("Unsupported regional policy")
    h, w = image.shape
    origins = grid_origins(h, w)
    scores = np.asarray([focus_score(image[r:r+256, c:c+256]) for r, c in origins], dtype=np.float32)
    if policy == "full_quality":
        selected = diverse_origins(origins, scores, 8)[:5]
    else:
        anchors = np.array([(h-512)//2, (w-512)//2]) + np.array([(0, 0), (256, 256), (128, 128)])
        selected = np.concatenate((anchors, diverse_origins(origins, scores, 2, anchors=anchors)))
    views = []
    for r, c in selected:
        tile = normalize(image[r:r+256, c:c+256])
        views.extend(np.stack((dihedral(tile, v),)*3) for v in range(8))
    return np.stack(views)


def load_region(net, path, policy):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    if cfg.get("region_policy") != policy or cfg.get("seed") != 77:
        raise ValueError("Wrong policy or seed")
    if cfg.get("pool") != "mean" or cfg.get("k_crops") != 2 or cfg.get("feature_agg", 0) != 0 or cfg.get("bag_pool", "mean") != "mean":
        raise ValueError("Wrong pooling")
    if cfg.get("objective") != "mixed_mape":
        raise ValueError("Wrong objective")
    np.testing.assert_array_equal(cfg["lm"], LM.tolist())
    np.testing.assert_array_equal(cfg["ls"], LS.tolist())
    net.load_state_dict(ck["model"], strict=True)
    net.eval()
    return cfg


@torch.no_grad()
def predict_views(net, views, dtype=torch.bfloat16):
    pieces = []
    for start in range(0, len(views), 8):
        x = torch.from_numpy(views[start:start+8]).to("cuda")
        with torch.autocast("cuda", dtype=dtype):
            pieces.append(net(x).float())
    encoded = torch.cat(pieces).mean(0)
    value = torch.exp(encoded * torch.as_tensor(LS, device="cuda") + torch.as_tensor(LM, device="cuda")).cpu().numpy()
    if not np.isfinite(value).all() or not (value > 0).all():
        raise ValueError("Nonfinite/nonpositive regional output")
    return value
