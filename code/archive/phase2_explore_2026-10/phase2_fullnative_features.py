#!/usr/bin/env python3
"""Extract frozen CPSAM features from every pixel of the native CLEAR-EC frame.

The image is not cropped, tiled, or resized. Four replicated rows only make its
height divisible by the encoder patch size. Two shards can run independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from phase2_fullnative_probe import FullNativeEncoder

DATA = Path("datasets/clear_ec_fullframe_20260904")
CENTRAL = Path("datasets/clear_ec_official/cache_native_meta_512.npz")
ROOT = Path("experiments/phase2_fullnative_20260929")
SHAPE = (24, 512)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_image(raw: np.ndarray, clahe) -> torch.Tensor:
    image = clahe.apply(np.ascontiguousarray(raw))
    low, high = np.percentile(image, (1, 99))
    image = np.clip((image.astype(np.float32) - low) / (high - low + 1e-6), 0, 1)
    image = np.pad(image, ((0, 4), (0, 0)), mode="edge")
    return torch.from_numpy(np.stack((image, image, image))[None]).cuda(non_blocking=True)


def main(shard: int) -> None:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("One isolated H200 GPU is required")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    report = json.loads((DATA / "fullframe_manifest.json").read_text())
    if report["cache_shape"] != [8954, 972, 1296] or not report["central_pixels_exact_match"]:
        raise ValueError("Full-resolution cache failed provenance check")
    with np.load(DATA / "fullframe_meta.npz", allow_pickle=True) as data:
        ids = data["ids"].astype(str)
    with np.load(CENTRAL, allow_pickle=True) as data:
        np.testing.assert_array_equal(ids, data["ids"].astype(str))
    if len(ids) != 8954:
        raise ValueError("Unexpected image count")
    begin, end = (0, 4477) if shard == 0 else (4477, 8954)
    output = ROOT / "features"
    output.mkdir(parents=True, exist_ok=True)
    partial = output / f"shard{shard}.partial.dat"
    final = output / f"shard{shard}.dat"
    complete = output / f"shard{shard}.json"
    if partial.exists() or final.exists() or complete.exists():
        raise FileExistsError(f"Shard {shard} output already exists")

    from cellpose import models
    original = models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
    encoder = original.net.encoder.float()
    del original
    network = FullNativeEncoder(encoder).cuda().eval()
    network.requires_grad_(False)
    images = np.memmap(DATA / "img_full_raw.dat", mode="r", dtype=np.uint8,
                       shape=(8954, 972, 1296))
    features = np.memmap(partial, mode="w+", dtype=np.float16, shape=(end - begin, *SHAPE))
    clahe = cv2.createCLAHE(2.0, (40, 30))
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    for offset, index in enumerate(range(begin, end)):
        tensor = source_image(images[index], clahe)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            fmap = network(tensor).float()
            mean = F.adaptive_avg_pool2d(fmap, (4, 6))
            maximum = F.adaptive_max_pool2d(fmap, (4, 6))
            vector = torch.cat((mean, maximum), 1).permute(0, 2, 3, 1).reshape(*SHAPE)
        features[offset] = vector.cpu().numpy().astype(np.float16)
        if (offset + 1) % 100 == 0 or offset + 1 == end - begin:
            features.flush()
            elapsed = time.monotonic() - start
            print(json.dumps({"shard": shard, "done": offset + 1, "total": end - begin,
                              "seconds_per_image": elapsed / (offset + 1),
                              "peak_gib": torch.cuda.max_memory_allocated() / 2**30}), flush=True)
    del features
    partial.rename(final)
    record = {
        "shard": shard, "start": begin, "end": end, "shape": [end - begin, *SHAPE],
        "dtype": "float16", "geometry": [972, 1296], "input_geometry": [976, 1296],
        "cropped_or_resized": False, "pool_grid": [4, 6],
        "feature_sha256": digest(final), "source_sha256": digest(Path(__file__)),
        "foundation_sha256": digest(Path("cpsam_models/cpsam_v2")),
        "ids_sha256": hashlib.sha256("\n".join(ids.tolist()).encode()).hexdigest(),
        "duration_seconds": time.monotonic() - start,
    }
    complete.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": True, **record}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    main(parser.parse_args().shard)
