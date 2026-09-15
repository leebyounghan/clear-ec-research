#!/usr/bin/env python3
"""Build a provenance-aware image manifest for CLEAR-EC domain SSL.

Small specular/confocal datasets remain in their original form.  The very
large FECD whole-mount TIFFs are converted to a compact, multi-scale set of
high-information 512-pixel tiles so they can be sampled efficiently during
MAE/JEPA pretraining.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import tifffile


SMALL_SOURCES = (
    {
        "name": "srep_topcon_sp3000",
        "pattern": "srep_topcon_sp3000/raw/images_and_labels/*.tiff",
        "modality": "non-contact specular microscopy (Topcon SP-3000)",
        "relation": "same tissue and same acquisition modality",
        "license": "research use; citation required",
        "citation": "Boehringer et al., Scientific Reports 2019, PMID 30894636",
    },
    {
        "name": "rodrep_confocal_endothelium",
        "pattern": "rodrep_confocal/raw/image*.png",
        "modality": "in-vivo confocal microscopy (ConfoScan 4)",
        "relation": "same tissue and same morphometry task; adjacent modality",
        "license": "ROI non-commercial research license; no redistribution",
        "citation": "Selig et al., BMC Medical Imaging 2015;15:13",
    },
    {
        "name": "github_cornea_cells",
        "pattern": "quarantine_unverified/cornea_cells_github/dataset/*.tif",
        "modality": "corneal-endothelium microscopy",
        "relation": "same tissue; acquisition provenance not documented upstream",
        "license": "MIT repository license",
        "citation": "svdeepak99/U-Net_Segmentation-Cornea_Cells",
    },
)


def sha256sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    image = np.squeeze(image)
    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ValueError(f"unsupported image shape: {image.shape}")
    if image.shape[-1] in (3, 4):
        rgb = image[..., :3].astype(np.float32)
    elif image.shape[0] in (3, 4):
        rgb = np.moveaxis(image[:3], 0, -1).astype(np.float32)
    else:
        # Multipage TIFFs are not expected here; use the first plane explicitly.
        return as_gray(image[0])
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)
    values = image[finite]
    low, high = np.percentile(values, (1.0, 99.0))
    if high <= low + 1e-6:
        low, high = float(values.min()), float(values.max())
    output = np.clip((image - low) / max(high - low, 1e-6), 0.0, 1.0)
    output[~finite] = 0
    return np.rint(output * 255.0).astype(np.uint8)


def perceptual_hash(image: np.ndarray) -> str:
    gray = normalize_uint8(as_gray(image))
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coefficients = cv2.dct(small)[:8, :8]
    threshold = np.median(coefficients.flat[1:])
    bits = coefficients > threshold
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def read_image(path: Path) -> np.ndarray:
    # Some public corpora carry JPEG-encoded files with a .tiff suffix, while
    # another uses legacy TIFF compression that is not supported by every
    # tifffile/imagecodecs combination.  OpenCV sniffs the actual file header,
    # so prefer it for these small inputs and keep tifffile as a fallback.
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is not None:
        return image
    if path.suffix.lower() in (".tif", ".tiff"):
        return tifffile.imread(str(path))
    raise ValueError(f"could not decode {path}")


def small_entries(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    entries: list[dict[str, Any]] = []
    errors: list[str] = []
    for source in SMALL_SOURCES:
        for path in sorted(root.glob(source["pattern"])):
            try:
                image = read_image(path)
                gray = as_gray(image)
                entries.append(
                    {
                        "path": str(path.resolve()),
                        "source": source["name"],
                        "modality": source["modality"],
                        "relation": source["relation"],
                        "license": source["license"],
                        "citation": source["citation"],
                        "height": int(gray.shape[0]),
                        "width": int(gray.shape[1]),
                        "sha256": sha256sum(path),
                        "phash": perceptual_hash(gray),
                        "sampling_tier": 1 if source["name"] == "srep_topcon_sp3000" else 2,
                    }
                )
            except Exception as exc:
                errors.append(f"{path}: {exc}")
    return entries, errors


def quality_score(tile: np.ndarray) -> float:
    normalized = normalize_uint8(tile)
    contrast = float(np.percentile(normalized, 95) - np.percentile(normalized, 5)) / 255.0
    mid_fraction = float(np.mean((normalized > 8) & (normalized < 247)))
    laplacian = float(cv2.Laplacian(normalized, cv2.CV_32F).var())
    return contrast * math.sqrt(max(laplacian, 0.0) + 1.0) * max(mid_fraction, 0.05)


def choose_tiles(
    image: np.ndarray,
    window: int,
    number: int,
    rng: np.random.Generator,
) -> list[tuple[float, int, int, np.ndarray]]:
    height, width = image.shape[:2]
    if height < window or width < window:
        return []
    candidates: list[tuple[float, int, int, np.ndarray]] = []
    for _ in range(max(number * 8, 96)):
        y = int(rng.integers(0, height - window + 1))
        x = int(rng.integers(0, width - window + 1))
        raw = image[y : y + window, x : x + window]
        gray = as_gray(raw)
        if window != 512:
            gray = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
        normalized = normalize_uint8(gray)
        candidates.append((quality_score(normalized), y, x, normalized))
    candidates.sort(key=lambda item: item[0], reverse=True)

    selected: list[tuple[float, int, int, np.ndarray]] = []
    minimum_distance = window * 0.35
    for candidate in candidates:
        _, y, x, _ = candidate
        if all(math.hypot(y - sy, x - sx) >= minimum_distance for _, sy, sx, _ in selected):
            selected.append(candidate)
            if len(selected) == number:
                break
    return selected


def tile_one_flatmount(
    source_path: Path,
    output_root: Path,
    tiles_per_scale: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    relative_group = source_path.parent.name
    output_dir = output_root / relative_group / source_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_marker = output_dir / ".complete.json"
    if complete_marker.exists():
        payload = json.loads(complete_marker.read_text(encoding="utf-8"))
        if all(Path(item["path"]).exists() for item in payload["entries"]):
            return payload["entries"], []

    errors: list[str] = []
    try:
        raw = tifffile.imread(str(source_path))
        image = as_gray(raw)
        del raw
        seed = int(hashlib.sha256(str(source_path).encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        entries: list[dict[str, Any]] = []
        for window in (512, 1024, 2048):
            for rank, (score, y, x, tile) in enumerate(
                choose_tiles(image, window=window, number=tiles_per_scale, rng=rng)
            ):
                destination = output_dir / f"w{window}_r{rank:02d}_y{y}_x{x}.png"
                if not destination.exists():
                    ok = cv2.imwrite(str(destination), tile, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    if not ok:
                        raise RuntimeError(f"failed to write {destination}")
                entries.append(
                    {
                        "path": str(destination.resolve()),
                        "source_path": str(source_path.resolve()),
                        "source": "zenodo_fecd_flatmount",
                        "modality": "transmitted-light microscopy",
                        "relation": "same corneal-endothelium tissue; adjacent modality",
                        "license": "CC-BY-4.0",
                        "citation": "Maurin and Thuret, Zenodo 2026, records 18335072-18340673",
                        "height": 512,
                        "width": 512,
                        "source_window": window,
                        "source_y": y,
                        "source_x": x,
                        "quality_score": score,
                        "sha256": sha256sum(destination),
                        "phash": perceptual_hash(tile),
                        "sampling_tier": 2,
                    }
                )
        payload = {"source": str(source_path.resolve()), "entries": entries}
        temporary = output_dir / f".complete.json.tmp.{os.getpid()}"
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, complete_marker)
        return entries, []
    except Exception as exc:
        errors.append(f"{source_path}: {exc}")
        return [], errors


def deduplicate(entries: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    unique: list[dict[str, Any]] = []
    duplicates: list[dict[str, str]] = []
    seen_sha: dict[str, str] = {}
    seen_phash: dict[tuple[str, str], str] = {}
    for entry in entries:
        if entry["sha256"] in seen_sha:
            duplicates.append(
                {"path": entry["path"], "duplicate_of": seen_sha[entry["sha256"]], "reason": "sha256"}
            )
            continue
        # Exact perceptual-hash matches are only removed within the same source,
        # preventing aggressive deletion of distinct but regular cell mosaics.
        phash_key = (entry["source"], entry["phash"])
        if phash_key in seen_phash:
            duplicates.append(
                {"path": entry["path"], "duplicate_of": seen_phash[phash_key], "reason": "phash"}
            )
            continue
        seen_sha[entry["sha256"]] = entry["path"]
        seen_phash[phash_key] = entry["path"]
        unique.append(entry)
    return unique, duplicates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/external"))
    parser.add_argument("--tiles-per-scale", type=int, default=16)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--small-only", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    tile_root = root / "ssl_tiles" / "zenodo_fecd_flatmount"
    tile_root.mkdir(parents=True, exist_ok=True)
    entries, errors = small_entries(root)

    if not args.small_only:
        flatmount_paths = sorted((root / "zenodo_fecd_flatmount").glob("record_*/*.tif"))
        with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool:
            futures = {
                pool.submit(tile_one_flatmount, path, tile_root, args.tiles_per_scale): path
                for path in flatmount_paths
            }
            for future in as_completed(futures):
                new_entries, new_errors = future.result()
                entries.extend(new_entries)
                errors.extend(new_errors)
                print(
                    f"processed {futures[future].name}: +{len(new_entries)} tiles, "
                    f"errors={len(new_errors)}",
                    flush=True,
                )

    unique, duplicates = deduplicate(entries)
    unique.sort(key=lambda item: (item["sampling_tier"], item["source"], item["path"]))
    manifest_path = root / "ssl_manifest.jsonl"
    temporary = root / f"ssl_manifest.jsonl.tmp.{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as stream:
        for entry in unique:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.replace(temporary, manifest_path)
    summary = {
        "manifest": str(manifest_path),
        "entries": len(unique),
        "sources": {
            source: sum(entry["source"] == source for entry in unique)
            for source in sorted({entry["source"] for entry in unique})
        },
        "duplicates_removed": duplicates,
        "errors": errors,
    }
    (root / "ssl_manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
