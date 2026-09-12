#!/usr/bin/env python3
"""Import a user-provided raw archive without changing the central512 dataset.

Verifies archive SHA-256, rejects unsafe tar entries, extracts MHA files into a
separate directory, and requires every existing cache row to match its original
central pixels exactly before publishing the full-frame cache.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_member(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name:
        raise ValueError(f"unsafe archive path: {name}")
    if len(p.parts) == 1 and p.name == "final_train_ids.csv":
        return p
    if not p.parts or p.parts[0] != "train_mha":
        raise ValueError(f"unexpected archive root: {name}")
    return p


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--reference", type=Path, default=Path("datasets/clear_ec_official"))
    parser.add_argument("--out", type=Path, default=Path("datasets/clear_ec_fullframe_20260904"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status_path = args.out / "import_status.json"

    def status(stage, **extra):
        save(status_path, dict(stage=stage, updated_at=time.strftime('%FT%T%z'),
                              elapsed_seconds=time.time() - started, **extra))
        print(stage, json.dumps(extra), flush=True)

    with (args.out / "import.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report_path = args.out / "fullframe_manifest.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            assert report["archive_sha256"] == args.sha256
            assert (args.out / "img_full_raw.dat").stat().st_size == int(np.prod(report["cache_shape"]))
            status("already_complete", matched=report["matched_cache_rows"])
            return
        status("verifying_archive", archive=str(args.archive), bytes=args.archive.stat().st_size)
        actual = sha256(args.archive)
        if actual.lower() != args.sha256.lower():
            raise ValueError(f"archive checksum mismatch: {actual}")
        # The completed upload is renamed only after verification. Never overwrite.
        if args.archive.name.endswith(".part"):
            completed = args.archive.with_name(args.archive.name[:-5])
            if completed.exists():
                raise FileExistsError(completed)
            args.archive.rename(completed)
            args.archive = completed
        status("extracting", archive_sha256=actual)
        seen, raw_count, total_bytes = set(), 0, 0
        with tarfile.open(args.archive, mode="r|gz") as archive:
            for member in archive:
                relative = safe_member(member.name)
                metadata_member = len(relative.parts) == 1 and relative.name == 'final_train_ids.csv'
                destination = (args.out / 'source_metadata' / relative.name if metadata_member
                               else args.out.joinpath(*relative.parts))
                if destination.is_symlink():
                    raise ValueError(f"symlink destination: {destination}")
                if member.isdir():
                    if len(relative.parts) != 1:
                        raise ValueError(f"unexpected subdirectory: {relative}")
                    destination.mkdir(exist_ok=True)
                    continue
                raw_member = len(relative.parts) == 2 and relative.suffix.lower() == '.mha'
                if not member.isfile() or not (raw_member or metadata_member):
                    raise ValueError(f"unexpected archive entry: {member.name}")
                if member.name in seen or member.size <= 0 or member.size > 64 * 1024 * 1024:
                    raise ValueError(f"duplicate/oversized/empty member: {member.name}")
                seen.add(member.name); raw_count += int(raw_member); total_bytes += member.size
                if raw_count > 12000 or total_bytes > 40 * 1024**3:
                    raise ValueError("archive exceeds expected training dataset bounds")
                destination.parent.mkdir(exist_ok=True)
                # A previous interrupted extraction can be resumed without overwriting
                # complete files; verify them against the archive before reusing them.
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(member.name)
                if destination.exists():
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                    if destination.stat().st_size != member.size or sha256(destination) != digest.hexdigest():
                        raise ValueError(f"existing extracted file differs: {destination}")
                else:
                    part = destination.with_suffix(destination.suffix + ".partial")
                    if part.is_symlink():
                        raise ValueError(part)
                    with part.open("wb") as out:
                        shutil.copyfileobj(source, out, length=8 * 1024 * 1024)
                    assert part.stat().st_size == member.size
                    part.rename(destination)
                source.close()
                if raw_member and raw_count % 500 == 0:
                    status("extracting", raw_count=raw_count, extracted_bytes=total_bytes)
        status("validating_ids", raw_count=raw_count)
        meta = np.load(args.reference / "cache_native_meta_512.npz", allow_pickle=True)
        ids = meta["ids"].astype(str); n = int(meta["N"]); sz = int(meta["SZ"])
        assert len(ids) == n and len(set(ids)) == n and sz == 512
        paths = {p.stem: p for p in (args.out / "train_mha").glob("*.mha")}
        missing = [i for i in ids if i not in paths]
        if missing:
            save(args.out / "missing_ids.json", missing)
            raise ValueError(f"{len(missing)} existing cache IDs absent from raw archive")
        extras = sorted(set(paths) - set(ids))
        with (args.reference / "final_train_ids.csv").open(newline="") as stream:
            label_ids = {row['ID'].removesuffix('.mha').removesuffix('.bmp') for row in csv.DictReader(stream)}
        source_labels = args.out / 'source_metadata' / 'final_train_ids.csv'
        source_labels_match = None
        if source_labels.exists():
            with source_labels.open(newline='') as stream:
                incoming = {r['ID'].removesuffix('.mha').removesuffix('.bmp'): r for r in csv.DictReader(stream)}
            missing_labels = [i for i in ids if i not in incoming]
            if missing_labels:
                raise ValueError(f'{len(missing_labels)} cached IDs missing from archive label CSV')
            yy = np.asarray([[float(incoming[i][m]) for m in ('CD','CV','HEX')] for i in ids])
            source_labels_match = bool(np.allclose(yy, meta['Y'], rtol=1e-5, atol=1e-7))
            if not source_labels_match:
                raise ValueError('Archive labels differ from existing cache labels; do not silently mix datasets')
        import SimpleITK as sitk
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        height, width = 972, 1296
        shape = (n, height, width)
        final_cache = args.out / "img_full_raw.dat"
        if final_cache.exists():
            raise FileExistsError(f"cache exists without complete manifest: {final_cache}")
        temporary_cache = args.out / "img_full_raw.partial.dat"
        assert not temporary_cache.is_symlink()
        full = np.memmap(temporary_cache, dtype=np.uint8, mode="w+", shape=shape)
        central = np.memmap(args.reference / "img_native_512.dat", dtype=np.uint8, mode="r", shape=(n, sz, sz))
        mismatches, image_metadata = [], []
        for row, image_id in enumerate(ids):
            path = paths[image_id]
            with path.open('rb') as stream:
                header = stream.read(8192)
            if b'ElementDataFile = LOCAL' not in header:
                raise ValueError(f"expected self-contained MHA: {path}")
            image = sitk.ReadImage(str(path))
            pixels = sitk.GetArrayFromImage(image)
            if pixels.shape != (height, width) or pixels.dtype != np.uint8:
                raise ValueError(f"unexpected raw format: {image_id} {pixels.shape} {pixels.dtype}")
            r0, c0 = (height - sz) // 2, (width - sz) // 2
            crop = pixels[r0:r0+sz, c0:c0+sz]
            if not np.array_equal(crop, central[row]):
                mismatches.append(dict(id=image_id, row=row,
                    different_pixels=int(np.count_nonzero(crop != central[row]))))
            full[row] = pixels
            image_metadata.append(dict(id=image_id, spacing=list(image.GetSpacing()),
                                       origin=list(image.GetOrigin()), direction=list(image.GetDirection())))
            if (row + 1) % 250 == 0:
                full.flush(); status("building_cache", matched_rows=row + 1, total=n, mismatches=len(mismatches))
        full.flush(); del full, central
        if mismatches:
            save(args.out / "central_mismatch.json", mismatches)
            raise ValueError(f"{len(mismatches)} images do not reproduce central512 cache")
        temporary_cache.rename(final_cache)
        np.savez_compressed(args.out / "fullframe_meta.npz", ids=ids, Y=meta["Y"], N=n, H=height, W=width)
        save(args.out / "image_geometry.json", image_metadata)
        report = dict(completed_at=time.strftime('%FT%T%z'), archive=str(args.archive), archive_sha256=actual,
            archive_bytes=args.archive.stat().st_size, raw_file_count=raw_count, extracted_bytes=total_bytes,
            reference=str(args.reference), cache_shape=list(shape), cache_dtype="uint8",
            matched_cache_rows=n, central_pixels_exact_match=True, extra_raw_ids=extras,
            archive_labels_match_cache=source_labels_match,
            extras_present_in_label_csv=[i for i in extras if i in label_ids],
            note="Existing training rows/folds and central cache are unchanged. Extra raw images are preserved but not silently added to experiments. ROI annotations are not used.")
        save(report_path, report)
        status("complete", raw_count=raw_count, matched_rows=n, extra_raw=len(extras), cache_bytes=final_cache.stat().st_size)


if __name__ == "__main__":
    main()
