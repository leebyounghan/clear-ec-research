#!/usr/bin/env python3
"""Download public corneal-endothelium images used for domain pretraining.

The CLEAR-EC images themselves are transferred separately.  This downloader is
for the large, CC-BY 4.0 FECD flat-mount collection published as five Zenodo
records.  Downloads are resumable and are verified against Zenodo's MD5 sums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ZENODO_RECORDS = (18335072, 18340536, 18340577, 18340671, 18340673)
PRINT_LOCK = threading.Lock()


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def read_json(url: str, attempts: int = 8) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "clear-ec-research/1.0"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code in {
                429, 500, 502, 503, 504
            }
            if not retryable or attempt + 1 == attempts:
                raise
            delay = min(2 ** (attempt + 1), 30)
            log(f"[retry] metadata {url}: {exc}; sleeping {delay}s")
            time.sleep(delay)
    raise RuntimeError(f"unreachable metadata retry state for {url}")


def md5sum(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(item: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    record_id = item["record_id"]
    destination = output_dir / f"record_{record_id}" / item["name"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(item["size"])
    expected_md5 = item["checksum"].removeprefix("md5:")

    if destination.exists() and destination.stat().st_size == expected_size:
        if md5sum(destination) == expected_md5:
            log(f"[skip] {destination.name} ({expected_size / 2**20:.1f} MiB)")
            return {**item, "path": str(destination), "verified": True}
        log(f"[redo] checksum mismatch: {destination}")
        destination.unlink()

    command = [
        "curl",
        "--location",
        "--fail",
        "--show-error",
        "--silent",
        "--retry",
        "8",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--continue-at",
        "-",
        "--output",
        str(destination),
        item["url"],
    ]
    log(f"[get ] r{record_id}/{destination.name} ({expected_size / 2**20:.1f} MiB)")
    subprocess.run(command, check=True)
    actual_size = destination.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(f"size mismatch for {destination}: {actual_size} != {expected_size}")
    actual_md5 = md5sum(destination)
    if actual_md5 != expected_md5:
        raise RuntimeError(f"MD5 mismatch for {destination}: {actual_md5} != {expected_md5}")
    log(f"[done] r{record_id}/{destination.name}")
    return {**item, "path": str(destination), "verified": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/external"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    output_dir = args.root.resolve() / "zenodo_fecd_flatmount"
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for record_id in ZENODO_RECORDS:
        record = read_json(f"https://zenodo.org/api/records/{record_id}")
        license_id = record["metadata"].get("license", {}).get("id")
        if license_id != "cc-by-4.0":
            raise RuntimeError(f"record {record_id} has unexpected license {license_id!r}")
        records.append(
            {
                "record_id": record_id,
                "doi": record["doi"],
                "title": record["metadata"]["title"],
                "description": record["metadata"]["description"],
                "license": license_id,
            }
        )
        for entry in record["files"]:
            files.append(
                {
                    "record_id": record_id,
                    "name": entry["key"],
                    "size": int(entry["size"]),
                    "checksum": entry["checksum"],
                    "url": entry["links"]["self"],
                }
            )

    expected_bytes = sum(item["size"] for item in files)
    log(
        f"FECD flat-mount collection: {len(files)} TIFFs, "
        f"{expected_bytes / 2**30:.1f} GiB, {len(records)} Zenodo records"
    )
    completed: list[dict[str, Any]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(download_one, item, output_dir): item for item in files}
        for future in as_completed(futures):
            try:
                completed.append(future.result())
            except Exception as exc:  # retain other downloads and report all failures
                item = futures[future]
                message = f"r{item['record_id']}/{item['name']}: {exc}"
                errors.append(message)
                log(f"[FAIL] {message}")

    manifest = {
        "collection": "Whole flat-mounted Descemet membrane from FECD",
        "modality": "transmitted-light microscopy",
        "task_relation": "same corneal-endothelium tissue; adjacent acquisition modality",
        "license": "CC-BY-4.0",
        "records": records,
        "expected_files": len(files),
        "expected_bytes": expected_bytes,
        "verified_files": sorted(completed, key=lambda value: (value["record_id"], value["name"])),
        "errors": errors,
    }
    temporary = output_dir / f"manifest.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, output_dir / "manifest.json")
    log(f"manifest: {output_dir / 'manifest.json'}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
