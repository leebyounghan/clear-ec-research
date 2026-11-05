"""Export a verified, selected model bundle and a legacy Docker image archive."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1] / "submission_candidate_20260909"
TAG = "clear-ec:full-quality77-20260909"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_report(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def model():
    manifest = json.loads((ROOT / "provenance/fetch_verified.json").read_text())
    files = {k.removeprefix("model/"): v for k, v in manifest["files"].items()
             if k.startswith("model/")}
    assert len(files) == 8 and "full_quality_seed77.pt" in files
    assert {p.name for p in (ROOT / "model").iterdir()} == set(files)
    for name, expected in files.items():
        path = ROOT / "model" / name
        assert path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"], name
        print(f"Verified source model: {name}", flush=True)
    out = ROOT / "artifacts/model_full_quality77_20260909.tar.gz"
    assert not out.exists()
    partial = out.with_suffix(out.suffix + ".partial")
    with partial.open("xb") as stream, gzip.GzipFile(filename="", fileobj=stream, mode="wb", compresslevel=1, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            for name in sorted(files):
                path = ROOT / "model" / name
                info = archive.gettarinfo(str(path), arcname=name)
                info.uid = info.gid = 10001
                info.uname = info.gname = "algorithm"
                info.mode = 0o644
                info.mtime = 0
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
                print(f"Packed model: {name}", flush=True)
    # Independently hash every archived member, not just the source directory.
    seen = set()
    with tarfile.open(partial, "r|gz") as archive:
        for entry in archive:
            assert entry.isfile() and entry.name in files and entry.name not in seen
            h = hashlib.sha256()
            handle = archive.extractfile(entry)
            for block in iter(lambda: handle.read(8 * 1024**2), b""):
                h.update(block)
            assert h.hexdigest() == files[entry.name]["sha256"], entry.name
            seen.add(entry.name)
    assert seen == set(files)
    partial.rename(out)
    write_report(ROOT / "provenance/model_archive.json", dict(
        name=out.name, bytes=out.stat().st_size, sha256=sha(out),
        members=files, archived_hashes_verified=True, created_at=time.strftime("%FT%T%z")))
    reports = [json.loads((ROOT / "provenance" / name).read_text())
               for name in ("image_archive.json", "model_archive.json")]
    with (ROOT / "artifacts/SHA256SUMS.txt").open("x") as handle:
        for report in reports:
            handle.write(f"{report['sha256']}  {report['name']}\n")
    print(f"Model archive verified: {out}", flush=True)


def image():
    target = ROOT / "artifacts/clear_ec_full_quality77_20260909.tar"
    assert not target.exists() and not target.with_suffix(".tar.gz").exists()
    assert shutil.disk_usage(ROOT).free > 20 * 1024**3
    subprocess.run(["docker", "buildx", "build", "--builder", "clearec-package-builder",
        "--platform", "linux/amd64", "--provenance=false", "--tag", TAG,
        "--output", f"type=docker,dest={target}", str(ROOT)], check=True, timeout=3600)
    with tarfile.open(target) as archive:
        names = set(archive.getnames())
        assert "manifest.json" in names, "Missing legacy Docker manifest"
        manifest = json.load(archive.extractfile("manifest.json"))
        assert len(manifest) == 1 and TAG in manifest[0]["RepoTags"]
        config = json.load(archive.extractfile(manifest[0]["Config"]))
        assert config["architecture"] == "amd64" and config["os"] == "linux"
        assert config["config"]["User"] == "10001:10001"
        assert config["config"]["Entrypoint"][-3:] == ["--config", "candidate6_full_quality.json", "--fp16"]
        assert all(p in names for p in manifest[0]["Layers"])
    out = target.with_suffix(".tar.gz")
    partial = out.with_suffix(out.suffix + ".partial")
    with target.open("rb") as source, partial.open("xb") as output:
        with gzip.GzipFile(filename="", fileobj=output, mode="wb", compresslevel=1, mtime=0) as compressed:
            shutil.copyfileobj(source, compressed, 8 * 1024**2)
    # Validate compressed container archive structure before publishing its name.
    with tarfile.open(partial, "r|gz") as archive:
        assert "manifest.json" in {entry.name for entry in archive}
    partial.rename(out)
    write_report(ROOT / "provenance/image_archive.json", dict(
        name=out.name, bytes=out.stat().st_size, sha256=sha(out), format="legacy Docker tar",
        image_config=config, manifest=manifest, gpu_container_validated=False,
        created_at=time.strftime("%FT%T%z")))
    print(f"Image archive verified: {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["model", "image"])
    args = parser.parse_args()
    (ROOT / "artifacts").mkdir(exist_ok=True)
    {"model": model, "image": image}[args.kind]()
