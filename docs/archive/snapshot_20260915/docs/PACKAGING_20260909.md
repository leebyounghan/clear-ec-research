> Historical snapshot retained during the September 15 documentation cleanup.
> This is not live status or current execution authorization. See the
> [current document](../../../PACKAGING_20260909.md) for the release-facing summary.

# Submission packaging — September 9 UTC / September 10 KST

User requested packaging after a successful Docker Desktop restart. No permission
to upload to Grand Challenge or make a final scored submission was inferred.

## Selection and immutable inputs

Selected full_quality seed77 as the sixth member, with the original frozen
candidate6_full_quality.json calibration and FP16 execution. All-data weights
and the six inference modules are taken from
`experiments/region_candidate_20260907/experimental_stage`. Neither protected
staged5 nor the experimental stage was modified. The unsuccessful specialist
and statistical-residual branches are excluded.

Destination: `submission_candidate_20260909/`. Model bytes are independently
verified against the September 7 stage manifest. The public cpsam_v2 was resumed
from its original Hugging Face location after its x-linked-etag matched the
frozen SHA256; its complete local SHA256 matched too. No new weights or model
quantization was introduced.

## Container result

- Image tag: `clear-ec:full-quality77-20260909`.
- Local image manifest digest: `369bf3c9df05f8da60b184d953f71d41aca65ae244503f232d80fd907d668f4a`.
- Config SHA256: `275803d816ef934ef6665a12c693a2cd31bc1e418fdb734a7672a51404111a7d`.
- Base: PyTorch 2.11.0 CUDA12.6 cuDNN9 runtime, digest
  `3bb77138e105723dd4ed760b82fb63d8310ae3a1afc58f76e0ecf0f776568d33`.
- Architecture amd64, user10001:10001, explicit six-member config and FP16.
- Offline actual-container imports passed. Compiled CUDA arches include
  sm75, sm80, sm86 and sm90; this is not a certification of the competition host.
- Actual-container no-network/read-only filesystem/UID10001 test matched the
  reference central/native preprocessing hashes exactly and rejected malformed configs.
- Exported Docker archive was loaded back into Docker successfully with the
  same image identity, architecture, user and entrypoint.
- Image `.tar.gz`: 4,064,075,652 bytes,
  SHA256 `cfa6c2c8a234a5aefc8cf0ff0a3413bbaf7a353fc3e131cd7cddf33434896c8a`.

The official base uses an externally managed Ubuntu Python installation. The
first pip layer failed without altering the host. The container-only install was
corrected with --break-system-packages; pip check and imports subsequently passed.
No host Python packages were modified.

The Docker exporter contains manifest.json plus OCI blob metadata. The current
Grand Challenge source explicitly handles Docker25+ Config blob paths. A root
manifest.json, exactly one image, non-root user, amd64 architecture and a valid
64-character config identifier were checked. Describing all archives containing
OCI metadata as invalid would be incorrect; the issue is missing Docker manifest
compatibility. Source: grandchallenge/components/tasks.py, _get_image_manifest
and _get_image_config_file, retrieved September9.

## GPU runtime checks and limitations

Temporary pod clear-ec-package-verify used2H200 and the same pinned base image.
Installed dependency versions matched the local build for inference dependencies.
Eight fixed source images were tested in FP16 onGPU0 and BF16 onGPU1, each against
its historic CUDA13 outputs. All24 output values in each precision matched exactly.
Separate-process inference also wrote exactly the three expected scalar JSONs.
FP16 peak allocation2.641GiB, ~8.1s/warm image and10.217s separate-process image.
The BF16 check finished19:45:06UTC and the FP16 check19:44:36UTC.

An attempted old-venv comparison first failed its runtime assertion because the
persistent venv inherits the containing image's system torch. It did not run a
mislabelled CUDA13 comparison. Both subsequent jobs explicitly tested CUDA12.6
and compared against preserved CUDA13 results.

Attempting to create a deny-egress NetworkPolicy returned Forbidden. The verifier
also detected that egress was still available and failed closed. This was NOT an
offline GPU validation. No privilege/security workaround was attempted. The GPU
pod was deleted after its reports were copied to local provenance; no GPU pod
remains for this task. Actual exported-image execution on the Grand Challenge GPU
with network isolation remains a Try Out gate, not a completed check.

## Model transfer (complete)

Read-only CPU pod clear-ec-package-reader mounts the existing clear-ec-workspace
PVC and has a7200s deadline. No dataset copies or full training archives are fetched.
Long remote streams produced partial output/timeouts. A loopback-only allow-listed
range server/port-forward was also tried and then stopped; it was not faster.
No externally accessible file service was created. Existing datamover only mounts
unrelated PVCs and was not modified or used.

A second bounded transfer check used the explicitly permitted pods/proxy API,
with a loopback-only local Kubernetes proxy and a temporary application-key
protected, allow-listed endpoint inside the reader pod. The API returned503/EOF,
so this path was not used for model transfer and was stopped. No ingress,
Service, public endpoint, authentication bypass or mesh-policy change was made.

Current finite transfer implementation: code/fetch_submission_ranges.py, 2MiB
read-only ranges, up to32 concurrent transfers, durable partial chunks, full
SHA256 verification after assembly. Original contiguous partial prefixes are
preserved. This is network transfer concurrency, not32GPUs. Progress is in the
active terminal session and provenance/transfer_ranges/plan.json. The command
chains model archive generation after all original file checks pass.

Completion markers: provenance/fetch_verified.json and provenance/model_archive.json.
Both completion markers now exist: all selected weights and source files matched
the frozen manifest, and all eight archived model members independently matched
their original SHA256 after reading the compressed archive.

Final model archive: artifacts/model_full_quality77_20260909.tar.gz,
7,964,629,097 bytes, SHA256
`51d8074a5fa45a71139bd6dd12c768dea7246c43c7002f8456777984396d0830`.
Completed2026-09-09T20:36:48UTC (September10 05:36KST).
artifacts/SHA256SUMS.txt records both final archives. No upload or scored
submission has been performed; target-platform Try Out is still required.

The final exported-image CPU smoke test passed all six strict checkpoint loads
and real-image single-view forwards in139.9s, with no network, read-only root,
input and model mounts,4CPU and6GiB RAM. Its report is
test/container-checkpoints/container_checkpoint_smoke.json. This is not full
GPU inference validation.

The exact clear-ec-package-reader pod was deleted after verified transfer.
A subsequent app=clear-ec pod listing was empty; clear-ec-workspace remained
Bound at500GiB. No PVC, original training data or original weights were deleted.
