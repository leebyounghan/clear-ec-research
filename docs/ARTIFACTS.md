# September 14 release artifacts

Download from [Google Drive](https://drive.google.com/drive/folders/1QyNtGhaEpePwBEyjj-UorqRkSFDRmKUn?usp=sharing).
Keep container and model archives separate; do not recompress or substitute files.

| File | Bytes | Role |
|---|---:|---|
| `clear_ec_full_quality77_20260909.tar.gz` | 4,064,075,652 | Unchanged Docker image |
| `model_full_quality77_ssl_raw_20260914.tar.gz` | 7,964,665,157 | Six model checkpoints, foundation weight and configuration |
| `SHA256SUMS.txt` | 217 | Checksums of both archives |

Model archive SHA256:

```text
e37c83db830cc5fa5215f7fd63267beb11e6c09a8778aefe2b8c6180a37a9730
```

Docker archive SHA256:

```text
cfa6c2c8a234a5aefc8cf0ff0a3413bbaf7a353fc3e131cd7cddf33434896c8a
```

Loaded Docker image ID (not an archive checksum):

```text
sha256:275803d816ef934ef6665a12c693a2cd31bc1e418fdb734a7672a51404111a7d
```

## Extracted model contents

```text
cpsam_v2
seed42.pt
seed77.pt
seed2026.pt
metric35.pt
predbag2_35.pt
full_quality_seed77.pt
candidate6_full_quality.json
```

The foundation file is required by model construction, even though each fitted
state is subsequently loaded. All eight resource hashes are in
[MODEL_SHA256SUMS.txt](../docker_review_20260914/MODEL_SHA256SUMS.txt).
The archive contains no challenge images or per-case predictions.

The September 10 platform image ID was
`730ff3f7-ea96-4180-936d-bc02a4eca762`.

## Official evaluation and shared files

Verified on September 15 using the authenticated platform UI and fresh checksum
computation over the complete Drive-downloaded archive:

| Record | Verified identity |
|---|---|
| [Official evaluation](https://clear-ec.grand-challenge.org/evaluation/7606a2b6-818f-46d0-a1fc-d060648fc2c6/) | `7606a2b6-818f-46d0-a1fc-d060648fc2c6`, succeeded, score **8.7010** |
| Submission | `a3ca83fb-7b44-4abd-8311-fbcae371481d` |
| [Attached model](https://grand-challenge.org/algorithms/lamda/models/b4137c8d-a71a-4f83-ab1f-d0635f11590b/) | `b4137c8d-a71a-4f83-ab1f-d0635f11590b`, import completed; comment `final phase 1` |
| Platform model filename | `model_full_quality77_ssl_raw_20260914.tar.gz` |
| Platform and recomputed model CRC64-NVME | **`65bbad82ffbb157e`**, exact match |
| Recomputed model SHA256 | `e37c83db830cc5fa5215f7fd63267beb11e6c09a8778aefe2b8c6180a37a9730` |
| [Attached image](https://grand-challenge.org/algorithms/lamda/images/730ff3f7-ea96-4180-936d-bc02a4eca762/) | `730ff3f7-ea96-4180-936d-bc02a4eca762`, platform image SHA256 matches the full image ID above |

The platform displays CRC64-NVME for the model archive, not SHA256. Both CRC64-NVME
and SHA256 were computed in one pass over the same 7,964,665,157-byte Drive download
using AWS CRT 0.36.3 and Python hashlib. Thus the evaluation-to-model record and
shared file identity have been checked; this is no longer an outstanding item.
CRC64 is a transfer-integrity checksum, not a cryptographic signature.
Both Docker and fresh-venv source inference have also passed. The older Docker
filename is expected: the image is unchanged and the model is mounted separately.

Aggregate evidence: [platform archive verification](PLATFORM_BINDING_20260915.json).

Use the [Docker guide](../docker_review_20260914/README.md) for verification,
extraction and execution. Both archives were downloaded without login on a
separate Linux host and passed the recorded hashes; see the
[September 15 verification](DOCKER_SMOKE_20260915.md). Evaluator GitHub access
must still be granted separately. Third-party weights retain
their upstream terms.

[Detailed historical record](archive/snapshot_20260915/docs/ARTIFACTS.md) — retained for provenance, not reviewer setup.
