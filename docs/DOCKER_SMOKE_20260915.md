# Released-container smoke test — September 15, 2026

The September 14 model package passed one-image inference on a separate Linux
NVIDIA Docker host. This is an execution check, not an accuracy evaluation.

| Item | Verified result |
|---|---|
| Source | Fresh HTTPS clone of `leebyounghan/clear-ec-research` |
| Tested commit | `f1d8bb446cf19b6871b2b923bebc9a98316b614a` |
| Checkout | Clean before and after; `docker_review_20260914/run.sh` unchanged |
| Hardware | NVIDIA RTX 4090, 24 GiB; driver 595.58.03 |
| Runtime | Linux x86_64, Docker 27.4.0; no Kubernetes configuration |
| Downloads | Complete Docker and model archives from the linked Drive folder, without login/cookies |
| Identity | Both archive SHA256s, all eight model-resource hashes, and exact Docker image ID passed |
| Input | One authorized original-resolution MHA image; not redistributed |
| Inference | Six ensemble members; container log reported 13.503 seconds |
| End-to-end runner | 45.353 seconds, including resource hashing and container startup |
| Output | Three expected JSON files, each containing a finite positive numeric scalar |
| Completion | Exit code 0; test container removed; no training started |

The output files were `cell-density.json`, `coefficient-of-variation.json`, and
`hexagonality.json`. Raw input and individual predictions are not included here.
The container used its normal GPU-enabled, network-disabled, read-only-root
runner with 4 CPUs and a 16 GB host-memory limit.

The release identities are recorded in [ARTIFACTS.md](ARTIFACTS.md). This test
used the released image rather than rebuilding it from source. It does not
establish T4 runtime/memory behavior, cross-device numerical parity, model
accuracy, full training reproducibility, or evaluator repository permissions.

This record supplements the technical report with a subsequent runtime check;
the tested inference code and released weights are unchanged by documentation
updates after the tested commit.
