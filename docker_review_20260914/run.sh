#!/usr/bin/env bash
# Archived-image inference only; no cluster connection or host Python.
set -euo pipefail
usage() {
  echo "Usage: bash run.sh INPUT_DIR MODEL_DIR EMPTY_OUTPUT_DIR [GPU_INDEX]"
}
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ $# -lt 3 || $# -gt 4 ]]; then usage >&2; exit 2; fi
[[ "$(uname -s)" == Linux ]] || { echo "Requires a Linux NVIDIA Docker host." >&2; exit 2; }
for tool in docker realpath sha256sum; do command -v "$tool" >/dev/null; done
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
input_dir=$(realpath -e -- "$1")
model_dir=$(realpath -e -- "$2")
output_dir=$(realpath -e -- "$3")
gpu_index=${4:-0}
[[ "$gpu_index" =~ ^[0-9]+$ ]] || { echo "GPU index must be a nonnegative integer." >&2; exit 2; }
for directory in "$input_dir" "$model_dir" "$output_dir"; do
  [[ -d "$directory" && "$directory" != *,* ]] || { echo "Use existing directories without commas." >&2; exit 2; }
done
[[ "$input_dir" != "$model_dir" && "$output_dir" != "$model_dir" && "$output_dir" != "$input_dir" ]] || {
  echo "Input, model and output directories must be distinct." >&2; exit 2;
}
shopt -s nullglob dotglob
output_entries=("$output_dir"/*)
[[ ${#output_entries[@]} -eq 0 ]] || { echo "Output directory must be empty." >&2; exit 2; }
[[ -f "$input_dir/inputs.json" ]] || { echo "Missing inputs.json." >&2; exit 2; }
input_images=("$input_dir"/images/corneal-specular-microscopy-image/*.mha)
[[ ${#input_images[@]} -eq 1 && -f "${input_images[0]}" ]] || { echo "Expected exactly one MHA image." >&2; exit 2; }
(cd -- "$model_dir" && sha256sum --check "$script_dir/MODEL_SHA256SUMS.txt")
image_id=sha256:275803d816ef934ef6665a12c693a2cd31bc1e418fdb734a7672a51404111a7d
[[ "$(docker image inspect "$image_id" --format '{{.Id}}')" == "$image_id" ]] || {
  echo "Load the verified original Docker archive first." >&2; exit 2;
}
exec docker run --rm --gpus "\"device=$gpu_index\"" --network none --read-only \
  --platform linux/amd64 --user 10001:10001 \
  --cpus 4 --memory 16g --shm-size 1g \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  --mount "type=bind,source=$input_dir,target=/input,readonly" \
  --mount "type=bind,source=$output_dir,target=/output" \
  --mount "type=bind,source=$model_dir,target=/opt/ml/model,readonly" \
  "$image_id"
