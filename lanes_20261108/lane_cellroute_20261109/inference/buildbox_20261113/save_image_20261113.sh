#!/bin/bash
# Re-save only the image archive of one tier (deploy lane, 2026-11-13), after a Dockerfile-only rebuild; the model archive is unchanged.
# usage: save_image_20261113.sh TIER
set -euo pipefail
T=$1; U=$HOME/workspace/clear-ec/package_20261112/upload_20261113; IMG=clear-ec:20261112-freeze-9687076-$T
IMGF=$U/clear-ec_20261112-freeze-9687076-$T.tar.gz
echo "start $(date)"
docker save "$IMG" | pigz -p 12 > "$IMGF.part" && mv "$IMGF.part" "$IMGF"
docker image inspect "$IMG" --format 'image id {{.Id}} size {{.Size}}'
(cd "$U" && sha256sum "$(basename $IMGF)" | tee SHA256SUMS_image_$T.txt)
echo "SAVE_DONE $T $(date)"
