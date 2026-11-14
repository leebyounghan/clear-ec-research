#!/bin/bash
# Upload artifacts for one tier of the 11-12 freeze image (deploy lane, 2026-11-13, PI-approved build; no registry push, no upload).
#   image:  docker save | pigz           -> upload_20261113/clear-ec_20261112-freeze-9687076-<tier>.tar.gz
#   model:  tar of <tier>/model, paths relative to the folder Grand Challenge extracts to /opt/ml/model (as submission 1's archive, './...'),
#           owner 0:0 and every file world-readable (--mode=a+rX): the image runs as uid 10001 and cpsam_v2 is mode 600 in the staged tree
#   check:  extract the model archive into test_model_<tier>/ (what the container tests mount), compare file count, bytes and a sha256 tree
#           hash with <tier>/model; sha256 of both archives into SHA256SUMS_<tier>.txt
# usage: pack_tier_20261113.sh TIER
set -euo pipefail
T=$1; B=$HOME/workspace/clear-ec/package_20261112; U=$B/upload_20261113; IMG=clear-ec:20261112-freeze-9687076-$T
mkdir -p "$U"; cd "$B"
IMGF=$U/clear-ec_20261112-freeze-9687076-$T.tar.gz; MODF=$U/model_freeze-9687076-${T}_20261112.tar.gz
echo "start $(date)"
( docker save "$IMG" | pigz -p 12 > "$IMGF.part" && mv "$IMGF.part" "$IMGF" && echo "image saved $(date)" ) &
PI=$!
tar -C "$T/model" --sort=name --owner=0 --group=0 --numeric-owner --mode=a+rX -cf - . | pigz -p 12 > "$MODF.part"
mv "$MODF.part" "$MODF"; echo "model archived $(date)"
rm -rf "$B/test_model_$T"; mkdir -p "$B/test_model_$T"
tar -C "$B/test_model_$T" -xzf "$MODF"; echo "model extracted $(date)"
th() { (cd "$1" && find . -type f | LC_ALL=C sort | xargs -d '\n' sha256sum | sha256sum | cut -c1-16); }
for d in "$T/model" "test_model_$T"; do
  echo "$d files $(find $d -type f | wc -l) bytes $(find $d -type f -printf '%s\n' | awk '{s+=$1} END {print s}') tree $(th $d) not_world_readable $(find $d -type f ! -perm -o=r | wc -l)"
done
wait $PI
docker image inspect "$IMG" --format 'image id {{.Id}} size {{.Size}}'
(cd "$U" && sha256sum "$(basename $IMGF)" "$(basename $MODF)" | tee SHA256SUMS_$T.txt)
ls -l "$U"
echo "PACK_DONE $T $(date)"
