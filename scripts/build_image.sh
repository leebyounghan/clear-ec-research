#!/usr/bin/env bash
# Build the p3b inference image from the repository root. Usage: scripts/build_image.sh [tag]
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:-clear-ec-p3b:local}
docker build --platform linux/amd64 -t "$TAG" .
echo "built $TAG"
