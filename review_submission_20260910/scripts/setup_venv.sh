#!/usr/bin/env bash
set -euo pipefail
LAMDA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LAMDA_ROOT"
LAMDA_PYTHON="${LAMDA_PYTHON:-python3.12}"
"$LAMDA_PYTHON" -c 'import platform, sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"; assert platform.system() == "Linux" and platform.machine() == "x86_64", "Use Linux x86_64 with an NVIDIA CUDA GPU; macOS/MPS is unsupported"'
if [[ -L .venv || ( -e .venv && ! -f .venv/pyvenv.cfg ) ]]; then
  echo 'Refusing to replace a symlink or non-venv .venv path.' >&2
  exit 1
fi
if [[ ! -d .venv ]]; then
  "$LAMDA_PYTHON" -m venv .venv
fi
.venv/bin/python -c 'import sys; assert sys.prefix != sys.base_prefix; assert sys.version_info[:2] == (3,12)'
.venv/bin/python -c 'from pathlib import Path; assert "include-system-site-packages = false" in Path(".venv/pyvenv.cfg").read_text().lower(), "Use an isolated venv without system-site packages"'
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  torch==2.11.0+cu126 torchvision==0.26.0+cu126 \
  --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install -r requirements-venv.txt
.venv/bin/python -m pip check
# Installation may also be tested on CPU-only Linux; inference still requires CUDA.
.venv/bin/python scripts/check_environment.py --allow-no-gpu
echo 'Installed. Run: source .venv/bin/activate'
echo 'Before inference: python scripts/check_environment.py'
