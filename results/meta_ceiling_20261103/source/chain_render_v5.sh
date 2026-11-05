#!/bin/bash
# Waits for the 3000-image geometry set, renders appearance v5 over it, then writes synth_data/v5/COMPLETE.
cd /workspace/clear-ec-research/experiments/meta_ceiling_20261103
until [ -f synth_data/v1/index.jsonl ]; do sleep 120; done
/workspace/venv/bin/python -u source/synth_render_v5_20261104.py synth_data/v1 synth_data/v5 3000 && [ "$(ls synth_data/v5/images | wc -l)" -ge 3000 ] && date -u > synth_data/v5/COMPLETE
