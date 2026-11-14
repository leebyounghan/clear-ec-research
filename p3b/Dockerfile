# Tier p3b (deploy lane, 2026-11-14, D-COORD-P3-PREP): the p2 context (package_20261114/p2) with the ViT-L s46 and s47 half of V replaced by
# the encoder-LR-2e-5 full-data run (fp16 copy, model_swap.tsv) and the CD constants of freeze_cd_params_p3b_20261114.json;
# build with --build-arg TIER=p3b, tag clear-ec:<build date>-freeze-9687076-p3b (never latest). Model archive = the tier-p archive with the
# files of model_swap.tsv swapped (rebuild_model_archive_20261114.py). Everything else is the p2 image:
# Tier p2 (deploy lane, 2026-11-14, D-COORD-P2-PREP): the tier-p context (package_20261112/p) with D = DINOv3-B s46b alone (CD constants
# lane_cd_20261108/reads/freeze_cd_params_p2_20261114.json) and the four read-once switches below; build with --build-arg TIER=p2,
# tag clear-ec:<build date>-freeze-9687076-p2 (never latest). Runs with the tier-p model archive unchanged (p2 reads a subset of it).
# 11-12 freeze candidate image (deploy lane, D-PKG-001): D-CD-001 + D-CV-001 + D-HEX-004 + D-HEX-005 through recipe_infer.
# Built like submissions 1 and 2 (results/submission1_package_20261031/PACKAGE.md): same base image (PyTorch 2.11.0, CUDA 12.6 -> runs on T4 sm75
# and A10G sm86), weights NOT in the image but in a separate model archive mounted read-only at /opt/ml/model, Grand Challenge I/O unchanged.
# Build context = package_20261112/context (stage_package.py): the research-tree code files at their repo-relative paths, model_files.txt,
# fp16_files.txt, base_shapes.json, requirements / constraints. Tag: clear-ec:20261112-freeze-<sha7 of manifests/deploy.json>; never 'latest'.
FROM --platform=linux/amd64 pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime@sha256:3bb77138e105723dd4ed760b82fb63d8310ae3a1afc58f76e0ecf0f776568d33
ARG TIER=full
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MALLOC_ARENA_MAX=2 \
    HOME=/tmp XDG_CACHE_HOME=/tmp/.cache TORCH_HOME=/tmp/torch MPLCONFIGDIR=/tmp/mpl \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    CLEAR_EC_FP16_LIST=/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference/docker/fp16_files.txt \
    CLEAR_EC_SKIP_BASE=/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference/docker/base_shapes.json \
    PYTHONPATH=/workspace/clear-ec-research/experiments/phase2_foundation_20261002/runtime \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CLEAR_EC_UNLOAD=1
# tier p2 read-once switches (2026-11-14, D-COORD-P2-PREP): dino_cv reuses the identity flip of dino_cd (members.py); recipe A s46 epoch_20, the six
# detector files hex6 shares with peaks / other_peaks, and cpsam_v2 are read once per job (pack_hooks.py). Each 0 / unset = the tier-p code path.
ENV CLEAR_EC_RO_DINOCV=1 CLEAR_EC_RO_RECIPEA=1 CLEAR_EC_RO_HEX6=1 CLEAR_EC_RO_CPSAM=1
# CLEAR_EC_UNLOAD=1: every pod test ran with it; without it all stages' models stay on the GPU (~17 GB) and the T4 (15 GB) runs out of memory.
# PYTHONPATH as on the research pod (2026-11-12 real-T4 check): cellpose.vit imports the CPDINO backbones from runtime/dinov3 inside a bare
# try/except; without the path every CPDINO member raised NameError and the entry answered every image with the fallback constants.
WORKDIR /workspace/clear-ec-research
COPY requirements.txt constraints.txt /opt/build/
RUN python -m pip install --break-system-packages --no-cache-dir -c /opt/build/constraints.txt -r /opt/build/requirements.txt \
    && python -m pip check \
    && python -c "import torch, timm, transformers, cellpose.models, SimpleITK, cv2, scipy; assert torch.__version__.split('+')[0] == '2.11.0'; assert timm.__version__ == '1.0.20'; assert transformers.__version__ == '5.3.0'" \
    && python -m pip freeze > /opt/build/runtime-lock.txt
RUN groupadd --gid 10001 algorithm && useradd --uid 10001 --gid algorithm --no-create-home algorithm \
    && mkdir -p /opt/ml/model /input /output && chown algorithm:algorithm /output
# research-tree code (frozen copies at their repo-relative paths: recipe_infer, members, the lanes' apply modules and builders)
COPY code/ /workspace/clear-ec-research/
COPY model_files.txt /opt/build/model_files.txt
# every weight path of the manifest points into the read-only model mount (dangling until Grand Challenge mounts the model archive)
RUN while read -r rel; do mkdir -p "$(dirname "/workspace/clear-ec-research/$rel")" && ln -s "/opt/ml/model/$rel" "/workspace/clear-ec-research/$rel"; done < /opt/build/model_files.txt
# reduced tiers (build argument): the tier manifest and the lanes' refit parameters must exist in the context; full = the freeze candidate
RUN set -e; I=/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference; \
    if [ "$TIER" != "full" ]; then test -f "$I/manifests/deploy_$TIER.json" && test -f "$I/params/deploy_$TIER.json" \
       || { echo "tier $TIER needs manifests/deploy_$TIER.json and params/deploy_$TIER.json (lane refits)"; exit 1; }; \
       cp "$I/manifests/deploy_$TIER.json" "$I/manifests/deploy.json"; cp "$I/params/deploy_$TIER.json" "$I/params/deploy.json"; fi; \
    echo "$TIER" > /opt/build/TIER
# the image runs as uid 10001 but COPY keeps the research tree's modes: phase2_lejepa_20261003.py is mode 600 there, so every CD-family member
# raised PermissionError and the entry answered every image with the fallback constants (2026-11-13 build-box test; the pod, Colab and the earlier
# tests all ran as root). Everything readable, then a loud build-time check as the runtime user (symlinks into /opt/ml/model are not followed).
RUN chmod -R a+rX /workspace/clear-ec-research /opt/build
USER 10001:10001
RUN bad="$(find /workspace/clear-ec-research /opt/build \( -type f ! -readable \) -o \( -type d ! -executable \) | head -3)"; \
    test -z "$bad" || { echo "not readable by uid 10001: $bad"; exit 1; }
ENTRYPOINT ["python", "/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference/docker/gc_entry.py"]
