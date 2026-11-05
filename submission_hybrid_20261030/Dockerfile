# Submission-1 hybrid container: Phase-1 six-member CPSAM package + three whole-frame CPDINO (recipe A) members.
# Same base image as the validated Phase-1 submission (PyTorch 2.11.0, CUDA 12.6 -> runs on the T4 / sm75).
# Model weights are mounted read-only at /opt/ml/model (Phase-1 layout unchanged, plus recipea/ subfolder).
FROM --platform=linux/amd64 pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime@sha256:3bb77138e105723dd4ed760b82fb63d8310ae3a1afc58f76e0ecf0f776568d33
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MALLOC_ARENA_MAX=2 \
    HOME=/tmp XDG_CACHE_HOME=/tmp/.cache TORCH_HOME=/tmp/torch \
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
WORKDIR /opt/app
COPY requirements.txt constraints.txt /opt/app/
# timm is the only addition over Phase 1 (CPDINO ViT-B backbone); dinov3 is vendored as source, not installed.
RUN python -m pip install --break-system-packages --no-cache-dir -c constraints.txt -r requirements.txt \
    && python -m pip check \
    && python -c "import torch, timm, cellpose.models, SimpleITK, cv2; assert torch.__version__.split('+')[0] == '2.11.0'; assert torch.version.cuda == '12.6'; assert timm.__version__ == '1.0.20'" \
    && python -m pip freeze > /opt/app/runtime-lock.txt
RUN groupadd --gid 10001 algorithm \
    && useradd --uid 10001 --gid algorithm --no-create-home algorithm \
    && mkdir -p /opt/ml/model /input /output \
    && chown algorithm:algorithm /output
# Phase-1 app (unchanged files), recipe A sources (frozen research copies), vendored dinov3 runtime, hybrid entrypoint.
COPY phase1_app/*.py /opt/app/phase1_app/
COPY recipea_src/*.py /opt/app/recipea_src/
COPY runtime/dinov3 /opt/app/runtime/dinov3
COPY hybrid_inference_20261030.py hybrid_guard_20261031.py hybrid_entrypoint.py /opt/app/
# The frozen recipe A sources resolve the CPDINO backbone by the research-tree relative path; map it onto the
# read-only model mount instead of editing the frozen sources. hybrid_entrypoint chdir()s to /opt/app.
RUN mkdir -p /opt/app/experiments/phase2_foundation_20261002/weights/cellpose \
    && ln -s /opt/ml/model/cpdino/cpdino-vitb /opt/app/experiments/phase2_foundation_20261002/weights/cellpose/cpdino-vitb
ENV PYTHONPATH=/opt/app/runtime
USER 10001:10001
# Start-up asserts the memory-efficient SDPA kernel for 20336 tokens before touching the input (loud by design);
# per-image failures fall back (hybrid_guard_20261031) instead of failing the evaluation.
ENTRYPOINT ["python", "/opt/app/hybrid_entrypoint.py", "--config", "candidate6_full_quality.json", "--fp16"]
