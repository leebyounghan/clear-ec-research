# Same PyTorch release as training; CUDA 12.6 instead of CUDA 13 for broader GPUs.
# Model weights are uploaded separately and mounted read-only at /opt/ml/model.
FROM --platform=linux/amd64 pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime@sha256:3bb77138e105723dd4ed760b82fb63d8310ae3a1afc58f76e0ecf0f776568d33
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MALLOC_ARENA_MAX=2 \
    HOME=/tmp XDG_CACHE_HOME=/tmp/.cache TORCH_HOME=/tmp/torch \
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
WORKDIR /opt/app
COPY app/requirements.txt /opt/app/requirements.txt
COPY constraints.txt /opt/app/constraints.txt
# The official base uses Ubuntu's externally managed Python. This change is
# confined to this disposable image, never the host Python installation.
RUN python -m pip install --break-system-packages --no-cache-dir -c constraints.txt -r requirements.txt \
    && python -m pip check \
    && python -c "import torch, cellpose.models, SimpleITK, cv2; assert torch.__version__.split('+')[0] == '2.11.0'; assert torch.version.cuda == '12.6'" \
    && python -m pip freeze > /opt/app/runtime-lock.txt
RUN groupadd --gid 10001 algorithm \
    && useradd --uid 10001 --gid algorithm --no-create-home algorithm \
    && mkdir -p /opt/ml/model /input /output \
    && chown algorithm:algorithm /output
COPY app/*.py /opt/app/
USER 10001:10001
ENTRYPOINT ["python", "/opt/app/inference.py", "--config", "candidate6_full_quality.json", "--fp16"]
