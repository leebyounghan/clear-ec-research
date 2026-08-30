FROM --platform=linux/amd64 pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime AS clear_ec_algorithm_amd64
ENV PYTHONUNBUFFERED=1
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
WORKDIR /opt/app
ENV PATH="/home/user/.local/bin:${PATH}"
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*
USER user
COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --user --no-cache-dir --requirement /opt/app/requirements.txt
COPY --chown=user:user inference.py /opt/app/
# model weights baked into the image (a mounted GC Model at /opt/ml/model takes precedence)
COPY --chown=user:user model/ /opt/app/model/
ENTRYPOINT ["python", "inference.py"]
