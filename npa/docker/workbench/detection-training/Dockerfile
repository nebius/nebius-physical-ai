# Reuse the accepted public detector's CUDA 13 / torch 2.12.1 runtime and
# replace its application source below. Pinning the published foundation makes
# clean builds independent of a builder-local shared CUDA base tag.
# NPA_CUDA_BASE remains available for an explicitly selected compatible base.
ARG NPA_CUDA_BASE=ghcr.io/nebius/nebius-physical-ai/npa-detection-training@sha256:a432dd2a9e6bafed4b78466ac806f584c7fcf549cf3be0edaf75f2fecdadfdd9
FROM ${NPA_CUDA_BASE}

# The base drops to a non-root user; build steps need root, and the image drops
# back to the ubuntu user before runtime (below).
USER root

# Refresh vulnerable userspace headers inherited from the accepted runtime;
# the CUDA, torch and torchvision closure remains the same.
ARG LINUX_LIBC_DEV_VERSION=6.8.0-139.139
RUN apt-get update \
    && apt-get install -y --no-install-recommends linux-libc-dev=${LINUX_LIBC_DEV_VERSION} \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    DETECTION_TRAINING_PORT=8790

WORKDIR /app

# libgl/libglib + the FastAPI + lancedb + pyarrow + pillow + boto3 + torch/CUDA
# stack come from the shared base; only the detector-specific torch extensions
# and COCO metrics are unique to this image.
COPY docker/workbench/detection-training/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN mkdir -p /app/npa/workbench
RUN touch /app/npa/__init__.py /app/npa/workbench/__init__.py
COPY src/npa/workbench/detection_training /app/npa/workbench/detection_training
# detection_training.schemas imports training_config at module load.
COPY src/npa/workbench/training_config.py /app/npa/workbench/training_config.py
COPY src/npa/workbench/storage_scope.py /app/npa/workbench/storage_scope.py
# Golden-eval smoke (standalone; runs against the in-image service module).
COPY src/npa/smoke/test_detection_training_functional.py /app/smoke_functional.py

RUN id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu \
    && chown -R ubuntu:ubuntu /app

EXPOSE 8790

USER ubuntu

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import os,urllib.request; p=os.environ.get('DETECTION_TRAINING_PORT','8790'); urllib.request.urlopen(f'http://127.0.0.1:{p}/readyz', timeout=3)" || exit 1

CMD ["uvicorn", "npa.workbench.detection_training.service:app", "--host", "0.0.0.0", "--port", "8790"]
