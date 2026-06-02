# Digest pinned 2026-06-02. Tag: python:3.11-slim-trixie
# To update: re-resolve via registry API and update digest below.
FROM python:3.11-slim-trixie@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0

ARG FIFTYONE_VERSION=1.15.0
ARG BOTO3_VERSION=1.43.6
ARG DATASETS_VERSION=4.8.5
ARG HUGGINGFACE_HUB_VERSION=1.14.0
ARG PYARROW_VERSION=24.0.0
ARG PILLOW_VERSION=12.2.0

LABEL npa.tool="fiftyone" \
      npa.version="${FIFTYONE_VERSION}"

ENV DEBIAN_FRONTEND=noninteractive \
    FIFTYONE_VERSION=${FIFTYONE_VERSION} \
    FIFTYONE_HOME=/opt/fiftyone \
    FIFTYONE_DATABASE_DIR=/opt/fiftyone/db \
    FIFTYONE_DEFAULT_APP_ADDRESS=0.0.0.0 \
    FIFTYONE_DEFAULT_APP_PORT=5151 \
    FIFTYONE_DO_NOT_TRACK=true \
    FIFTYONE_VENV=/opt/fiftyone/venv \
    PATH=/opt/fiftyone/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      ffmpeg \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      libsm6 \
      libxext6 \
      libxrender1 \
      procps \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash -u 1000 ubuntu \
    && install -d -m 0755 -o ubuntu -g ubuntu \
      /opt/fiftyone \
      /opt/fiftyone/db \
      /opt/fiftyone/datasets \
      /opt/fiftyone/smoke \
      /opt/fiftyone/venv \
      /opt/npa/docker/workbench/fiftyone \
    && printf '%s\n' \
      'export FIFTYONE_VENV=/opt/fiftyone/venv' \
      'export PATH=/opt/fiftyone/venv/bin:$PATH' \
      > /etc/profile.d/npa-fiftyone.sh

USER ubuntu
WORKDIR /opt/fiftyone

RUN python -m venv "${FIFTYONE_VENV}" \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
      "fiftyone==${FIFTYONE_VERSION}" \
      "boto3==${BOTO3_VERSION}" \
      "datasets==${DATASETS_VERSION}" \
      "huggingface_hub==${HUGGINGFACE_HUB_VERSION}" \
      "pyarrow==${PYARROW_VERSION}" \
      "pillow==${PILLOW_VERSION}"

COPY --chown=ubuntu:ubuntu docker/workbench/fiftyone/smoke_env.py /opt/npa/docker/workbench/fiftyone/smoke_env.py
COPY --chown=ubuntu:ubuntu docker/workbench/fiftyone/smoke_functional.py /opt/npa/docker/workbench/fiftyone/smoke_functional.py

RUN python -m py_compile /opt/npa/docker/workbench/fiftyone/smoke_env.py /opt/npa/docker/workbench/fiftyone/smoke_functional.py \
    && python - <<'PY'
from importlib import metadata
import os

expected = os.environ["FIFTYONE_VERSION"]
actual = metadata.version("fiftyone")
if actual != expected:
    raise RuntimeError(f"expected fiftyone {expected}, found {actual}")
print(f"FIFTYONE_BUILD_VERSION_OK {actual}")
PY

EXPOSE 5151
ENTRYPOINT ["/bin/bash"]
