# Digest pinned 2026-05-12. Tag: nvidia/cuda:12.4.1-devel-ubuntu22.04
# To update: re-resolve via registry API and update digest below.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:da6791294b0b04d7e65d87b7451d6f2390b4d36225ab0701ee7dfec5769829f5

ARG LEROBOT_VERSION=0.5.1

LABEL npa.tool="lerobot" \
      npa.version="${LEROBOT_VERSION}"

ENV DEBIAN_FRONTEND=noninteractive \
    HF_LEROBOT_HOME=/opt/lerobot/hf_cache \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    PYTHONUNBUFFERED=1 \
    NPA_SERVER_HOST=0.0.0.0 \
    NPA_SERVER_PORT=8080 \
    NPA_CHECKPOINT_DIR=/opt/lerobot/checkpoints \
    NPA_JOB_STATUS_DIR=/opt/lerobot/job_status \
    NPA_LOG_DIR=/var/log/npa-lerobot \
    NPA_LEROBOT_VERSION=${LEROBOT_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      gnupg \
      software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      ffmpeg \
      git \
      libegl1 \
      libegl1-mesa-dev \
      libffi-dev \
      libgl1 \
      libgl1-mesa-dev \
      libglib2.0-0 \
      libglvnd0 \
      libsm6 \
      libssl-dev \
      libx11-dev \
      libxext-dev \
      libxext6 \
      libxrender-dev \
      ninja-build \
      pkg-config \
      python3.12 \
      python3.12-dev \
      python3.12-venv \
      python3-pip \
      wget \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash -u 1000 ubuntu \
    && install -d -m 0755 -o ubuntu -g ubuntu \
      /opt/lerobot \
      /opt/lerobot/checkpoints \
      /opt/lerobot/data \
      /opt/lerobot/dataset_cache \
      /opt/lerobot/checkpoint_cache \
      /opt/lerobot/hf_cache \
      /opt/lerobot/job_status \
      /opt/lerobot/logs \
      /opt/lerobot/benchmarks \
      /opt/npa/src/npa \
      /var/log/npa-lerobot \
    && touch /opt/lerobot/.env \
    && chown ubuntu:ubuntu /opt/lerobot/.env \
    && chmod 600 /opt/lerobot/.env

USER ubuntu
WORKDIR /opt/lerobot

# 0.5.1 keeps the historic [pusht,libero] extra bundle + pinned torch upgrades.
# 0.6.0 uses lean feature extras and must NOT force torch>=2.12 (upstream ceiling).
RUN python3.12 -m venv /opt/lerobot/venv \
    && /opt/lerobot/venv/bin/python -m pip install --upgrade pip setuptools wheel \
    && if [ "${LEROBOT_VERSION}" = "0.6.0" ]; then \
         /opt/lerobot/venv/bin/pip install \
           "lerobot[training,evaluation,pusht,libero]==${LEROBOT_VERSION}" \
           boto3 \
           fastapi~=0.136.1 \
           num2words \
           tensorboard \
           "uvicorn[standard]"; \
       else \
         /opt/lerobot/venv/bin/pip install \
           "lerobot[pusht,libero]==${LEROBOT_VERSION}" \
           boto3 \
           fastapi~=0.136.1 \
           num2words \
           tensorboard \
           "uvicorn[standard]" \
         && /opt/lerobot/venv/bin/pip install --no-cache-dir --upgrade \
           "torch==2.12.1" \
           "torchvision==0.27.1" \
           "diffusers>=0.38.0"; \
       fi \
    && /opt/lerobot/venv/bin/pip uninstall -y wandb

ENV PATH=/opt/lerobot/venv/bin:$PATH \
    PYTHONPATH=/opt/npa/src \
    NPA_SKIP_EAGER_IMPORTS=1

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa/__init__.py /opt/npa/src/npa/__init__.py
COPY --chown=ubuntu:ubuntu src/npa/server /opt/npa/src/npa/server
COPY --chown=ubuntu:ubuntu src/npa/smoke /opt/npa/src/npa/smoke

RUN python -m npa.smoke.test_lerobot_env

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends --only-upgrade openssl libssl3 \
    && apt-get purge -y --auto-remove \
      dirmngr \
      gnupg \
      gnupg-utils \
      gnupg2 \
      gpg \
      gpg-agent \
      nodejs \
      software-properties-common \
    && dpkg --purge --force-depends linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*

USER ubuntu

EXPOSE 8080

ENTRYPOINT ["python", "-m", "npa.server.app"]
