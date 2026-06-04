# Cosmos 3 / Transfer 2.5 runner image for raw SkyPilot workflows.
#
# This image intentionally does not bake model weights or credentials. Runtime
# jobs fetch gated Hugging Face weights with a license-accepted HF token.
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ARG COSMOS_IMAGE_VERSION=3.0.0
ARG COSMOS3_SOURCE_REPO=https://github.com/NVIDIA/cosmos-framework.git
ARG COSMOS_TRANSFER_SOURCE_REPO=https://github.com/nvidia-cosmos/cosmos-transfer2.5.git

LABEL npa.tool="cosmos" \
      npa.version="${COSMOS_IMAGE_VERSION}" \
      npa.cosmos3.source="${COSMOS3_SOURCE_REPO}" \
      npa.transfer25.source="${COSMOS_TRANSFER_SOURCE_REPO}"

ENV DEBIAN_FRONTEND=noninteractive \
    COSMOS_HOME=/opt/cosmos \
    COSMOS3_HOME=/opt/cosmos/cosmos-framework \
    COSMOS_TRANSFER25_HOME=/opt/cosmos/cosmos-transfer2.5 \
    COSMOS_DATA_HOME=/opt/cosmos-data \
    COSMOS_MODEL_DIR=/opt/cosmos-data/models \
    COSMOS_HF_CACHE=/opt/cosmos-data/hf_cache \
    COSMOS_OUTPUT_DIR=/opt/cosmos-data/outputs \
    COSMOS_DISABLE_SAFETY=0 \
    HF_HOME=/opt/cosmos-data/hf_cache \
    HUGGINGFACE_HUB_CACHE=/opt/cosmos-data/hf_cache \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      ffmpeg \
      git \
      git-lfs \
      libegl1 \
      libgl1 \
      libglib2.0-0 \
      libglvnd0 \
      libsm6 \
      libx11-6 \
      libx11-dev \
      libxext6 \
      libxrender1 \
      python3 \
      python3-dev \
      python3-pip \
      python3-venv \
      tree \
      wget \
    && rm -rf /var/lib/apt/lists/*

RUN id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu \
    && install -d -m 0755 -o ubuntu -g ubuntu \
      /opt/cosmos \
      /opt/cosmos-data/models \
      /opt/cosmos-data/hf_cache \
      /opt/cosmos-data/outputs \
      /opt/npa/src/npa \
      /opt/npa/docker/workbench/cosmos

USER ubuntu
WORKDIR /opt/cosmos

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mkdir -p /opt/cosmos/bin \
    && ln -sf "${HOME}/.local/bin/uv" /opt/cosmos/bin/uv \
    && ln -sf "${HOME}/.local/bin/uvx" /opt/cosmos/bin/uvx

ENV PATH=/opt/cosmos/bin:/opt/cosmos/venv/bin:$PATH \
    PYTHONPATH=/opt/npa/src

RUN git clone --depth 1 "${COSMOS3_SOURCE_REPO}" "${COSMOS3_HOME}" \
    && git clone --depth 1 "${COSMOS_TRANSFER_SOURCE_REPO}" "${COSMOS_TRANSFER25_HOME}"

RUN python3 -m venv /opt/cosmos/awscli-venv \
    && /opt/cosmos/awscli-venv/bin/python -m pip install --upgrade pip setuptools wheel \
    && /opt/cosmos/awscli-venv/bin/python -m pip install "awscli>=1.32" \
    && ln -sf /opt/cosmos/awscli-venv/bin/aws /opt/cosmos/bin/aws

RUN python3 -m venv /opt/cosmos/venv \
    && /opt/cosmos/venv/bin/python -m pip install --upgrade pip setuptools wheel \
    && /opt/cosmos/venv/bin/python -m pip install \
      "boto3>=1.42" \
      "huggingface_hub[cli]>=0.23" \
      "numpy>=1.26" \
      "pillow>=10.0" \
      "pyyaml>=6.0" \
      "rich>=15.0" \
      "typer>=0.24" \
      "httpx>=0.28"

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu workflows /opt/npa/workflows
COPY --chown=ubuntu:ubuntu docker/workbench/cosmos /opt/npa/docker/workbench/cosmos

RUN python -m pip install -e /opt/npa \
    && python -m py_compile \
      /opt/npa/src/npa/workbench/cosmos/cosmos3.py \
      /opt/npa/src/npa/workbench/cosmos/workflows.py \
      /opt/npa/docker/workbench/cosmos/smoke_env.py \
      /opt/npa/docker/workbench/cosmos/smoke_functional.py

WORKDIR /opt/cosmos
ENTRYPOINT ["/bin/bash"]
