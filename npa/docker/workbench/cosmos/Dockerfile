# Digest pinned 2026-08-03. Tag: nvidia/cuda:12.8.1-devel-ubuntu22.04
# To update: re-resolve via registry API and update digest below.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97

ARG COSMOS_VERSION=1.0.9
ARG COSMOS_TORCH_VERSION=2.7.0
ARG COSMOS_TORCHVISION_VERSION=0.22.0
ARG COSMOS_TRITON_VERSION=3.3.0
ARG COSMOS_FLASH_ATTN_VERSION=2.7.3
ARG COSMOS_NATTEN_VERSION=0.21.0
ARG COSMOS_TRANSFORMER_ENGINE_VERSION=1.13.0

LABEL npa.tool="cosmos" \
      npa.version="${COSMOS_VERSION}"

ENV DEBIAN_FRONTEND=noninteractive \
    COSMOS_HOME=/opt/cosmos \
    COSMOS_DATA_HOME=/opt/cosmos-data \
    COSMOS_MODEL_DIR=/opt/cosmos-data/models \
    COSMOS_HF_CACHE=/opt/cosmos-data/hf_cache \
    COSMOS_OUTPUT_DIR=/opt/cosmos-data/outputs \
    COSMOS_MODEL_ID=nvidia/Cosmos-1.0-Diffusion-7B-Text2World \
    COSMOS_DISABLE_SAFETY=0 \
    COSMOS_VERSION=${COSMOS_VERSION} \
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
      libegl1 \
      libgl1 \
      libglib2.0-0 \
      libglvnd0 \
      libsm6 \
      libx11-6 \
      libxext6 \
      libxrender1 \
      python3-dev \
      python3-pip \
      python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash -u 1000 ubuntu \
    && install -d -m 0755 -o ubuntu -g ubuntu \
      /opt/cosmos \
      /opt/cosmos-data/models \
      /opt/cosmos-data/hf_cache \
      /opt/cosmos-data/outputs \
      /opt/npa/src/npa \
      /opt/npa/docker/workbench/cosmos

USER ubuntu
WORKDIR /opt/cosmos

RUN python3 -m venv /opt/cosmos/venv \
    && /opt/cosmos/venv/bin/python -m pip install --upgrade pip setuptools wheel \
    && /opt/cosmos/venv/bin/pip install \
      "torch==${COSMOS_TORCH_VERSION}" \
      "torchvision==${COSMOS_TORCHVISION_VERSION}" \
      --extra-index-url https://nvidia-cosmos.github.io/cosmos-dependencies/v1.2.0/cu128_torch27/simple \
    # Predict2 1.0.9's legacy metadata pins triton 3.2 for torch 2.6. Installing
    # it normally makes pip backtrack off the explicitly selected torch
    # 2.7/cu128 wheel. Install the package itself without dependencies, then
    # install its declared non-CUDA closure while excluding the six artifacts
    # whose versions are fixed by this cu128 port.
    && /opt/cosmos/venv/bin/pip install --no-deps \
      "cosmos-predict2==${COSMOS_VERSION}" \
    && /opt/cosmos/venv/bin/python - <<'PY' > /tmp/cosmos-predict2-noncuda-requirements.txt
from importlib import metadata
from packaging.requirements import Requirement

excluded = {
    "flash-attn",
    "natten",
    "torch",
    "torchvision",
    "transformer-engine",
    "triton",
}
for raw in metadata.requires("cosmos-predict2") or ():
    requirement = Requirement(raw)
    normalized = requirement.name.lower().replace("_", "-")
    if normalized not in excluded and (requirement.marker is None or requirement.marker.evaluate()):
        print(raw)
PY
RUN printf '%s\n' \
      "torch==${COSMOS_TORCH_VERSION}" \
      "torchvision==${COSMOS_TORCHVISION_VERSION}" \
      "triton==${COSMOS_TRITON_VERSION}" \
      > /tmp/cosmos-cu128-constraints.txt \
    && /opt/cosmos/venv/bin/pip install \
      -r /tmp/cosmos-predict2-noncuda-requirements.txt \
      -c /tmp/cosmos-cu128-constraints.txt \
    && curl -L -o "/tmp/flash_attn-${COSMOS_FLASH_ATTN_VERSION}-cp310-cp310-linux_x86_64.whl" \
      "https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.2.0/flash_attn-2.7.3%2Bcu128.torch27-cp310-cp310-linux_x86_64.whl" \
    && /opt/cosmos/venv/bin/pip install --no-deps "/tmp/flash_attn-${COSMOS_FLASH_ATTN_VERSION}-cp310-cp310-linux_x86_64.whl" \
    && curl -L -o "/tmp/natten-${COSMOS_NATTEN_VERSION}-cp310-cp310-linux_x86_64.whl" \
      "https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.2.0/natten-0.21.0%2Bcu128.torch27-cp310-cp310-linux_x86_64.whl" \
    && /opt/cosmos/venv/bin/pip install --no-deps "/tmp/natten-${COSMOS_NATTEN_VERSION}-cp310-cp310-linux_x86_64.whl" \
    && curl -L -o "/tmp/transformer_engine-${COSMOS_TRANSFORMER_ENGINE_VERSION}-cp310-cp310-linux_x86_64.whl" \
      "https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.2.0/transformer_engine-1.13.0%2Bcu128.torch27-cp310-cp310-linux_x86_64.whl" \
    && /opt/cosmos/venv/bin/pip install --no-deps "/tmp/transformer_engine-${COSMOS_TRANSFORMER_ENGINE_VERSION}-cp310-cp310-linux_x86_64.whl" \
    && /opt/cosmos/venv/bin/pip install \
      -c /tmp/cosmos-cu128-constraints.txt \
      "diffusers==0.38.0" \
      "peft==0.19.1" \
      "transformers==4.51.3" \
      "accelerate==1.13.0" \
      "fastapi==0.136.1" \
      "uvicorn[standard]==0.46.0" \
      "huggingface_hub==0.36.2" \
      "pillow==11.1.0" \
      "imageio[ffmpeg]==2.37.0" \
      "pydantic==2.13.4" \
      "python-multipart==0.0.27" \
      "typer==0.24.1" \
      "httpx==0.28.1" \
      "paramiko==5.0.0" \
      "pyyaml==6.0.3" \
      "rich==15.0.0" \
      "boto3==1.42.91" \
      "jinja2==3.1.6" \
    && /opt/cosmos/venv/bin/pip install --no-deps "cosmos_guardrail==0.1.0" \
    && /opt/cosmos/venv/bin/pip install --no-cache-dir --no-deps --upgrade \
      "modelscope>=1.27.0" \
      "nltk>=3.9.4" \
      "pillow>=12.2.0" \
      "protobuf==5.29.6" \
      "python-multipart>=0.0.30" \
      "sentencepiece>=0.2.1" \
      "transformers>=5.3.0" \
    && /opt/cosmos/venv/bin/pip uninstall -y wandb \
    && rm -f /tmp/flash_attn-*.whl /tmp/natten-*.whl /tmp/transformer_engine-*.whl \
      /tmp/cosmos-predict2-noncuda-requirements.txt /tmp/cosmos-cu128-constraints.txt

ENV PATH=/opt/cosmos/venv/bin:$PATH \
    PYTHONPATH=/opt/npa/src \
    NPA_SKIP_EAGER_IMPORTS=1

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu workflows /opt/npa/workflows
COPY --chown=ubuntu:ubuntu docker/workbench/cosmos/build_server.py /opt/npa/docker/workbench/cosmos/build_server.py
COPY --chown=ubuntu:ubuntu docker/workbench/cosmos/smoke_env.py /opt/npa/docker/workbench/cosmos/smoke_env.py
COPY --chown=ubuntu:ubuntu docker/workbench/cosmos/smoke_functional.py /opt/npa/docker/workbench/cosmos/smoke_functional.py

RUN python -m pip install --no-deps -e /opt/npa \
    && python /opt/npa/docker/workbench/cosmos/build_server.py

RUN python -m py_compile /opt/cosmos/server.py /opt/npa/docker/workbench/cosmos/build_server.py /opt/npa/docker/workbench/cosmos/smoke_env.py /opt/npa/docker/workbench/cosmos/smoke_functional.py

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends --only-upgrade gpgv openssl libssl3 \
    && apt-get purge -y --auto-remove \
      dirmngr \
      gnupg \
      gnupg-utils \
      gnupg2 \
      gpg \
      gpg-agent \
      software-properties-common \
    && dpkg --purge --force-depends linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*

USER ubuntu

WORKDIR /opt/cosmos
EXPOSE 8080
ENTRYPOINT ["/bin/bash"]
