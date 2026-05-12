# Digest pinned 2026-05-12. Tag: nvidia/cuda:12.4.1-devel-ubuntu22.04
# To update: re-resolve via registry API and update digest below.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:da6791294b0b04d7e65d87b7451d6f2390b4d36225ab0701ee7dfec5769829f5

ARG GENESIS_VERSION=0.4.6
ARG LEROBOT_VERSION=0.4.4
ARG RSL_RL_VERSION=2.2.4
ARG TORCH_VERSION=2.6.0
ARG TORCHVISION_VERSION=0.21.0

LABEL npa.tool="genesis" \
      npa.version="${GENESIS_VERSION}"

ENV DEBIAN_FRONTEND=noninteractive \
    GENESIS_VERSION=${GENESIS_VERSION} \
    LEROBOT_VERSION=${LEROBOT_VERSION} \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    PYTHONUNBUFFERED=1 \
    NPA_GENESIS_HOME=/opt/genesis

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      ffmpeg \
      git \
      libegl1 \
      libffi-dev \
      libgl1 \
      libglib2.0-0 \
      libglvnd0 \
      libsm6 \
      libssl-dev \
      libx11-6 \
      libxext6 \
      libxrender1 \
      ninja-build \
      pkg-config \
      python3-dev \
      python3-pip \
      python3-venv \
      wget \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash -u 1000 ubuntu \
    && install -d -m 0755 -o ubuntu -g ubuntu \
      /opt/genesis \
      /opt/genesis/outputs \
      /opt/npa/src/npa \
      /opt/npa/docker/genesis

USER ubuntu
WORKDIR /opt/genesis

RUN python3 -m venv /opt/genesis/venv \
    && /opt/genesis/venv/bin/python -m pip install --upgrade pip setuptools wheel \
    && /opt/genesis/venv/bin/pip install \
      "torch==${TORCH_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" \
      --index-url https://download.pytorch.org/whl/cu124 \
    && /opt/genesis/venv/bin/pip install \
      "genesis-world==${GENESIS_VERSION}" \
      "rsl-rl-lib==${RSL_RL_VERSION}" \
      "tensorboard==2.20.0" \
      "lerobot==${LEROBOT_VERSION}" \
      "typer==0.24.1" \
      "httpx==0.28.1" \
      "paramiko==4.0.0" \
      "pyyaml==6.0.3" \
      "rich==15.0.0" \
      "boto3==1.42.91" \
      "jinja2==3.1.6" \
      "pyarrow==22.0.0"

ENV PATH=/opt/genesis/venv/bin:$PATH \
    PYTHONPATH=/opt/npa/src

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu docker/genesis/smoke_env.py /opt/npa/docker/genesis/smoke_env.py
COPY --chown=ubuntu:ubuntu docker/genesis/smoke_functional.py /opt/npa/docker/genesis/smoke_functional.py

RUN python -m pip install --no-deps -e /opt/npa \
    && python - <<'PY'
import os
from importlib import metadata
import genesis

expected = os.environ["GENESIS_VERSION"]
actual = getattr(genesis, "__version__", metadata.version("genesis-world"))
if actual != expected:
    raise RuntimeError(f"expected genesis-world {expected}, found {actual}")
print(f"GENESIS_BUILD_IMPORT_OK {actual}")
PY

WORKDIR /opt/genesis
ENTRYPOINT ["/bin/bash"]
