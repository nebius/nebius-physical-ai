# syntax=docker/dockerfile:1.7

# Redistributable Cosmos Transfer 2.5 capability image.
#
# This build intentionally starts from NVIDIA's public CUDA container, not from a
# privately assembled npa-cosmos2-transfer parent. The source and every Python
# dependency are immutable inputs: the source checkout is a full commit SHA and
# `uv sync --locked` consumes the upstream hash-bearing lockfile. No credential is
# accepted or needed during the build. Gated model weights are fetched only at run
# time into HF_HOME after the operator supplies HF_TOKEN and accepts the NVIDIA Open
# Model License.
ARG CUDA_BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04@sha256:3986465b3dd3b4d602c07061f2cff417e0bfb24810129408d4eb12e111015a6c
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.8.12@sha256:e1d7fa999ae39871fc2fd6c9c572d970cc877a239d2a16238a204c23772d1c0e
ARG UBUNTU_SNAPSHOT=20260801T053000Z

FROM ${UV_IMAGE} AS uv-bin

FROM ${CUDA_BASE_IMAGE} AS build

ARG COSMOS_TRANSFER_REVISION=67d56b7d550a3911024a32dc23ae0bae5258e633
ARG COSMOS_PYTHON_VERSION=3.10.18
ARG UBUNTU_SNAPSHOT

ENV DEBIAN_FRONTEND=noninteractive \
    GIT_LFS_SKIP_SMUDGE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/cosmos/uv-python

COPY --from=uv-bin /uv /usr/local/bin/uv

RUN rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*.list \
      /etc/apt/sources.list.d/*.sources \
    && printf '%s\n' \
      'Types: deb' \
      "URIs: https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/" \
      'Suites: noble noble-updates noble-backports noble-security' \
      'Components: main restricted universe multiverse' \
      'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg' \
      > /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/cosmos/cosmos-transfer2.5

# Both fetch and checkout explicitly disable LFS smudging. Fetching the immutable
# SHA directly avoids depending on a moving branch or tag. The entire Git database,
# all upstream assets (including LFS pointers), examples other than the real
# inference entrypoint, docs, tests, and caches are removed before this RUN is
# committed, so they never appear in an image layer.
RUN set -eux; \
    git init; \
    git remote add origin https://github.com/nvidia-cosmos/cosmos-transfer2.5.git; \
    GIT_LFS_SKIP_SMUDGE=1 git -c filter.lfs.smudge= -c filter.lfs.required=false \
      fetch --depth=1 --filter=blob:none origin "${COSMOS_TRANSFER_REVISION}"; \
    GIT_LFS_SKIP_SMUDGE=1 git -c filter.lfs.smudge= -c filter.lfs.required=false \
      checkout --detach FETCH_HEAD; \
    test "$(git rev-parse HEAD)" = "${COSMOS_TRANSFER_REVISION}"; \
    grep -q 'filter=lfs' .gitattributes; \
    rm -rf .git assets .github docker docs scripts tests; \
    find examples -mindepth 1 -maxdepth 1 ! -name inference.py -exec rm -rf -- {} +; \
    test -f examples/inference.py; \
    test ! -e assets; \
    test ! -e .git

COPY --chmod=755 docker/workbench/cosmos2-transfer/assert_no_forbidden_payload.sh \
    /usr/local/bin/assert-no-forbidden-cosmos2-payload
COPY docker/workbench/cosmos2-transfer/security-overrides.txt \
    /tmp/cosmos2-security-overrides.txt

# Python itself and the complete cu128 inference environment are exact-version or
# lockfile pinned. `--no-editable` makes the venv independent of removed source-only
# material while retaining the Apache source tree used for provenance and debugging.
RUN set -eux; \
    uv python install "${COSMOS_PYTHON_VERSION}"; \
    printf '%s\n' "${COSMOS_PYTHON_VERSION}" > .python-version; \
    uv sync --locked --no-dev --no-editable --extra=cu128 \
      --python "${COSMOS_PYTHON_VERSION}"; \
    uv pip install --python .venv/bin/python --no-deps \
      --requirement /tmp/cosmos2-security-overrides.txt; \
    .venv/bin/python -c 'import defusedxml, importlib.metadata, msgpack, nltk, numpy; assert importlib.metadata.version("defusedxml") == "0.7.1"; assert importlib.metadata.version("nltk") == "3.10.0"; assert importlib.metadata.version("pip") == "26.2"; assert importlib.metadata.version("msgpack") == "1.2.1"; assert importlib.metadata.version("setuptools") == "83.0.0"; assert tuple(map(int, numpy.__version__.split(".")[:1])) >= (2,)'; \
    .venv/bin/python -m pip --version; \
    rm -rf .venv/lib/python3.10/site-packages/matplotlib/mpl-data/sample_data; \
    rm -rf .venv/lib/python3.10/site-packages/skimage/data \
      .venv/lib/python3.10/site-packages/wandb/bin; \
    cp .venv/lib/python3.10/site-packages/numpy/_core/tests/_natype.py \
      /tmp/numpy-_natype.py; \
    find .venv/lib/python3.10/site-packages/numpy \
      .venv/lib/python3.10/site-packages/scipy \
      -type d -name tests -prune -exec rm -rf -- {} +; \
    install -D -m 0644 /tmp/numpy-_natype.py \
      .venv/lib/python3.10/site-packages/numpy/_core/tests/_natype.py; \
    rm -f /tmp/numpy-_natype.py; \
    test ! -e .venv/lib/python3.10/site-packages/matplotlib/mpl-data/sample_data; \
    test ! -e .venv/lib/python3.10/site-packages/skimage/data; \
    test ! -e .venv/lib/python3.10/site-packages/wandb/bin; \
    test "$(readlink -f .venv/bin/python)" = \
      "$(find "${UV_PYTHON_INSTALL_DIR}" -path '*/bin/python3.10' -type f -print -quit)"; \
    ! grep -q '/root' .venv/pyvenv.cfg; \
    .venv/bin/python -c 'import importlib.metadata, flash_attn, scipy.linalg, torch, wandb; from transformers import SiglipModel; assert importlib.metadata.version("nltk") == "3.10.0"; assert importlib.metadata.version("defusedxml") == "0.7.1"; assert wandb.run is None; print(torch.__version__)'; \
    base_site="$(find "${UV_PYTHON_INSTALL_DIR}" -path '*/lib/python3.10/site-packages' -type d -print -quit)"; \
    base_python="$(find "${UV_PYTHON_INSTALL_DIR}" -path '*/bin/python3.10' -type f -print -quit)"; \
    test -n "${base_site}"; \
    test -n "${base_python}"; \
    rm -rf "${base_site}"/pip "${base_site}"/pip-*.dist-info \
      "${base_site}"/setuptools "${base_site}"/setuptools-*.dist-info \
      "${base_site}"/_distutils_hack "${base_site}"/distutils-precedence.pth; \
    "${base_python}" -c 'import importlib.util; assert importlib.util.find_spec("pip") is None; assert importlib.util.find_spec("setuptools") is None'; \
    assert-no-forbidden-cosmos2-payload /opt/cosmos/cosmos-transfer2.5

FROM ${CUDA_RUNTIME_IMAGE} AS runtime

ARG COSMOS_TRANSFER_REVISION=67d56b7d550a3911024a32dc23ae0bae5258e633
ARG UBUNTU_SNAPSHOT
ARG NPA_SOURCE_SHA

LABEL org.opencontainers.image.title="NPA Cosmos Transfer 2.5" \
      org.opencontainers.image.description="Cosmos Transfer 2.5 source runtime; model weights are operator-fetched at run time" \
      org.opencontainers.image.source="https://github.com/nvidia-cosmos/cosmos-transfer2.5" \
      org.opencontainers.image.revision="${COSMOS_TRANSFER_REVISION}" \
      org.opencontainers.image.licenses="Apache-2.0 AND LicenseRef-NVIDIA-Deep-Learning-Container" \
      npa.source_revision="${NPA_SOURCE_SHA}" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

ENV DEBIAN_FRONTEND=noninteractive \
    NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    HF_HOME=/opt/cosmos/model-cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/opt/cosmos/model-cache/huggingface/hub \
    NLTK_DATA=/opt/cosmos/model-cache \
    TORCH_HOME=/opt/cosmos/model-cache/torch \
    XDG_CACHE_HOME=/opt/cosmos/model-cache/xdg \
    TOKENIZERS_PARALLELISM=false \
    UV_PYTHON_INSTALL_DIR=/opt/cosmos/uv-python

# Upstream deliberately invokes `uvx hf ...` to download gated model files at
# run time. Ship the two binaries from the same digest-pinned, dual-licensed uv
# image used by the build; no package or model download occurs here.
COPY --from=uv-bin /uv /usr/local/bin/uv
COPY --from=uv-bin /uvx /usr/local/bin/uvx

# ffmpeg creates and validates the repository-authored procedural fixture. SSH,
# rsync, sudo, procps, and netcat are the runtime prerequisites of SkyPilot's
# Kubernetes bootstrap for a non-root container.
RUN rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*.list \
      /etc/apt/sources.list.d/*.sources \
    && printf '%s\n' \
      'Types: deb' \
      "URIs: https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/" \
      'Suites: noble noble-updates noble-backports noble-security' \
      'Components: main restricted universe multiverse' \
      'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg' \
      > /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
      ca-certificates ffmpeg netcat-openbsd openssh-server procps rsync sudo \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/ssh/ssh_host_* \
    && test -z "$(find /etc/ssh -maxdepth 1 -type f -name 'ssh_host_*' -print -quit)" \
    && mkdir -p /run/sshd

# Transformer Engine loads the unversioned CUDA runtime name directly. NVIDIA's
# runtime image ships the versioned redistributable library but, unlike the devel
# image, not its unversioned linker symlink. Supply that compatibility name from
# the same digest-pinned runtime bits and keep the containing directory explicit.
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-12.8/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
RUN set -eux; \
    cudart_dir=/usr/local/cuda-12.8/targets/x86_64-linux/lib; \
    test -e "${cudart_dir}/libcudart.so.12"; \
    ln -sfn libcudart.so.12 "${cudart_dir}/libcudart.so"; \
    ldconfig

COPY --from=build --chown=1000:1000 /opt/cosmos /opt/cosmos
COPY --chown=1000:1000 pyproject.toml /opt/npa/pyproject.toml
COPY --chown=1000:1000 src/npa /opt/npa/src/npa
COPY --chown=1000:1000 workflows /opt/npa/workflows
COPY docker/workbench/cosmos2-transfer/npa-cli-requirements.txt \
    /tmp/npa-cli-requirements.txt
COPY --from=build /usr/local/bin/assert-no-forbidden-cosmos2-payload \
    /usr/local/bin/assert-no-forbidden-cosmos2-payload
COPY --chmod=755 docker/workbench/cosmos2-transfer/smoke_functional.sh \
    /opt/cosmos2-transfer/smoke_functional.sh
COPY --chmod=755 docker/workbench/cosmos2-transfer/entrypoint.sh \
    /opt/cosmos2-transfer/entrypoint.sh
COPY --chmod=755 src/npa/workbench/cosmos/fixture.py \
    /opt/cosmos2-transfer/generate_fixture.py
COPY docker/workbench/cosmos2-transfer/REDISTRIBUTION.md \
    /usr/share/doc/npa-cosmos2-transfer/REDISTRIBUTION.md

# This purpose-built component image runs the exact NPA source tree without
# invoking a PEP 517 backend.  The source is already immutable via
# NPA_SOURCE_SHA; avoiding a package build also removes any implicit Hatchling
# download or mutable build-isolation environment. Keep package initialization
# lazy so importing the attestation and Cosmos component modules does not
# require CLI/viewer extras.
ENV NPA_SKIP_EAGER_IMPORTS=1 \
    PYTHONPATH=/opt/npa/src \
    NPA_BAKED_PYTHON=/opt/cosmos/cosmos-transfer2.5/.venv/bin/python

# Deliberately shadow the global python3: SkyPilot and orchestrator-supplied
# commands invoke python3; /usr/bin/python3 remains available to system tooling.
RUN set -eux; \
    test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40; \
    id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu; \
    ln -sfn /opt/npa/src/npa \
      /opt/cosmos/cosmos-transfer2.5/.venv/lib/python3.10/site-packages/npa; \
    printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu; \
    chmod 0440 /etc/sudoers.d/ubuntu; \
    printf '%s\n' '#!/bin/sh' \
      'exec /opt/cosmos/cosmos-transfer2.5/.venv/bin/python "$@"' \
      > /usr/local/bin/python3; \
    chmod 0755 /usr/local/bin/python3; \
    ln -sfn python3 /usr/local/bin/python; \
    mkdir -p /opt/cosmos2-transfer \
      /opt/cosmos/model-cache/huggingface \
      /opt/cosmos/model-cache/torch \
      /opt/cosmos/model-cache/xdg \
      /opt/cosmos-data/hf_cache; \
    chown -R ubuntu:ubuntu /opt/cosmos/model-cache /opt/cosmos2-transfer /opt/cosmos-data; \
    assert-no-forbidden-cosmos2-payload /opt/cosmos/cosmos-transfer2.5; \
    su ubuntu -s /bin/bash -c \
      '/opt/cosmos/cosmos-transfer2.5/.venv/bin/python -c "import sys, torch, flash_attn; from distutils.sysconfig import get_config_var; assert \"/root\" not in (sys._home or \"\"); get_config_var(\"prefix\")"'; \
    su ubuntu -s /bin/bash -c \
      '/opt/cosmos/cosmos-transfer2.5/.venv/bin/python -c "import os; from pathlib import Path; from nltk.pathsec import _get_allowed_roots; assert Path(os.environ[\"NLTK_DATA\"]).resolve() in _get_allowed_roots()"'; \
    su ubuntu -s /bin/bash -c \
      '/opt/cosmos/cosmos-transfer2.5/.venv/bin/python -c "import ctypes; ctypes.CDLL(\"libcudart.so\")"'; \
    su ubuntu -s /bin/bash -c \
      'test "$(command -v python3)" = /usr/local/bin/python3 && python3 -c "import cosmos_transfer2" && python3 -m pip --version'; \
    su ubuntu -s /bin/bash -c \
      'test "$(command -v uvx)" = /usr/local/bin/uvx && uvx --version'; \
    su ubuntu -s /bin/bash -c \
      'test -w /opt/cosmos/model-cache/huggingface && test -x /opt/cosmos/cosmos-transfer2.5/.venv/bin/python'

# The full top-level NPA command tree is imported by the console script even for
# `workbench cosmos2`.  Install its missing import-time packages as an exact,
# hash-verified overlay without resolving or upgrading the vendor lock.  Merely
# checking `import npa` is insufficient: that package is intentionally lazy and
# was green in the image that failed live with a missing Typer dependency.  The
# console wrapper is authored locally from fixed text and executes the copied
# source directly; no build backend or package index is consulted.
RUN set -eux; \
    uv pip install --python /opt/cosmos/cosmos-transfer2.5/.venv/bin/python \
      --no-deps --require-hashes \
      --requirement /tmp/npa-cli-requirements.txt; \
    printf '%s\n' '#!/bin/sh' \
      'exec /opt/cosmos/cosmos-transfer2.5/.venv/bin/python -m npa.cli.main "$@"' \
      > /opt/cosmos/cosmos-transfer2.5/.venv/bin/npa; \
    chmod 0755 /opt/cosmos/cosmos-transfer2.5/.venv/bin/npa; \
    env -u PYTHONPATH "${NPA_BAKED_PYTHON}" -c \
      'import npa.cli.main; import npa.workflows.sim2real.runtime_attestation'; \
    /opt/cosmos/cosmos-transfer2.5/.venv/bin/npa \
      workbench cosmos2 transfer --help > /tmp/npa-cosmos2-help.txt; \
    grep -q -- '--control-asset' /tmp/npa-cosmos2-help.txt; \
    rm -f /tmp/npa-cli-requirements.txt /tmp/npa-cosmos2-help.txt

VOLUME ["/opt/cosmos/model-cache"]
WORKDIR /opt/cosmos/cosmos-transfer2.5
USER ubuntu

# An orchestrator-supplied argv must execute verbatim; this is the SkyPilot setup
# behavior fixed by PR #231.
ENTRYPOINT ["/opt/cosmos2-transfer/entrypoint.sh"]
