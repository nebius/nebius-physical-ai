# syntax=docker/dockerfile:1.7

# Restricted, build-your-own NVIDIA Content Agents runtime.  The Content Agents
# source is Apache-2.0, but the isolated OVRTX wheel installed below is NVIDIA
# Proprietary Software.  This image must remain in the operator's private
# registry and is never eligible for the public mirror.
ARG CUDA_BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.8.12@sha256:e1d7fa999ae39871fc2fd6c9c572d970cc877a239d2a16238a204c23772d1c0e
ARG UBUNTU_SNAPSHOT=20260801T053000Z

FROM ${UV_IMAGE} AS uv-bin
FROM ${CUDA_BASE_IMAGE}

ARG CONTENT_AGENTS_REVISION=36dbf3f274f8e256637230a05a085853f65cc175
ARG CONTENT_AGENTS_VERSION=0.5.2
ARG NPA_SOURCE_SHA
ARG UBUNTU_SNAPSHOT

LABEL org.opencontainers.image.title="NPA NVIDIA Content Agents" \
      org.opencontainers.image.description="Restricted RTX USD material, physics, and validation job runtime" \
      org.opencontainers.image.source="https://github.com/NVIDIA-Omniverse/content-agents" \
      org.opencontainers.image.revision="${CONTENT_AGENTS_REVISION}" \
      org.opencontainers.image.version="${CONTENT_AGENTS_VERSION}" \
      org.opencontainers.image.licenses="Apache-2.0 AND LicenseRef-NVIDIA-Proprietary-OVRTX" \
      npa.tool="content-agents" \
      npa.redistribution="restricted" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

ENV DEBIAN_FRONTEND=noninteractive \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NPA_SKIP_EAGER_IMPORTS=1 \
    NPA_CONTENT_AGENTS_REVISION=${CONTENT_AGENTS_REVISION} \
    WU_OVRTX_VENV_DIR=/opt/content-agents/.ovrtx_venv \
    WU_OVRTX_AUTO_PROVISION=0 \
    OVRTX_LOG_LEVEL=warn \
    OVRTX_NUM_SENSOR_UPDATES=500 \
    OVRTX_RENDER_MODE=pt \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
    NVIDIA_VISIBLE_DEVICES=all \
    PYTHONPATH=/opt/npa/src:/opt/content-agents:/opt/content-agents/apps \
    PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin

COPY --from=uv-bin /uv /usr/local/bin/uv

# Snapshot the distro inputs used by the immutable base, including SkyPilot's
# non-root Kubernetes bootstrap prerequisites and OVRTX's Vulkan/X11 runtime.
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
      build-essential ca-certificates curl git libgl1 libglib2.0-0 libgomp1 \
      libopengl0 libvulkan1 libx11-6 libxt6 netcat-openbsd openssh-server \
      procps python3.12 python3.12-dev python3.12-venv rsync sudo tini xvfb \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/ssh/ssh_host_* \
    && mkdir -p /run/sshd /tmp/.X11-unix \
    && chmod 1777 /tmp/.X11-unix

RUN python3.12 -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir --upgrade 'pip>=26.1'

WORKDIR /opt/content-agents

# Fetch exactly one reviewed release commit.  The Git database is removed in
# the same layer, and no LFS/example/model payload is fetched or retained.
RUN set -eux; \
    git init; \
    git remote add origin https://github.com/NVIDIA-Omniverse/content-agents.git; \
    GIT_LFS_SKIP_SMUDGE=1 git -c filter.lfs.smudge= -c filter.lfs.required=false \
      fetch --depth=1 --filter=blob:none origin "${CONTENT_AGENTS_REVISION}"; \
    GIT_LFS_SKIP_SMUDGE=1 git -c filter.lfs.smudge= -c filter.lfs.required=false \
      checkout --detach FETCH_HEAD; \
    test "$(git rev-parse HEAD)" = "${CONTENT_AGENTS_REVISION}"; \
    test "$(tr -d '\n' < VERSION.md)" = "${CONTENT_AGENTS_VERSION}"; \
    cp apps/ovrtx_rendering_api/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; \
    chmod 0644 /etc/vulkan/icd.d/nvidia_icd.json; \
    rm -rf .git .github .build-resources; \
    find . -type d -name __pycache__ -prune -exec rm -rf -- {} +

# Follow the upstream v0.5.2 service installation path.  Scene Optimizer Core,
# optional backend wheels, model weights, sample assets, and OvPhysX are not
# installed.  OVRTX is installed separately from its upstream hash lock because
# its bundled OpenUSD conflicts with the parent process by design.
RUN set -eux; \
    SETUPTOOLS_SCM_PRETEND_VERSION="${CONTENT_AGENTS_VERSION}" \
      UV_DYNAMIC_VERSIONING_BYPASS="${CONTENT_AGENTS_VERSION}" \
      uv pip install --python /opt/venv/bin/python -e '.[telemetry]' \
      --no-config --no-sources; \
    for package in material_agent physics_agent validation_agent; do \
      cd "/opt/content-agents/apps/${package}"; \
      UV_DYNAMIC_VERSIONING_BYPASS="${CONTENT_AGENTS_VERSION}" \
        uv pip install --python /opt/venv/bin/python -e . --no-config --no-sources; \
    done; \
    cd /opt/content-agents; \
    uv pip install --python /opt/venv/bin/python 'boto3>=1.34' 'pyyaml>=6.0' \
      --no-config --no-sources; \
    uv venv "${WU_OVRTX_VENV_DIR}" --python /opt/venv/bin/python; \
    uv pip install --python "${WU_OVRTX_VENV_DIR}/bin/python" \
      -r world_understanding/functions/graphics/pylock.ovrtx-runtime.toml \
      --require-hashes --no-deps --no-config --no-sources; \
    WU_OVRTX_LOCK_DIR=/tmp/wu-ovrtx-locks \
      /opt/venv/bin/python -m world_understanding.functions.graphics.render_ovrtx \
      --provision-only; \
    /opt/venv/bin/python -c \
      'import importlib.util; assert importlib.util.find_spec("ovphysx") is None'; \
    rm -rf /tmp/wu-ovrtx-locks /root/.cache/uv \
      apps/*/tests apps/*/examples docs tests; \
    rm -rf apps/material_agent/data apps/physics_agent/data; \
    test ! -e .build-resources/scene_optimizer_core

COPY --chown=1000:1000 pyproject.toml /opt/npa/pyproject.toml
COPY --chown=1000:1000 src/npa /opt/npa/src/npa
COPY --chown=1000:1000 workflows /opt/npa/workflows
COPY docker/workbench/content-agents/REDISTRIBUTION.md \
    /usr/share/doc/npa-content-agents/REDISTRIBUTION.md
COPY docker/workbench/content-agents/npa-content-agents-entrypoint \
    /usr/local/bin/npa-content-agents-entrypoint

# Keep changing NPA provenance after the pinned NVIDIA/OVRTX layers so an
# adapter-only checkpoint does not invalidate the multi-gigabyte restricted
# runtime cache.  The scanner still requires this exact full SHA from the
# finished image.
LABEL npa.source_revision="${NPA_SOURCE_SHA}" \
      npa.driver_provisioning="gpu-operator-host-mounted" \
      npa.driver_capabilities="compute,utility,graphics,display"

# The workflow renderer invokes the first-party ``npa`` console entry point in
# every toolRef stage.  Copying the package source makes module imports work but
# does not create that executable, so install the already-vendored package
# without resolving any additional runtime dependencies.  The Content Agents
# environment above already carries the narrow CLI dependencies used here.
RUN uv pip install --python /opt/venv/bin/python /opt/npa \
      --no-deps --no-config --no-sources \
    && command -v npa \
    && npa --version

# SkyPilot currently performs privileged SSH/bootstrap setup in each ephemeral
# task pod.  The final process is uid 1000, host keys are generated only at run
# time, and neither sshd nor a network service is started by this image.
RUN set -eux; \
    test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40; \
    id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu; \
    printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu; \
    chmod 0440 /etc/sudoers.d/ubuntu; \
    chmod 0555 /usr/local/bin/npa-content-agents-entrypoint; \
    rm -f /etc/ssh/ssh_host_*; \
    install -d -o ubuntu -g ubuntu /workspace /home/ubuntu/.cache/ovrtx-nv-shadercache; \
    ovrtx_bin="${WU_OVRTX_VENV_DIR}/lib/python3.12/site-packages/ovrtx/bin"; \
    mkdir -p "${ovrtx_bin}/cache"; \
    rm -rf "${ovrtx_bin}/cache/nv_shadercache"; \
    chown -R ubuntu:ubuntu "${ovrtx_bin}/cache"; \
    ln -s /home/ubuntu/.cache/ovrtx-nv-shadercache "${ovrtx_bin}/cache/nv_shadercache"; \
    chmod -R a+rX /opt/content-agents /opt/npa/src; \
    su ubuntu -s /bin/bash -c \
      '/opt/venv/bin/python -m npa.workflows.content_agents inspect-runtime >/tmp/content-agents-runtime.json'; \
    grep -q '"status": "ready"' /tmp/content-agents-runtime.json; \
    rm -f /tmp/content-agents-runtime.json

ENV HOME=/home/ubuntu \
    XDG_CACHE_HOME=/home/ubuntu/.cache

WORKDIR /workspace
USER ubuntu

# Xvfb is a local OVRTX runtime dependency, not a service/orchestrator. Direct
# container runs use this entrypoint. SkyPilot replaces image entrypoints, so
# the workflow renderer invokes the same wrapper in render-stage run shells.
ENTRYPOINT ["/usr/local/bin/npa-content-agents-entrypoint"]
