# syntax=docker/dockerfile:1.7

# Public NVIDIA Content Agents adapter image. The reviewed Apache-2.0 source and
# immutable upstream OVRTX lock are included; the proprietary OVRTX SDK is not.
# Render stages fetch the exact SDK directly from NVIDIA into operator-owned
# runtime-cache storage on first use.
ARG CUDA_BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.8.12@sha256:e1d7fa999ae39871fc2fd6c9c572d970cc877a239d2a16238a204c23772d1c0e
ARG UBUNTU_SNAPSHOT=20260801T053000Z

FROM ${UV_IMAGE} AS uv-bin
FROM ${CUDA_BASE_IMAGE} AS assembled

ARG CONTENT_AGENTS_REVISION=36dbf3f274f8e256637230a05a085853f65cc175
ARG CONTENT_AGENTS_VERSION=0.5.2
ARG NPA_IMAGE_VERSION=0.5.2-npa2
ARG NPA_SOURCE_SHA
ARG UBUNTU_SNAPSHOT

ENV DEBIAN_FRONTEND=noninteractive \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NPA_SKIP_EAGER_IMPORTS=1 \
    NPA_CONTENT_AGENTS_REVISION=${CONTENT_AGENTS_REVISION} \
    WU_OVRTX_AUTO_PROVISION=0 \
    OTEL_SDK_DISABLED=true \
    OTEL_TRACES_EXPORTER=none \
    OTEL_METRICS_EXPORTER=none \
    OTEL_LOGS_EXPORTER=none \
    OVRTX_LOG_LEVEL=warn \
    OVRTX_NUM_SENSOR_UPDATES=500 \
    OVRTX_RENDER_MODE=pt \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
    NVIDIA_VISIBLE_DEVICES=all \
    PYTHONPATH=/opt/npa-runtime:/opt/content-agents:/opt/content-agents/apps \
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
    rm -rf .git .github .build-resources docs tests assets; \
    rm -rf apps/*/tests apps/*/examples apps/*/data; \
    find . -type f \( -name '.env*' -o -iname '*.usd' -o -iname '*.usda' \
      -o -iname '*.usdc' -o -iname '*.usdz' -o -iname '*.png' \
      -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \
      -o -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mcap' \
      -o -iname '*.bag' \) -delete; \
    find . -type d -name __pycache__ -prune -exec rm -rf -- {} +

# Follow the upstream v0.5.2 service installation path. Scene Optimizer Core,
# optional telemetry exporters, backend wheels, model weights, sample assets,
# OvPhysX, and OVRTX are not installed. The upstream OVRTX lock remains as the
# reviewed runtime download description only.
RUN set -eux; \
    SETUPTOOLS_SCM_PRETEND_VERSION="${CONTENT_AGENTS_VERSION}" \
      UV_DYNAMIC_VERSIONING_BYPASS="${CONTENT_AGENTS_VERSION}" \
      uv pip install --python /opt/venv/bin/python -e . \
      --no-config --no-sources; \
    for package in material_agent physics_agent validation_agent; do \
      cd "/opt/content-agents/apps/${package}"; \
      UV_DYNAMIC_VERSIONING_BYPASS="${CONTENT_AGENTS_VERSION}" \
        uv pip install --python /opt/venv/bin/python -e . --no-config --no-sources; \
    done; \
    cd /opt/content-agents; \
    uv pip install --python /opt/venv/bin/python 'boto3>=1.34' 'pyyaml>=6.0' \
      --no-config --no-sources; \
    /opt/venv/bin/python -c \
      'import importlib.util; assert importlib.util.find_spec("ovphysx") is None'; \
    /opt/venv/bin/python -c \
      'import importlib.util; assert importlib.util.find_spec("ovrtx") is None'; \
    rm -rf /root/.cache/uv; \
    find /opt/venv/lib/python3.12/site-packages/boto3 -type d \
      -name examples -prune -exec rm -rf -- {} +; \
    find /opt/venv/lib/python3.12/site-packages/botocore/data \
      -type f -name 'examples-1.json' -delete; \
    find /opt/venv/lib/python3.12/site-packages/botocore/data \
      -mindepth 1 -maxdepth 1 -type d ! -name s3 -exec rm -rf -- {} +; \
    /opt/venv/bin/python -c \
      'import boto3; client = boto3.client("s3", endpoint_url="https://127.0.0.1", aws_access_key_id="x", aws_secret_access_key="x"); assert client.meta.service_model.service_name == "s3"'; \
    test ! -e .build-resources/scene_optimizer_core

COPY src/npa/__init__.py /opt/npa-runtime/npa/__init__.py
COPY src/npa/clients/__init__.py /opt/npa-runtime/npa/clients/__init__.py
COPY src/npa/clients/storage.py /opt/npa-runtime/npa/clients/storage.py
COPY src/npa/workflows/__init__.py /opt/npa-runtime/npa/workflows/__init__.py
COPY src/npa/workflows/content_agents.py \
    /opt/npa-runtime/npa/workflows/content_agents.py
COPY src/npa/workflows/content_agents_runtime.py \
    /opt/npa-runtime/npa/workflows/content_agents_runtime.py
COPY docker/workbench/content-agents/REDISTRIBUTION.md \
    /usr/share/doc/npa-content-agents/REDISTRIBUTION.md
COPY docker/workbench/content-agents/npa-content-agents-entrypoint \
    /usr/local/bin/npa-content-agents-entrypoint

# The scanner requires this exact full NPA source SHA from the finished image.
LABEL npa.source_revision="${NPA_SOURCE_SHA}" \
      npa.driver_provisioning="gpu-operator-host-mounted" \
      npa.driver_capabilities="compute,utility,graphics,display"

# Content Agents toolRefs invoke only the reviewed module adapter. Do not carry
# the rest of the NPA repository, CLI, examples, credentials guidance, or other
# workbench implementations in this narrow image.
RUN /opt/venv/bin/python -c \
      'from npa.workflows import content_agents, content_agents_runtime; assert content_agents and content_agents_runtime'

# SkyPilot currently performs privileged SSH/bootstrap setup in each ephemeral
# task pod.  The final process is uid 1000, host keys are generated only at run
RUN set -eux; \
    test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40; \
    id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu; \
    printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu; \
    chmod 0440 /etc/sudoers.d/ubuntu; \
    chmod 0555 /usr/local/bin/npa-content-agents-entrypoint; \
    rm -f /etc/ssh/ssh_host_*; \
    install -d -o ubuntu -g ubuntu /workspace /home/ubuntu/.cache; \
    chmod -R a+rX /opt/content-agents /opt/npa-runtime; \
    su ubuntu -s /bin/bash -c \
      '/opt/venv/bin/python -m npa.workflows.content_agents inspect-image >/tmp/content-agents-image.json'; \
    grep -q '"status": "image-ready"' /tmp/content-agents-image.json; \
    rm -f /tmp/content-agents-image.json; \
    test ! -e /opt/content-agents/.ovrtx_venv; \
    find /opt/content-agents /opt/npa-runtime -type d -name __pycache__ \
      -prune -exec rm -rf -- {} +; \
    # Header-only build input inherited through build-essential. Runtime fetch
    # installs the reviewed prebuilt OVRTX wheel and never compiles native code;
    # retaining linux-libc-dev would only expose an avoidable fixed CVE.
    PATH="/usr/local/sbin:/usr/sbin:/sbin:${PATH}" \
      dpkg --purge --force-depends linux-libc-dev; \
    rm -rf /usr/local/cuda-*/compat /root/.cache /home/ubuntu/.cache/*

# The CUDA base's driver-compatibility directory is useful on unmanaged hosts,
# but NPA GPU pods receive their graphics/display stack from the host. Assemble
# a clean final filesystem so neither those driver bytes nor deleted upstream
# samples remain in any layer of the published image.
FROM scratch AS public-image

ARG CONTENT_AGENTS_REVISION=36dbf3f274f8e256637230a05a085853f65cc175
ARG CONTENT_AGENTS_VERSION=0.5.2
ARG NPA_IMAGE_VERSION=0.5.2-npa2
ARG NPA_SOURCE_SHA

COPY --from=assembled / /

LABEL org.opencontainers.image.title="NPA NVIDIA Content Agents" \
      org.opencontainers.image.description="Public zero-vendor-payload RTX USD material, physics, and validation adapter" \
      org.opencontainers.image.source="https://github.com/NVIDIA-Omniverse/content-agents" \
      org.opencontainers.image.revision="${CONTENT_AGENTS_REVISION}" \
      org.opencontainers.image.version="${NPA_IMAGE_VERSION}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      npa.tool="content-agents" \
      npa.content_agents.version="${CONTENT_AGENTS_VERSION}" \
      npa.ovrtx.delivery="runtime-fetch-from-nvidia" \
      npa.ovrtx.version="0.3.0.312915" \
      npa.redistribution="public" \
      npa.source_revision="${NPA_SOURCE_SHA}" \
      npa.driver_provisioning="gpu-operator-host-mounted" \
      npa.driver_capabilities="compute,utility,graphics,display" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

ENV DEBIAN_FRONTEND=noninteractive \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NPA_SKIP_EAGER_IMPORTS=1 \
    NPA_CONTENT_AGENTS_REVISION=${CONTENT_AGENTS_REVISION} \
    WU_OVRTX_AUTO_PROVISION=0 \
    OTEL_SDK_DISABLED=true \
    OTEL_TRACES_EXPORTER=none \
    OTEL_METRICS_EXPORTER=none \
    OTEL_LOGS_EXPORTER=none \
    OVRTX_LOG_LEVEL=warn \
    OVRTX_NUM_SENSOR_UPDATES=500 \
    OVRTX_RENDER_MODE=pt \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
    NVIDIA_VISIBLE_DEVICES=all \
    PYTHONPATH=/opt/npa-runtime:/opt/content-agents:/opt/content-agents/apps \
    PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin \
    HOME=/home/ubuntu \
    XDG_CACHE_HOME=/home/ubuntu/.cache

WORKDIR /workspace
USER ubuntu

# Xvfb is a local OVRTX runtime dependency, not a service/orchestrator. Direct
# container runs use this entrypoint. SkyPilot replaces image entrypoints, so
# the workflow renderer invokes the same wrapper in render-stage run shells.
ENTRYPOINT ["/usr/local/bin/npa-content-agents-entrypoint"]
