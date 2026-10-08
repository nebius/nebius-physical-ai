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
    NPA_LEROBOT_VERSION=${LEROBOT_VERSION} \
    IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg

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

COPY --chown=ubuntu:ubuntu docker/workbench/curobo/remove_scikit_image_recipe.py \
    /opt/lerobot/remove-scikit-image-recipe.py
COPY --chown=ubuntu:ubuntu docker/workbench/lerobot/default-runtime-requirements.txt \
    /opt/lerobot/default-constraints.txt
COPY --chown=ubuntu:ubuntu docker/workbench/lerobot/smoke_native_cpu.py \
    /opt/lerobot/smoke-native-cpu.py
COPY --chown=ubuntu:ubuntu docker/workbench/lerobot/prepare_secure_wheel.py \
    /opt/lerobot/prepare-secure-wheel.py

ENV PIP_NO_CACHE_DIR=1

# 0.5.1 is an explicitly versioned, source-verified NPA integration: the upstream
# model code and license are retained, and dependency changes face native gates.
# 0.6.0 uses lean feature extras and must NOT force torch>=2.12 (upstream ceiling).
# 0.6.0 also moved diffusers/transformers behind extras and enforces them in each
# policy's __init__, so diffusion and smolvla must be requested explicitly or
# --policy.type=diffusion fails inside make_policy() at run time, not at build.
RUN python3.12 -m venv /opt/lerobot/venv \
    && /opt/lerobot/venv/bin/python -m pip install --upgrade pip setuptools wheel \
    && if [ "${LEROBOT_VERSION}" = "0.6.0" ]; then \
         /opt/lerobot/venv/bin/pip install \
           "lerobot[training,evaluation,pusht,libero,diffusion,smolvla]==${LEROBOT_VERSION}" \
           boto3 \
           fastapi~=0.136.1 \
           num2words \
           tensorboard \
           "uvicorn[standard]" \
         && /opt/lerobot/venv/bin/pip uninstall -y wandb; \
       elif [ "${LEROBOT_VERSION}" = "0.5.1" ]; then \
         /opt/lerobot/venv/bin/python /opt/lerobot/prepare-secure-wheel.py \
           --output-dir /opt/lerobot/secure-wheel \
         && /opt/lerobot/venv/bin/pip install \
           --constraint /opt/lerobot/default-constraints.txt \
           'lerobot[pusht,libero] @ file:///opt/lerobot/secure-wheel/lerobot-0.5.1+npa.secure1-py3-none-any.whl' \
           "scikit-image==0.26.0" \
           boto3 \
           fastapi~=0.136.1 \
           num2words \
           tensorboard \
           "uvicorn[standard]" \
         && cp /opt/lerobot/secure-wheel/source-integration.json \
           /opt/lerobot/source-integration.json \
         && rm -rf /opt/lerobot/secure-wheel \
         && /opt/lerobot/venv/bin/python /opt/lerobot/remove-scikit-image-recipe.py \
           --site-packages /opt/lerobot/venv/lib/python3.12/site-packages \
           > /opt/lerobot/dependency-source-correction.json \
         && /opt/lerobot/venv/bin/pip check; \
       else \
         echo 'Unsupported LeRobot package version' >&2; exit 1; \
       fi \
    && find /opt/lerobot/venv -type f \
         -path '*/imageio_ffmpeg/binaries/ffmpeg*' -delete \
    && test "$(/opt/lerobot/venv/bin/python -c \
         'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" \
         = /usr/bin/ffmpeg \
    && test -z "$(find /opt/lerobot/venv -type f \
         -path '*/imageio_ffmpeg/binaries/ffmpeg*' -print -quit)"

ENV PATH=/opt/lerobot/venv/bin:$PATH \
    PYTHONPATH=/opt/npa/src \
    NPA_SKIP_EAGER_IMPORTS=1

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
# Keep this image-local SDK intentionally partial. The stage's own NPA install
# must supply npa.workbench; shipping it here would make the repair's import
# probe pass against the baked tree.
COPY --chown=ubuntu:ubuntu src/npa/__init__.py /opt/npa/src/npa/__init__.py
COPY --chown=ubuntu:ubuntu src/npa/clients /opt/npa/src/npa/clients
COPY --chown=ubuntu:ubuntu src/npa/server /opt/npa/src/npa/server
COPY --chown=ubuntu:ubuntu src/npa/smoke /opt/npa/src/npa/smoke

RUN python -m npa.smoke.test_lerobot_env

USER root

# SkyPilot-on-Kubernetes runtime prerequisites. Installed BEFORE the CVE hardening below:
# that step purges linux-libc-dev with --force-depends, which strands libc6-dev and its
# reverse-dependencies and leaves apt refusing to install anything at all (observed
# in-cluster: even plain `rsync` was blocked). Installing first keeps apt healthy and the
# hardening intact.
# SkyPilot prerequisites, so this image can host a workflow stage.
# SkyPilot's bootstrap runs inside the task container; without these, provisioning fails with
# "container not found (ray-node)" (see Dockerfile.k8s-prereqs and
# npa/tests/guardrails/test_workbench_image_k8s_prereqs.py).
#
# The PATH line flips the vendor venv OFF the front of PATH, which SkyPilot requires. The
# vendor python stays reachable by absolute path, and the renderer's
# TOOL_REF_VENDOR_INTERPRETERS installs npa into it and records it as the stage interpreter —
# without that half, a tool importing `lerobot` would run under the system python and fail.
# Only these three packages: this image purges linux-libc-dev as CVE hardening, so pulling
# anything that depends on libc6-dev (python3-pip, openssh-client) breaks apt. The system
# python3 the bootstrap needs is already present, so it is asserted rather than installed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends rsync sudo netcat-openbsd \
    && rm -rf /var/lib/apt/lists/* \
    && printf '%s ALL=(ALL) NOPASSWD:ALL\n' ubuntu \
         > /etc/sudoers.d/99-npa-runtime-user \
    && chmod 0440 /etc/sudoers.d/99-npa-runtime-user \
    && test -x /usr/bin/python3 \
    && /usr/bin/python3 --version

ENV PATH=/usr/bin:$PATH

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

# Exercise the final hardened 0.5.1 integration and declared server interpreter.
# The separately qualified 0.6.0 recipe retains its existing acceptance path.
RUN if [ "${LEROBOT_VERSION}" = "0.5.1" ]; then \
      /opt/lerobot/venv/bin/python /opt/lerobot/smoke-native-cpu.py; \
    fi

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD /opt/lerobot/venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)" || exit 1

ENTRYPOINT ["/opt/lerobot/venv/bin/python", "-m", "npa.server.app"]
