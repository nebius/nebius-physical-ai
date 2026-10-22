# This Python 3.11 recipe requires an explicit, source-attested CUDA 13
# workbench parent. The catalog's historical npa-base digest is evidence for
# that historical image only; it is not a fallback for a current-source build.
# Build the matching local npa-base recipe first, then pass its immutable local
# image ID or source-SHA tag with --build-arg BASE_IMAGE=... .
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG NPA_SOURCE_SHA
ARG GENESIS_VERSION=0.4.6
ARG LEROBOT_VERSION=0.4.4
ARG RSL_RL_VERSION=5.5.1
ARG UBUNTU_SNAPSHOT=20261002T000000Z

LABEL npa.tool="genesis" \
      npa.version="${GENESIS_VERSION}" \
      npa.image.flavor="sm80-sm90-sm100-sm103-sm120" \
      npa.base.profile="cuda13-workbench" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}"

ENV DEBIAN_FRONTEND=noninteractive \
    GENESIS_VERSION=${GENESIS_VERSION} \
    LEROBOT_VERSION=${LEROBOT_VERSION} \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NPA_GENESIS_SUPPORTED_STUDENT_POLICIES=act \
    NPA_GENESIS_HOME=/opt/genesis \
    GENESIS_VENV=/opt/genesis/venv \
    NPA_BAKED_PYTHON=/opt/genesis/venv/bin/python \
    NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility \
    EGL_PLATFORM=device \
    PATH=/opt/genesis/venv/bin:/opt/npa/venv/bin:$PATH \
    PYTHONPATH=/opt/npa/src \
    NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    NPA_SKIP_EAGER_IMPORTS=1

SHELL ["/bin/bash", "-lc"]

USER root

RUN id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu; \
    install -d -m 0755 -o ubuntu -g ubuntu \
      /opt/genesis \
      /opt/genesis/outputs \
      /opt/npa/src/npa \
      /opt/npa/docker/workbench/genesis; \
    groupadd -g 44 video 2>/dev/null || true; \
    groupadd -g 992 render 2>/dev/null || true; \
    usermod -aG video,render ubuntu

# npa-base force-purges linux-libc-dev in its final hardening layer. Restore
# build headers only while native dependencies install; the final layer purges
# them again. Its Python is the base's audited Python 3.11 virtual environment.
RUN apt-get update \
    && apt-get install -y --no-install-recommends --fix-broken

COPY --chmod=0644 docker/workbench/curobo/remove_scikit_image_recipe.py \
    /opt/genesis/remove-scikit-image-recipe.py
COPY --chmod=0644 docker/workbench/genesis/prepare_public_runtime.py \
    /opt/genesis/prepare-public-runtime.py
COPY --chmod=0644 docker/workbench/genesis/prepare_public_policies.py \
    /opt/genesis/prepare-public-policies.py
COPY --chmod=0644 docker/workbench/common/envgen_compat/tetgen.py \
    /opt/genesis/compat/tetgen.py

USER ubuntu
WORKDIR /opt/genesis

# Keep the base's CUDA 13/Torch 2.13 environment intact.  Genesis 0.4.6's
# supported LeRobot 0.4.4 closure declares torch <2.11, so it belongs in its
# own Python 3.11 venv rather than silently downgrading the shared Blackwell
# runtime.  The final runtime selects this venv explicitly through PATH and
# NPA_BAKED_PYTHON.
RUN python3.11 -m venv "${GENESIS_VENV}"

# Keep the publicly redistributable Genesis capability intentionally narrow:
# native rigid Panda/Box and ACT only. The exact metadata/source correction and
# CPU round trip below fail closed if an upstream wheel changes.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-binary imageio-ffmpeg \
      "genesis-world==${GENESIS_VERSION}" \
      "imageio-ffmpeg==0.6.0" \
      "scikit-image==0.26.0" \
      "rsl-rl-lib==${RSL_RL_VERSION}" \
      "moviepy @ https://github.com/Zulko/moviepy/archive/97316f37f6a8d3843abfb53eba8f3bb0ea46a008.tar.gz#sha256=6bef8575b091f6a7342ed271ff5dba97dbb07255df0a5b81fec38f8248b143d9" \
      "pillow>=13.0.0,<14.0" \
      "tensorboard==2.20.0" \
      "lerobot==${LEROBOT_VERSION}" \
      "huggingface-hub==0.35.3" \
      "av>=15,<16" \
      "onnxruntime>=1.23,<2" \
      "typer==0.24.1" \
      "httpx==0.28.1" \
      "paramiko==5.0.0" \
      "pyyaml==6.0.3" \
      "rich==15.0.0" \
      "boto3==1.42.91" \
      "jinja2==3.1.6" \
      "pyarrow==24.0.0" \
    && python -m pip install --no-cache-dir --no-deps --upgrade \
      "pillow>=13.0.0,<14.0" "fsspec==2026.6.0" \
    && python /opt/genesis/prepare-public-runtime.py \
      --site-packages /opt/genesis/venv/lib/python3.11/site-packages --check-only \
    && python /opt/genesis/prepare-public-policies.py \
      --site-packages /opt/genesis/venv/lib/python3.11/site-packages --check-only \
    && python -m pip uninstall -y tetgen diffusers wandb torchcodec transformers peft \
    && python /opt/genesis/prepare-public-runtime.py \
      --site-packages /opt/genesis/venv/lib/python3.11/site-packages \
      > /opt/genesis/public-runtime-dependencies.json \
    && python /opt/genesis/prepare-public-policies.py \
      --site-packages /opt/genesis/venv/lib/python3.11/site-packages \
      > /opt/genesis/public-policy-source.json \
    && printf '%s\n' /opt/genesis/compat \
      > /opt/genesis/venv/lib/python3.11/site-packages/npa-genesis-compat.pth \
    && python -m pip check \
    && test -z "$(find /opt/genesis/venv -type f \
      -path '*/imageio_ffmpeg/binaries/ffmpeg*' -print -quit)" \
    && python -c 'import imageio_ffmpeg; assert imageio_ffmpeg.get_ffmpeg_exe() == "/usr/bin/ffmpeg"' \
    && python /opt/genesis/remove-scikit-image-recipe.py \
      --site-packages /opt/genesis/venv/lib/python3.11/site-packages \
      --capability genesis \
      > /opt/genesis/dependency-source-correction.json

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu workflows /opt/npa/workflows
COPY --chown=ubuntu:ubuntu docker/workbench/genesis/smoke_env.py /opt/npa/docker/workbench/genesis/smoke_env.py
COPY --chown=ubuntu:ubuntu docker/workbench/genesis/smoke_functional.py /opt/npa/docker/workbench/genesis/smoke_functional.py
COPY --chown=ubuntu:ubuntu docker/workbench/genesis/smoke_public_cpu.py /opt/npa/docker/workbench/genesis/smoke_public_cpu.py
COPY --chmod=0555 docker/workbench/common/configure_ubuntu_snapshot.sh \
    /usr/local/bin/configure-ubuntu-snapshot
COPY --chmod=0555 docker/workbench/common/install_workflow_runtime_prereqs.sh \
    /usr/local/bin/install-workflow-runtime-prereqs

RUN test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40 \
    && python -m pip install --no-deps -e /opt/npa \
    && python - <<'PY'
import os
from importlib import metadata

import genesis
import torch

expected = os.environ["GENESIS_VERSION"]
actual = getattr(genesis, "__version__", metadata.version("genesis-world"))
if actual != expected:
    raise RuntimeError(f"expected genesis-world {expected}, found {actual}")
print(f"GENESIS_BUILD_IMPORT_OK {actual}")
print(f"TORCH_BUILD_CUDA {torch.__version__} cuda={torch.version.cuda}")
PY

RUN python /opt/npa/docker/workbench/genesis/smoke_public_cpu.py

USER root

# The normal NPA/SkyPilot runtime starts its controller bootstrap in this
# non-root container. Reuse the pinned shared closure rather than a local
# approximation so rsync, sudo, and sshd retain the verified user/key contract.
RUN install-workflow-runtime-prereqs "${UBUNTU_SNAPSHOT}"

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
WORKDIR /opt/genesis
ENTRYPOINT ["/bin/bash"]
