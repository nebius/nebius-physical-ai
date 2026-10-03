# Private BYOF validation image for the official MolmoAct2 LIBERO workflow.
# It intentionally contains no checkpoints, datasets, Hugging Face token, or
# model cache.  The workflow retrieves immutable public inputs at run time.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:da6791294b0b04d7e65d87b7451d6f2390b4d36225ab0701ee7dfec5769829f5

ARG MOLMOACT2_REVISION=6070080a20321b4f498ab30f28e1d09ac465edb7

LABEL org.opencontainers.image.source="https://github.com/allenai/molmoact2" \
      org.opencontainers.image.revision="${MOLMOACT2_REVISION}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      npa.solution="molmoact2-official-libero-byof" \
      npa.redistribution="unvalidated-private-byof"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/npa/src \
    MOLMOACT2_UPSTREAM_ROOT=/opt/molmoact2/experiments \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    WANDB_MODE=disabled

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake ffmpeg git git-lfs libegl1 libgl1 \
      libglib2.0-0 libsm6 libxext6 libxrender1 netcat-openbsd openssh-client \
      openssh-server procps python3 python3-pip python3-venv rsync sudo \
      software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-dev python3.12-venv \
    && rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install boto3 huggingface_hub rerun-sdk
ENV PATH=/opt/venv/bin:$PATH

# Fetch source only. LFS smudging is disabled so no checkpoint/data payload can
# enter a layer. The runnable trainer imports the ``experiments/lerobot`` tree
# committed at this source revision; it is not the separate root gitlink.
RUN git lfs install --skip-smudge \
    && GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout https://github.com/allenai/molmoact2.git /opt/molmoact2 \
    && git -C /opt/molmoact2 config filter.lfs.process '' \
    && git -C /opt/molmoact2 config filter.lfs.required false \
    && git -C /opt/molmoact2 config filter.lfs.smudge cat \
    && cd /opt/molmoact2 \
    && GIT_LFS_SKIP_SMUDGE=1 git checkout --force --detach "${MOLMOACT2_REVISION}" \
    && test -f experiments/lerobot/pyproject.toml \
    && rm -rf /opt/molmoact2/.git

# The two upstream dependency sets both permit this current protobuf release;
# installing it after LeRobot lets `pip check` verify the assembled runtime.
# The editable LeRobot source includes tiny test fixture tensors/datasets. They
# are not runtime code and must not be redistributed in the BYOF runtime image.
RUN cd /opt/molmoact2/experiments \
    && pip install -e ".[all]" \
    && pip install -e "./lerobot[async,libero]" \
    && pip install "protobuf>=6.33.5,<7" \
    && pip check \
    && rm -rf /opt/molmoact2/experiments/lerobot/tests

COPY docker/workbench/molmoact2/THIRD_PARTY_NOTICES.md /usr/share/doc/molmoact2/THIRD_PARTY_NOTICES.md
COPY docker/workbench/molmoact2/REDISTRIBUTION.md /usr/share/doc/molmoact2/REDISTRIBUTION.md
COPY src/npa /opt/npa/src/npa

RUN useradd --create-home --shell /bin/bash --uid 1000 ubuntu \
    && printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/npa-runtime \
    && chmod 0440 /etc/sudoers.d/npa-runtime \
    && install -d -o ubuntu -g ubuntu /work /opt/cache \
    && chown -R ubuntu:ubuntu /opt/npa /opt/molmoact2 /usr/share/doc/molmoact2

USER ubuntu
WORKDIR /work
CMD ["python3", "-m", "npa.workflows.byof.molmoact2_pipeline", "--help"]
