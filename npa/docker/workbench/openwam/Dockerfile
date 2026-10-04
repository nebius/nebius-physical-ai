# Operator-private, unvalidated OpenWAM runtime.  Model, backbone, data, and
# every populated cache are fetched by the workflow at run time, never baked.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66

ARG NPA_SOURCE_SHA=unknown
ARG OPENWAM_SOURCE_URL=https://github.com/OpenWAM-Official/OpenWAM.git
ARG OPENWAM_SOURCE_REF=48bd67b89d489b14d03b8d92bc66e65d306df32e
ARG LIBERO_SOURCE_URL=https://github.com/Lifelong-Robot-Learning/LIBERO.git
ARG LIBERO_SOURCE_REF=8f1084e3132a39270c3a13ebe37270a43ece2a01

LABEL npa.tool="openwam" \
      npa.source.repo="${OPENWAM_SOURCE_URL}" \
      npa.source.ref="${OPENWAM_SOURCE_REF}" \
      npa.weights="operator-runtime-fetch" \
      npa.dataset="operator-runtime-fetch" \
      org.opencontainers.image.source="https://github.com/OpenWAM-Official/OpenWAM" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT AND CC-BY-4.0" \
      org.nebius.npa.redistribution="unvalidated-operator-private" \
      org.nebius.npa.third-party-notices="/usr/share/doc/npa-openwam/THIRD_PARTY_NOTICES.md" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/ubuntu \
    HF_HOME=/workspace/.cache/huggingface \
    LIBERO_PYTHON=/opt/openwam-libero/bin/python \
    PATH=/opt/openwam-venv/bin:${PATH} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WANDB_MODE=offline \
    IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake ffmpeg git git-lfs libegl1 libgl1 \
      libglib2.0-0 libgomp1 netcat-openbsd openssh-client openssh-server patch \
      python3 python3-dev python3-pip python3-venv rsync sudo \
    && rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* \
    && (id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu) \
    && printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/npa-runtime \
    && chmod 0440 /etc/sudoers.d/npa-runtime

# OpenWAM's locked CUDA environment requires the Ubuntu 24.04 Python 3.12
# interpreter.  OpenWAM's own LIBERO instructions separately require Python
# 3.10, so provision that exact minor version into an isolated client venv.
RUN python3 -m venv /opt/uv \
    && /opt/uv/bin/python -m pip install --no-cache-dir uv==0.11.15 \
    && UV_CACHE_DIR=/tmp/openwam-uv-cache /opt/uv/bin/uv python install 3.10 \
    && rm -rf /tmp/openwam-uv-cache

RUN git clone --filter=blob:none --no-checkout "${OPENWAM_SOURCE_URL}" /opt/openwam \
    && git -C /opt/openwam checkout --detach "${OPENWAM_SOURCE_REF}" \
    && test "$(git -C /opt/openwam rev-parse HEAD)" = "${OPENWAM_SOURCE_REF}" \
    && test -s /opt/openwam/LICENSE \
    && git -C /opt/openwam lfs uninstall \
    && rm -rf /opt/openwam/.git

# This follows the pinned upstream CUDA-12.8 resolver: its lock remains the
# dependency authority and no checkpoint or benchmark payload is part of a
# layer.  The second environment is deliberately isolated for OpenWAM's own
# documented LIBERO client dependency set.
RUN python3 -m venv /opt/openwam-venv \
    && /opt/openwam-venv/bin/python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && /opt/openwam-venv/bin/python -m pip install --no-cache-dir --no-deps \
      --index-url https://download.pytorch.org/whl/cu128 \
      -r <(grep -E '^(torch|torchvision|torchaudio)==' /opt/openwam/docker/requirements-cu128.txt) \
    && grep -v '^deepspeed==' /opt/openwam/docker/requirements-cu128.txt > /tmp/openwam-no-deepspeed.txt \
    && /opt/openwam-venv/bin/python -m pip install --no-cache-dir --no-build-isolation -r /tmp/openwam-no-deepspeed.txt \
    && DS_BUILD_OPS=0 /opt/openwam-venv/bin/python -m pip install --no-cache-dir --no-build-isolation \
      -r /opt/openwam/docker/requirements-cu128.txt \
    && /opt/openwam-venv/bin/python -m pip install --no-cache-dir --no-deps -e /opt/openwam \
    && /opt/openwam-venv/bin/python -m pip install --no-cache-dir boto3==1.42.91 rerun-sdk==0.38.1 \
    && find /opt/openwam-venv -type f -path '*/imageio_ffmpeg/binaries/ffmpeg*' -delete \
    && /opt/openwam-venv/bin/python -c 'import imageio_ffmpeg; assert imageio_ffmpeg.get_ffmpeg_exe() == "/usr/bin/ffmpeg"' \
    && /opt/openwam-venv/bin/python -m pip check

RUN git clone --filter=blob:none --no-checkout "${LIBERO_SOURCE_URL}" /opt/openwam-libero-source \
    && git -C /opt/openwam-libero-source checkout --detach "${LIBERO_SOURCE_REF}" \
    && test "$(git -C /opt/openwam-libero-source rev-parse HEAD)" = "${LIBERO_SOURCE_REF}" \
    && test -s /opt/openwam-libero-source/LICENSE \
    && git -C /opt/openwam-libero-source apply /opt/openwam/benchmarks/libero/patches/libero-pytorch-load.patch \
    && git -C /opt/openwam-libero-source lfs uninstall \
    && rm -rf /opt/openwam-libero-source/.git \
    && UV_CACHE_DIR=/tmp/openwam-uv-cache /opt/uv/bin/uv venv --seed --python 3.10 /opt/openwam-libero \
    && /opt/openwam-libero/bin/python -m pip install --no-cache-dir --upgrade pip==26.2.1 setuptools==84.0.0 wheel==0.48.0 \
    && /opt/openwam-libero/bin/python -m pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      hydra-core==1.2.0 numpy==1.22.4 wandb==0.13.1 easydict==1.9 \
      transformers==4.21.1 opencv-python==4.6.0.66 robomimic==0.2.0 \
      einops==0.4.1 thop==0.1.1.post2209072238 robosuite==1.4.0 \
      bddl==1.0.1 future==0.18.2 matplotlib==3.5.3 cloudpickle==2.1.0 \
      gym==0.25.2 torch==2.7.1+cpu mujoco==3.3.2 imageio==2.37.4 \
      imageio-ffmpeg==0.6.0 PyYAML==6.0.3 websockets==13.1 \
    && /opt/openwam-libero/bin/python -m pip install --no-cache-dir -r /opt/openwam-libero-source/requirements.txt \
    && /opt/openwam-libero/bin/python -m pip install --no-cache-dir --no-deps -e /opt/openwam-libero-source \
    && printf '%s\n' /opt/openwam-libero-source > "$(/opt/openwam-libero/bin/python -c 'import site; print(site.getsitepackages()[0])')/libero_source.pth" \
    && /opt/openwam-libero/bin/python -m pip check \
    && find /opt/openwam-libero -type f -path '*/imageio_ffmpeg/binaries/ffmpeg*' -delete \
    && /opt/openwam-libero/bin/python -c 'import imageio_ffmpeg, libero, mujoco; assert imageio_ffmpeg.get_ffmpeg_exe() == "/usr/bin/ffmpeg"; assert mujoco.__version__ == "3.3.2"; assert libero is not None' \
    && rm -rf /tmp/openwam-uv-cache

COPY pyproject.toml README.md /opt/npa/
COPY src/npa /opt/npa/src/npa
# Each upstream environment declares imageio-ffmpeg, whose wheel bundles a
# second static executable. Each install layer removes it before that layer is
# committed; the Ubuntu ffmpeg package above is sufficient for the runtime.
RUN /opt/openwam-venv/bin/python -m pip install --no-cache-dir --no-deps /opt/npa \
    && test -z "$(find /opt/openwam-venv /opt/openwam-libero -type f -path '*/imageio_ffmpeg/binaries/ffmpeg*' -print -quit)" \
    && /opt/openwam-venv/bin/python -m npa.workflows.openwam_pipeline --help >/dev/null \
    && /opt/openwam-venv/bin/rerun rrd --help >/dev/null \
    && test -z "$(find /opt/openwam /opt/openwam-venv /opt/openwam-libero /opt/openwam-libero-source -type f \
      \( -iname '*.safetensors' -o -iname '*.ckpt' -o -iname 'pytorch_model*.bin' \) -print -quit)" \
    && rm -rf /root/.cache /tmp/openwam-no-deepspeed.txt

COPY --chmod=0755 docker/workbench/openwam/entrypoint.sh /usr/local/bin/openwam-entrypoint
COPY --chmod=0444 docker/workbench/openwam/REDISTRIBUTION.md /usr/share/doc/npa-openwam/REDISTRIBUTION.md
COPY --chmod=0444 docker/workbench/openwam/THIRD_PARTY_NOTICES.md /usr/share/doc/npa-openwam/THIRD_PARTY_NOTICES.md

RUN install -d -m 0755 -o ubuntu -g ubuntu /workspace /workspace/.cache /workspace/.cache/huggingface \
    && chmod 0755 /usr/share/doc/npa-openwam \
    && chown -R ubuntu:ubuntu /home/ubuntu /workspace /opt/openwam \
    && sudo -u ubuntu -H /opt/openwam-venv/bin/rerun analytics disable >/dev/null 2>&1 \
    && test -r /usr/share/doc/npa-openwam/REDISTRIBUTION.md \
    && test -r /usr/share/doc/npa-openwam/THIRD_PARTY_NOTICES.md \
    && test -s /home/ubuntu/.config/rerun/analytics.json

VOLUME ["/workspace/.cache/huggingface"]
WORKDIR /opt/openwam
USER ubuntu
ENTRYPOINT ["/usr/local/bin/openwam-entrypoint"]
CMD ["python3", "-m", "npa.workflows.openwam_pipeline", "--help"]
