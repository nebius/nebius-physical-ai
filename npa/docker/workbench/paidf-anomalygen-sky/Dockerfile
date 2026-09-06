# syntax=docker/dockerfile:1
# Operator-built PAIDF AnomalyGen runtime. It fetches the exact Apache-2.0
# component source, compiles upstream's CUDA wheels, and never embeds weights,
# input data, credentials, or EULA state. The image remains restricted until a
# separate redistribution review approves the complete resolved dependency set.

ARG SOURCE_IMAGE=alpine/git:2.49.1@sha256:c0280cf9572316299b08544065d3bf35db65043d5e3963982ec50647d2746e26
ARG CUDA_DEVEL_IMAGE=nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04@sha256:6435dc5a825b0095648d87a3c91240fd7788a85fafaf215739544d389ab74366
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:13.2.1-base-ubuntu24.04@sha256:7be56e69d8ae7c3648b8fca009fa35980ecd9c6eefaafbff3ee8c224e0043eb5

FROM ${SOURCE_IMAGE} AS source
ARG PAIDF_ANOMALYGEN_REVISION=dbaf7d7d9003f048230f9026da5969e9e5931785
RUN git init -q /src \
    && git -C /src remote add origin https://github.com/NVIDIA/paidf-anomalygen.git \
    && git -C /src fetch -q --depth 1 origin "${PAIDF_ANOMALYGEN_REVISION}" \
    && git -C /src checkout -q --detach FETCH_HEAD \
    && test "$(git -C /src rev-parse HEAD)" = "${PAIDF_ANOMALYGEN_REVISION}" \
    && rm -rf /src/.git

FROM ${CUDA_DEVEL_IMAGE} AS wheelbuilder
ARG TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0 10.0 12.0"
ARG MAX_JOBS=16
ARG NVCC_THREADS=1
ENV DEBIAN_FRONTEND=noninteractive \
    MAX_JOBS=${MAX_JOBS} \
    NVCC_THREADS=${NVCC_THREADS} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
COPY --from=ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 /uv /uvx /usr/local/bin/
COPY --from=source /src/docker/build_wheels.sh /tmp/build_wheels.sh
RUN bash /tmp/build_wheels.sh --in-container

FROM ${CUDA_RUNTIME_IMAGE} AS runtime
ARG NPA_SOURCE_SHA
ARG PYTHON_VERSION=3.13.15
ARG PY_VER=3.13
ARG TORCH_VER=2.13.0
ARG TORCHVISION_VER=0.28.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132
ARG PAIDF_ANOMALYGEN_REVISION=dbaf7d7d9003f048230f9026da5969e9e5931785

LABEL npa.tool="paidf-anomalygen-sky" \
      npa.redistribution="operator-private-only" \
      npa.paidf-anomalygen.ref="${PAIDF_ANOMALYGEN_REVISION}" \
      org.opencontainers.image.source="https://github.com/nebius/nebius-physical-ai" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}"

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/usr/local/bin:/opt/venv/bin:${PATH} \
    UV_PYTHON=/opt/venv/bin/python \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    PIP_NO_CACHE_DIR=1 \
    SAM2_BUILD_ALLOW_ERRORS=1 \
    PYTHONPATH=/workspace/paidf-anomalygen \
    HF_HOME=/tmp/paidf-models

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         ca-certificates curl git gcc g++ jq libgomp1 libgl1 libglib2.0-0 \
         netcat-openbsd openssh-client openssh-server rsync sudo \
    && rm -f /etc/ssh/ssh_host_* \
    && (id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu) \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 /uv /uvx /usr/local/bin/
RUN uv venv --python "${PYTHON_VERSION}" /opt/venv \
    && uv pip install pip \
    && uv pip install torch=="${TORCH_VER}" torchvision=="${TORCHVISION_VER}" \
         --index-url "${TORCH_INDEX_URL}"

RUN --mount=type=bind,from=wheelbuilder,source=/wheels,target=/wheels \
    uv pip install /wheels/*.whl \
    && mkdir -p /tmp/cvwheel \
    && cp /wheels/opencv_python_headless-*.whl /tmp/cvwheel/

COPY --from=source /src/requirements.txt /tmp/requirements.txt
COPY --from=source /src/requirements-nodeps.txt /tmp/requirements-nodeps.txt
# Replace upstream NLTK 3.10.0 with its hash-pinned CVE-2026-79675 fix.
# Version 3.10.3 keeps the same required dependency set.
# Replace W&B in this same layer: 0.28.2 updates wandb-core's x/crypto
# to fix CVE-2026-56854, without retaining the old binary in layer history.
RUN uv pip install -r /tmp/requirements.txt \
    && (uv pip uninstall av opencv-python 2>/dev/null || true) \
    && uv pip install --no-deps --force-reinstall /tmp/cvwheel/*.whl \
    && uv pip install -r /tmp/requirements-nodeps.txt --no-deps \
    && uv pip install --no-deps \
         'nltk @ https://files.pythonhosted.org/packages/b6/6d/ebd2af4640b12168fdf0cb74b6118df2f32a2f62ec7e0c06fbfd80706639/nltk-3.10.3-py3-none-any.whl#sha256=ff9598a8e20518ee0d557745890cc4435b9578489e2dcbc69c4f81fa060caf7c' \
    && uv pip install --no-deps --force-reinstall \
         'wandb @ https://files.pythonhosted.org/packages/3c/7f/f3e23a2ff01848d7a5adab9f307992b92ef99c5e6bff2bd89cd46d651e3a/wandb-0.28.2-py3-none-manylinux_2_28_x86_64.whl#sha256=1db698d107871c66b2dcbb0cf4dc2af1ddb159ba94e957e890158ec60ab2de54' \
    && python -c 'import importlib.metadata as m; assert m.version("nltk") == "3.10.3"' \
    && rm -rf /tmp/cvwheel /tmp/requirements.txt /tmp/requirements-nodeps.txt \
    && python -c 'import cv2; cv2.connectedComponentsWithStats' \
    && python -c 'import cosmos_framework.model.generator.mot.unified_mot'

ENV NV_SITE=/opt/venv/lib/python${PY_VER}/site-packages/nvidia \
    CUDA_COMPAT_PATH=/usr/local/cuda/compat
ENV LD_LIBRARY_PATH=${CUDA_COMPAT_PATH}:${NV_SITE}/cudnn/lib:${NV_SITE}/nccl/lib:${NV_SITE}/cusparselt/lib:${LD_LIBRARY_PATH}
RUN test "$(dpkg-query -W -f='${Version}' cuda-compat-13-2)" = "595.58.03-1ubuntu1" \
    && test -f ${CUDA_COMPAT_PATH}/libcuda.so.595.58.03 \
    && test -f ${CUDA_COMPAT_PATH}/libnvidia-ptxjitcompiler.so.595.58.03 \
    && { for directory in ${NV_SITE}/*/lib; do echo "${directory}"; done; } \
         > /etc/ld.so.conf.d/cosmos.conf \
    && ldconfig

COPY --from=source --chown=ubuntu:ubuntu /src /workspace/paidf-anomalygen
COPY paidf-anomalygen-sky/patch_wandb_run_id.py /usr/local/lib/npa/patch_wandb_run_id.py
WORKDIR /workspace/paidf-anomalygen
# Upstream holds W&B at 0.28.1 for a removed alias. Patch exactly its two
# framework callers to the identical generator, retaining the NVIDIA header.
RUN python /usr/local/lib/npa/patch_wandb_run_id.py \
    && uv pip install -e assets/lerobot_stub --no-deps \
    && uv pip install -e . --no-deps \
    && test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40 \
    && test "$(python -c 'from pathlib import Path; print(Path("VERSION").read_text().strip())')" = "1.1.0" \
    && test -z "$(find /workspace/paidf-anomalygen /tmp/paidf-models -type f \
         \( -name '*.safetensors' -o -name '*.ckpt' -o -name '*.pth' -o -name '*.pt' \) \
         -size +50M -print -quit)"

# Keep NPA's installation separate from upstream's compiled CUDA environment.
# Both environments use the same pinned Python, with independent site-packages.
RUN uv venv --python "${PYTHON_VERSION}" /opt/npa-venv \
    && uv pip install --python /opt/npa-venv/bin/python pip \
    && chown -R ubuntu:ubuntu /opt/npa-venv \
    && printf '%s\n' 'export PATH="/opt/npa-venv/bin:$PATH"' \
         > /etc/profile.d/npa-runtime-python.sh \
    && install -d -m 0755 /run/sshd \
    && sed -i '/^  start)$/a\    ssh-keygen -A' /etc/init.d/ssh \
    && grep -q 'ssh-keygen -A' /etc/init.d/ssh \
    && rm -f /etc/ssh/ssh_host_* \
    && printf '%s\n' 'PasswordAuthentication no' 'PermitRootLogin no' \
         > /etc/ssh/sshd_config.d/99-npa-container.conf \
    && printf 'ubuntu ALL=(ALL) NOPASSWD:ALL\n' \
         > /etc/sudoers.d/99-npa-runtime-user \
    && chmod 0440 /etc/sudoers.d/99-npa-runtime-user \
    && install -d -m 0755 -o ubuntu -g ubuntu /tmp/paidf-models /workspace

COPY paidf-anomalygen-sky/entrypoint.sh /usr/local/bin/npa-sky-entrypoint
RUN chmod 0755 /usr/local/bin/npa-sky-entrypoint

USER ubuntu
ENV HOME=/home/ubuntu \
    PATH=/opt/npa-venv/bin:${PATH}
ENTRYPOINT ["/usr/local/bin/npa-sky-entrypoint"]
CMD ["/bin/bash"]
