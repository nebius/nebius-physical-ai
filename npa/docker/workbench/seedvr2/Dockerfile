# Apache-2.0 SeedVR2 source/runtime only; public model weights stay runtime-only.
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04@sha256:ae7f650405a3964972dacfa889273bf8e3fbe9709899afd187da01c4cdff3105

ARG NPA_SOURCE_SHA
ARG SOURCE_DATE_EPOCH
ARG SEEDVR2_SOURCE_REF=e4de8c24441a67e1b7df56abea10645059bb1185
ARG SEEDVR2_SOURCE_SHA256=aeaed7c661f4c3804c71542bbf7ce6c45c103e5f62caf917c2d2ab6e8a29dc98
ARG APEX_SOURCE_REF=8a6508aaad6e75a2b939e33f308cd63d745d97f1
ARG APEX_SOURCE_SHA256=59b3e915823a2f41fec7f8cb20b4a558c9658839709f171953c1f6c16a8975f8
ARG FLASH_ATTN_VERSION=2.8.3.post1
ARG FLASH_ATTN_SHA256=55d5103ed846da8b56e0797acf4bde07dee4b1c7e8907fcfc6699c203030c348
ARG FLASH_ATTN_MAX_JOBS=2
ARG FLASH_ATTN_NVCC_THREADS=1
ARG APEX_MAX_JOBS=2

LABEL npa.tool="seedvr2" \
      npa.version="0.1.0-cu130-unbuilt" \
      npa.source.repo="https://github.com/ByteDance-Seed/SeedVR" \
      npa.source.ref="${SEEDVR2_SOURCE_REF}" \
      npa.weights="operator-runtime-fetch" \
      npa.build.flash-attn.max-jobs="${FLASH_ATTN_MAX_JOBS}" \
      npa.build.flash-attn.nvcc-threads="${FLASH_ATTN_NVCC_THREADS}" \
      npa.build.apex.max-jobs="${APEX_MAX_JOBS}" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.opencontainers.image.licenses="Apache-2.0 AND BSD-3-Clause"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_CACHE=/workspace/.cache/huggingface/hub \
    SEEDVR2_SOURCE_ROOT=/opt/seedvr2 \
    SEEDVR2_PYTHON=/opt/seedvr2-venv/bin/python \
    NPA_SEEDVR2_WORK_DIR=/workspace/seedvr2-runs \
    NPA_SKIP_EAGER_IMPORTS=1 \
    NPA_LIGHT_WORKBENCH_TOOL=seedvr2 \
    TORCH_CUDA_ARCH_LIST=9.0 \
    PATH=/opt/npa-venv/bin:/opt/seedvr2-venv/bin:${PATH}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER root

RUN [[ "$NPA_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] \
    && [[ "$SOURCE_DATE_EPOCH" =~ ^[1-9][0-9]*$ ]]

RUN rm -f /etc/apt/sources.list.d/*.list \
    && apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl ffmpeg git libgl1 libglib2.0-0 \
      ninja-build openssh-server python3.12 python3.12-dev python3.12-venv \
      rsync sudo \
    && rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub \
    && rm -rf /opt/nvidia/nsight-compute \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* \
    && (id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu) \
    && printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/npa-runtime \
    && chmod 0440 /etc/sudoers.d/npa-runtime

COPY docker/workbench/seedvr2/requirements.lock /tmp/seedvr2-requirements.lock
RUN python3.12 -m venv /opt/seedvr2-venv \
    && pip install --no-cache-dir --upgrade 'pip==26.2.1' \
    && pip install --no-cache-dir --require-hashes -r /tmp/seedvr2-requirements.lock

ARG FLASH_ATTN_CUDA_ARCHS=90
RUN curl --fail --location \
      "https://files.pythonhosted.org/packages/source/f/flash-attn/flash_attn-${FLASH_ATTN_VERSION}.tar.gz" \
      -o /tmp/flash-attn.tar.gz \
    && echo "${FLASH_ATTN_SHA256}  /tmp/flash-attn.tar.gz" | sha256sum -c - \
    && FLASH_ATTENTION_FORCE_BUILD=TRUE \
      FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS}" \
      MAX_JOBS="${FLASH_ATTN_MAX_JOBS}" \
      NVCC_THREADS="${FLASH_ATTN_NVCC_THREADS}" \
      pip install --no-cache-dir --no-index --no-build-isolation --no-deps \
      /tmp/flash-attn.tar.gz \
    && rm -f /tmp/flash-attn.tar.gz

LABEL npa.build.flash-attn.cuda-archs="${FLASH_ATTN_CUDA_ARCHS}"

RUN curl --fail --location \
      "https://codeload.github.com/NVIDIA/apex/tar.gz/${APEX_SOURCE_REF}" \
      -o /tmp/apex.tar.gz \
    && echo "${APEX_SOURCE_SHA256}  /tmp/apex.tar.gz" | sha256sum -c - \
    && mkdir /tmp/apex \
    && tar -xzf /tmp/apex.tar.gz --strip-components=1 -C /tmp/apex \
    && APEX_CPP_EXT=1 APEX_CUDA_EXT=1 MAX_JOBS="${APEX_MAX_JOBS}" \
      pip install --no-cache-dir --no-index --no-build-isolation --no-deps \
      /tmp/apex \
    && rm -rf /tmp/apex /tmp/apex.tar.gz

RUN pip check && python - <<'PY'
import torch
import torchvision
from apex.normalization import FusedLayerNorm, FusedRMSNorm
from flash_attn import flash_attn_varlen_func
assert torch.version.cuda == "13.0", torch.version.cuda
assert "sm_90" in set(torch._C._cuda_getArchFlags().split())
assert torchvision.__version__.startswith("0.28.0")
assert FusedLayerNorm and FusedRMSNorm and flash_attn_varlen_func
PY

RUN curl --fail --location \
      "https://codeload.github.com/ByteDance-Seed/SeedVR/tar.gz/${SEEDVR2_SOURCE_REF}" \
      -o /tmp/seedvr2.tar.gz \
    && echo "${SEEDVR2_SOURCE_SHA256}  /tmp/seedvr2.tar.gz" | sha256sum -c - \
    && mkdir /opt/seedvr2 \
    && tar -xzf /tmp/seedvr2.tar.gz --strip-components=1 -C /opt/seedvr2 \
    && test -s /opt/seedvr2/LICENSE \
    && printf '%s\n' "${SEEDVR2_SOURCE_REF}" > /opt/seedvr2/NPA_SOURCE_REVISION \
    && rm -rf /opt/seedvr2/.git \
    && test -z "$(find /opt/seedvr2 -type f \
      \( -iname '*.pth' -o -iname '*.pt' -o -iname '*.safetensors' \
         -o -iname '*.ckpt' -o -iname 'pytorch_model*.bin' \) -print -quit)"

COPY pyproject.toml README.md /opt/npa-src/
COPY src/npa /opt/npa-src/src/npa
RUN python3.12 -m venv /opt/npa-venv \
    && /opt/npa-venv/bin/pip install --no-cache-dir 'uvicorn==0.41.0' /opt/npa-src \
    && /opt/npa-venv/bin/npa workbench seedvr2 restore \
      --input-path s3://dry-run/input.mp4 \
      --output-path s3://dry-run/output/ \
      --run-id image-build --dry-run >/dev/null \
    && find /opt/seedvr2-venv -type f \
      -path '*/imageio_ffmpeg/binaries/ffmpeg*' -delete \
    && rm -rf /root/.cache /tmp/apex /tmp/*.tar.gz /tmp/pip-* \
    && test -z "$(find /opt/npa-src /opt/npa-venv \
      \( -iname '*.pth' -o -iname '*.pt' -o -iname '*.safetensors' \
         -o -iname '*.ckpt' -o -iname 'pytorch_model*.bin' \) -print -quit)"

COPY --chmod=0444 docker/workbench/seedvr2/REDISTRIBUTION.md /usr/share/doc/npa-seedvr2/REDISTRIBUTION.md
COPY --chmod=0444 docker/workbench/seedvr2/THIRD_PARTY_NOTICES.md /usr/share/doc/npa-seedvr2/THIRD_PARTY_NOTICES.md
COPY --chmod=0755 docker/workbench/seedvr2/entrypoint.sh /usr/local/bin/seedvr2-entrypoint

RUN install -d -m 0755 -o ubuntu -g ubuntu \
      /workspace /workspace/.cache /workspace/.cache/huggingface \
      /workspace/seedvr2-runs \
    && chown -R ubuntu:ubuntu /workspace

VOLUME ["/workspace/.cache/huggingface"]
WORKDIR /workspace
USER ubuntu
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"]
ENTRYPOINT ["seedvr2-entrypoint"]
CMD ["serve"]
