# Apache-2.0 Alpamayo 2 source/runtime only. OpenMDW model weights and the
# restricted PhysicalAI-AV dataset are fetched into an operator-owned cache.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04@sha256:24c8e3581ea6330038b0d374920721983312627f8adbfcf390bdb4b399d280ed

ARG ALPAMAYO2_SOURCE_URL=https://github.com/NVlabs/alpamayo2.git
ARG ALPAMAYO2_SOURCE_REF=beb2977d9a7e9d66837d4a3ad5144ff59de37519
ARG NPA_SOURCE_COMMIT=unknown

LABEL npa.tool="alpamayo2-super" \
      npa.version="0.1.0-cu128" \
      npa.source.repo="${ALPAMAYO2_SOURCE_URL}" \
      npa.source.ref="${ALPAMAYO2_SOURCE_REF}" \
      npa.weights="operator-runtime-fetch" \
      npa.dataset="operator-runtime-fetch" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/workspace/.cache/huggingface \
    MPLCONFIGDIR=/workspace/.cache/matplotlib \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0" \
    UV_PROJECT_ENVIRONMENT=/opt/alpamayo2/.venv \
    UV_CACHE_DIR=/tmp/uv-cache \
    PATH=/opt/npa-venv/bin:/opt/alpamayo2/.venv/bin:${PATH}

SHELL ["/bin/bash", "-lc"]
USER root

RUN rm -f /etc/apt/sources.list.d/*.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates ffmpeg git openssh-server python3.12 python3.12-dev \
      python3.12-venv rsync sudo \
    && rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* \
    && (id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu) \
    && printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/npa-runtime \
    && chmod 0440 /etc/sudoers.d/npa-runtime

COPY --chmod=0444 docker/workbench/alpamayo2-super/dataset-revision.patch /tmp/dataset-revision.patch
RUN git clone --filter=blob:none --no-checkout "${ALPAMAYO2_SOURCE_URL}" /opt/alpamayo2 \
    && git -C /opt/alpamayo2 checkout --detach "${ALPAMAYO2_SOURCE_REF}" \
    && test "$(git -C /opt/alpamayo2 rev-parse HEAD)" = "${ALPAMAYO2_SOURCE_REF}" \
    && test -s /opt/alpamayo2/LICENSE \
    && git -C /opt/alpamayo2 apply --unidiff-zero /tmp/dataset-revision.patch \
    && grep -Fq 'Modified by Nebius' /opt/alpamayo2/src/alpamayo2_super/load_physical_aiavdataset.py \
    && find /opt/alpamayo2 -type f -name '*.parquet' -delete \
    && rm -rf /opt/alpamayo2/.git

# Upstream's lock is authoritative. uv's no-build-isolation declaration lets
# flash-attn compile against the locked Torch 2.8/CUDA 12.8 environment for the
# H100, B200, and RTX PRO 6000 architecture paths selected above.
RUN python3.12 -m venv /opt/uv \
    && /opt/uv/bin/pip install --no-cache-dir 'uv==0.9.7' \
    && cd /opt/alpamayo2 \
    && /opt/uv/bin/uv sync --locked --no-dev \
    && /opt/alpamayo2/.venv/bin/python - <<'PY'
import torch
flags = set(torch._C._cuda_getArchFlags().split())
assert {'sm_90', 'sm_100', 'sm_120'} <= flags, flags
PY

COPY pyproject.toml README.md /opt/npa-src/
COPY src/npa /opt/npa-src/src/npa
COPY workflows/workbench/npa-workflows/sim2real.yaml /opt/npa-src/workflows/workbench/npa-workflows/sim2real.yaml
COPY workflows/workbench/npa-workflows/physical-ai-data-factory.yaml /opt/npa-src/workflows/workbench/npa-workflows/physical-ai-data-factory.yaml
RUN python3.12 -m venv /opt/npa-venv \
    && /opt/npa-venv/bin/pip install --no-cache-dir 'uvicorn==0.41.0' /opt/npa-src \
    && rm -rf /root/.cache /tmp/uv-cache /tmp/pip-* \
    && test -z "$(find /opt/alpamayo2 /opt/npa-src /opt/npa-venv \
      \( -iname '*.safetensors' -o -iname '*.ckpt' -o -iname 'pytorch_model*.bin' \) \
      -print -quit)"

COPY --chmod=0755 docker/workbench/alpamayo2-super/entrypoint.sh /usr/local/bin/alpamayo2-entrypoint
COPY --chmod=0444 docker/workbench/alpamayo2-super/REDISTRIBUTION.md /usr/share/doc/npa-alpamayo2-super/REDISTRIBUTION.md
COPY --chmod=0444 docker/workbench/alpamayo2-super/THIRD_PARTY_NOTICES.md /usr/share/doc/npa-alpamayo2-super/THIRD_PARTY_NOTICES.md

RUN install -d -m 0755 -o ubuntu -g ubuntu /workspace /workspace/.cache \
      /workspace/.cache/huggingface /workspace/.cache/matplotlib \
    && chown -R ubuntu:ubuntu /workspace \
    && /opt/npa-venv/bin/npa workbench alpamayo2-super infer \
      --output-path /tmp/alpamayo2-plan --dry-run >/dev/null

VOLUME ["/workspace/.cache/huggingface"]
WORKDIR /workspace
USER ubuntu
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["/opt/npa-venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"]
ENTRYPOINT ["alpamayo2-entrypoint"]
CMD ["serve"]
