# Native Ray Serve derivative of the accepted npa-cosmos3 framework runtime.
# The digest-pinned parent contains OpenMDW-1.1 cosmos-framework at commit
# 5e67049cd94acb667786f1e6dd0dab821cb90c97 and its frozen CUDA 13 inference
# environment. No model or guardrail weights are present in either image.
ARG BASE_IMAGE=ghcr.io/nebius/nebius-physical-ai/npa-cosmos3@sha256:0eb459769eba9942b5b9004c4df68721f2c3e5490a6614e85ced67e8a6f9cd5d
FROM ${BASE_IMAGE}

ARG NPA_SOURCE_SHA
ARG COSMOS3_RAY_SERVE_VERSION=ray1-cu130
ARG COSMOS3_REF=5e67049cd94acb667786f1e6dd0dab821cb90c97
ARG COSMOS3_RAY_VERSION=2.46.0
# Exact Ubuntu Noble security build that fixes the parent image's known
# linux-libc-dev critical findings. This package is compile-time headers only;
# the workload still uses the host's managed kernel and NVIDIA driver.
ARG LINUX_LIBC_DEV_VERSION=6.8.0-138.138

LABEL npa.tool="cosmos3-ray-serve" \
      npa.version="${COSMOS3_RAY_SERVE_VERSION}" \
      npa.cosmos3.ref="${COSMOS3_REF}" \
      npa.cosmos3.ray.version="${COSMOS3_RAY_VERSION}" \
      npa.serving.backend="cosmos-framework-native-ray-serve" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.opencontainers.image.source="https://github.com/nebius/nebius-physical-ai" \
      org.opencontainers.image.licenses="OpenMDW-1.1"

ENV NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    NPA_COSMOS3_FRAMEWORK_REVISION=${COSMOS3_REF} \
    NPA_COSMOS3_FRAMEWORK_PYTHON=/opt/cosmos3/cosmos-framework/.venv/bin/python \
    NPA_COSMOS3_RAY_MODEL=Cosmos3-Nano \
    NPA_COSMOS3_RAY_CHECKPOINT=Cosmos3-Nano \
    NPA_COSMOS3_RAY_OUTPUT_DIR=/outputs \
    NPA_COSMOS3_RAY_HOST=0.0.0.0 \
    NPA_COSMOS3_RAY_PORT=8000 \
    NPA_COSMOS3_RAY_GUARDRAILS=true \
    NPA_COSMOS3_RAY_PARALLELISM_PRESET=throughput \
    NPA_COSMOS3_RAY_WORLD_SIZE=1 \
    NPA_COSMOS3_RAY_MAX_BATCH_SIZE=4 \
    NPA_COSMOS3_RAY_BATCH_WAIT_TIMEOUT_S=0.05 \
    PYTHONPATH=/opt/npa/src

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       "linux-libc-dev=${LINUX_LIBC_DEV_VERSION}" \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /outputs /tmp/ray \
    && chown ubuntu:ubuntu /outputs /tmp/ray
USER ubuntu

# Replace only NPA's thin control/client layer. The model implementation and
# dependency lock remain the exact pinned upstream bytes in the parent. The
# parent generation image intentionally omits upstream's `serve` extra, so sync
# that extra from the already-pinned upstream uv.lock. `--inexact` retains the
# parent's two explicitly pinned generate-only dependencies.
COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu workflows /opt/npa/workflows
COPY --chown=ubuntu:ubuntu docker/workbench/cosmos3-ray-serve/verify_env.py /opt/cosmos3-ray-serve/verify_env.py
COPY --chown=ubuntu:ubuntu --chmod=755 docker/workbench/cosmos3-ray-serve/entrypoint.sh /usr/local/bin/cosmos3-ray-serve-entrypoint
COPY --chown=ubuntu:ubuntu --chmod=755 docker/workbench/cosmos3-ray-serve/smoke_functional.sh /opt/cosmos3-ray-serve/smoke_functional.sh

RUN cd /opt/cosmos3/cosmos-framework \
    && UV_CACHE_DIR=/tmp/uv-cache uv sync --frozen --inexact \
       --extra guardrail --extra serve --group cu130 \
    && test "$(.venv/bin/python -c 'import ray; print(ray.__version__)')" = "${COSMOS3_RAY_VERSION}" \
    && rm -rf /tmp/uv-cache \
    && uv pip install --python /opt/npa/.venv/bin/python --no-deps -e /opt/npa \
    && /opt/cosmos3/cosmos-framework/.venv/bin/python /opt/cosmos3-ray-serve/verify_env.py \
    && test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40 \
    && test -w /outputs \
    && test -w /tmp/ray \
    && test -z "$(find /opt/cosmos3 /opt/npa /outputs -type f \
         \( -name '*.safetensors' -o -name '*.ckpt' -o -name '*.pth' -o -name '*.pt' \) \
         -size +50M -print -quit 2>/dev/null)"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30m --retries=3 \
  CMD /opt/npa/.venv/bin/python -c "import os,urllib.request; r=urllib.request.Request('http://127.0.0.1:'+os.environ.get('NPA_COSMOS3_RAY_PORT','8000')+'/ready',headers={'Authorization':'Bearer '+os.environ['NPA_COSMOS3_RAY_TOKEN']}); urllib.request.urlopen(r,timeout=8)" || exit 1

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/cosmos3-ray-serve-entrypoint"]
CMD []
