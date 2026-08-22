# Zero-payload Cosmos3 serving bootstrap. Restricted/NVIDIA runtime bytes and
# all gated models are delivered directly to the operator only at run time.
ARG BASE_IMAGE=python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
FROM ${BASE_IMAGE}

ARG COSMOS3_SERVING_VERSION=0.2.0
ARG VLLM_OMNI_REVISION=a4ea67a21b20054dacc6e83952f9bd407e8ee4e7
ARG VLLM_OMNI_SOURCE_SHA256=2a4ca4d3d83417a88717767fcdfdc5cb214200c6957d26d70625f17f58954800
ARG COSMOS3_CLOSURE_SHA256=9d9edf4dd685c329a36bde45ed05bf5e0d51a3d78cf764d91bedb031e1a94694
ARG COSMOS3_MODEL_REVISION=e0262be9d8f7586bc24c069a2aed2b665bdff266
ARG COSMOS3_GUARDRAIL_REVISION=cf03c0395fac8c4de386c0bdab12cc4fc8d66362
ARG DEBIAN_SNAPSHOT=20260817T000000Z

LABEL org.opencontainers.image.source="https://github.com/nebius/nebius-physical-ai" \
      org.opencontainers.image.licenses="Apache-2.0" \
      npa.tool="cosmos3-serving" \
      npa.version="${COSMOS3_SERVING_VERSION}" \
      npa.vllm_omni.revision="${VLLM_OMNI_REVISION}" \
      npa.runtime="operator-fetch" \
      npa.weights="runtime-fetch-only"

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/npa-cosmos3-serving/hf-cache \
    NPA_COSMOS3_CLOSURE_SHA256=${COSMOS3_CLOSURE_SHA256} \
    NPA_VLLM_OMNI_REVISION=${VLLM_OMNI_REVISION} \
    NPA_VLLM_OMNI_SOURCE_SHA256=${VLLM_OMNI_SOURCE_SHA256} \
    NPA_COSMOS3_SERVE_MODEL=nvidia/Cosmos3-Super \
    NPA_COSMOS3_SERVE_MODEL_REVISION=${COSMOS3_MODEL_REVISION} \
    NPA_COSMOS3_SERVE_GUARDRAIL_MODEL=nvidia/Cosmos-1.0-Guardrail \
    NPA_COSMOS3_SERVE_GUARDRAIL_REVISION=${COSMOS3_GUARDRAIL_REVISION} \
    NPA_COSMOS3_SERVE_HOST=0.0.0.0 \
    NPA_COSMOS3_SERVE_PORT=8000 \
    NPA_COSMOS3_SERVE_GUARDRAILS=on \
    NPA_COSMOS3_SERVE_INIT_TIMEOUT=1800 \
    NPA_COSMOS3_SERVE_GPUS=8

USER root
RUN sed -ri \
      -e "s|https?://deb.debian.org/debian-security|https://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}|g" \
      -e "s|https?://deb.debian.org/debian|https://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}|g" \
      /etc/apt/sources.list.d/debian.sources \
    && printf 'Acquire::Check-Valid-Until "false";\n' >/etc/apt/apt.conf.d/99snapshot \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       ca-certificates curl ffmpeg gcc g++ libgl1 libglib2.0-0 libgnutls30 libssl3 openssl \
    && rm -rf /var/lib/apt/lists/*

COPY --chmod=0644 docker/workbench/cosmos3-serving/requirements.lock /opt/npa-cosmos3-serving/requirements.lock
COPY docker/workbench/cosmos3-serving/runtime_bootstrap.sh /opt/npa-cosmos3-serving/runtime_bootstrap.sh
COPY docker/workbench/cosmos3-serving/entrypoint.sh /opt/npa-cosmos3-serving/entrypoint.sh
COPY docker/workbench/cosmos3-serving/smoke_serving.sh /opt/npa-cosmos3-serving/smoke_serving.sh
COPY --chmod=0644 docker/workbench/cosmos3-serving/access_preflight.py /opt/npa-cosmos3-serving/access_preflight.py
COPY --chmod=0644 docker/workbench/cosmos3-serving/hf_snapshot_pin.py /opt/npa-cosmos3-serving/hf_snapshot_pin.py
COPY --chmod=0644 docker/workbench/cosmos3-serving/prepare_guardrail_runtime.py /opt/npa-cosmos3-serving/prepare_guardrail_runtime.py
COPY --chmod=0644 docker/workbench/cosmos3-serving/verify_env.py /opt/npa-cosmos3-serving/verify_env.py

RUN useradd -m -s /bin/bash -u 1000 ubuntu \
    && mkdir -p /opt/npa-cosmos3-serving/hf-cache /opt/npa-cosmos3-serving/runtime \
    && chmod 0755 /opt/npa-cosmos3-serving \
    && chmod 0755 /opt/npa-cosmos3-serving/*.sh \
    && chmod 0644 /opt/npa-cosmos3-serving/*.py /opt/npa-cosmos3-serving/requirements.lock \
    && chown -R ubuntu:ubuntu /opt/npa-cosmos3-serving /home/ubuntu \
    && test "$(stat -c '%a' /opt/npa-cosmos3-serving/verify_env.py)" = 644 \
    && su -s /bin/sh -c 'test -r /opt/npa-cosmos3-serving/verify_env.py' ubuntu

USER ubuntu
ENV HOME=/home/ubuntu
RUN python /opt/npa-cosmos3-serving/verify_env.py

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=1800s --retries=3 \
  CMD /opt/npa-cosmos3-serving/runtime/venv/bin/python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('NPA_COSMOS3_SERVE_PORT','8000')+'/v1/models', timeout=8)" || exit 1
ENTRYPOINT ["/opt/npa-cosmos3-serving/runtime_bootstrap.sh", "/opt/npa-cosmos3-serving/entrypoint.sh"]
