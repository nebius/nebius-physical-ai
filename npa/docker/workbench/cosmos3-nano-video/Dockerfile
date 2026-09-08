# Public bootstrap parent contains no CUDA, vLLM, or model payload. The pinned
# serving and Ray closures are fetched only into the operator's runtime cache.
ARG BASE_IMAGE=ghcr.io/nebius/nebius-physical-ai/npa-cosmos3-serving@sha256:3342bbe44bd1c00ebf05ab4c9d7286058a94bb5ce90b49b164b23604d3acf180
FROM ${BASE_IMAGE}

ARG VLLM_OMNI_REVISION=eb11446b7f2e30ca582f8aff3afe12e9a2e66f6c
ARG VLLM_OMNI_SOURCE_SHA256=a0380c02b403f3c71ba4360fd23fc2461ea8a84c28d5543b8ac4c08a31a97a3f
ARG COSMOS3_CLOSURE_SHA256=c13b5b13868fa45bb125e0f7f32ec718bcb10e554ce34b85af944dbd4bafc6aa
ARG DEBIAN_SNAPSHOT=20260907T000000Z
ENV NPA_COSMOS3_CLOSURE_SHA256=${COSMOS3_CLOSURE_SHA256} \
    NPA_VLLM_OMNI_REVISION=${VLLM_OMNI_REVISION} \
    NPA_VLLM_OMNI_SOURCE_SHA256=${VLLM_OMNI_SOURCE_SHA256}

USER root
# The accepted parent's older runtime recipe is replaced by these reviewed files.
COPY --chmod=0644 docker/workbench/sonic/packaging-requirements.txt /opt/npa-cosmos3-serving/packaging-requirements.txt
COPY --chmod=0644 docker/workbench/cosmos3-serving/requirements.lock /opt/npa-cosmos3-serving/requirements.lock
COPY --chmod=0755 docker/workbench/cosmos3-serving/runtime_bootstrap.sh \
     docker/workbench/cosmos3-serving/entrypoint.sh \
     docker/workbench/cosmos3-serving/smoke_serving.sh /opt/npa-cosmos3-serving/
COPY --chmod=0644 docker/workbench/cosmos3-serving/access_preflight.py \
     docker/workbench/cosmos3-serving/hf_snapshot_pin.py \
     docker/workbench/cosmos3-serving/prepare_guardrail_runtime.py \
     docker/workbench/cosmos3-serving/backport_setuptools_manifest.py \
     docker/workbench/cosmos3-serving/verify_env.py /opt/npa-cosmos3-serving/
RUN cp /opt/npa-cosmos3-serving/prepare_guardrail_runtime.py \
       /opt/npa-cosmos3-serving/prepare_guardrail_payload.py \
    && sed -ri "s|(https?://snapshot.debian.org/archive/debian(-security)?/)[0-9TZ]+|\\1${DEBIAN_SNAPSHOT}|g" \
       /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y \
    && apt-get install -y --no-install-recommends wget util-linux \
    && python -m pip install --no-cache-dir --require-hashes --only-binary=:all: \
       -r /opt/npa-cosmos3-serving/packaging-requirements.txt \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /bin/bash npa-video \
    && mkdir -p /opt/npa/npa/workbench/cosmos /models /outputs /tmp/ray \
    && chown -R 10001:10001 /models /outputs /tmp/ray /opt/npa-cosmos3-serving/runtime
COPY --chmod=0644 docker/workbench/cosmos3-nano-video/requirements.lock /opt/npa/nano-requirements.lock
COPY --chmod=0755 docker/workbench/cosmos3-nano-video/runtime_bootstrap.sh /usr/local/bin/npa-cosmos3-nano-bootstrap
COPY --chmod=0755 docker/workbench/cosmos3-nano-video/ray.sh /usr/local/bin/ray
COPY --chmod=0644 docker/workbench/cosmos3-super-benchmark/prepare_guardrail_runtime.py /opt/npa-cosmos3-serving/prepare_guardrail_runtime.py
COPY --chmod=0644 docker/workbench/cosmos3-super-benchmark/bootstrap-source-sha256s.txt /opt/npa-cosmos3-serving/bootstrap-source-sha256s.txt
COPY --chmod=0644 docker/workbench/cosmos3-nano-video/THIRD_PARTY_NOTICES.md /opt/npa/THIRD_PARTY_NOTICES.md
COPY --chmod=0644 docker/workbench/cosmos3-super-benchmark/LICENSE-APACHE-2.0 /opt/npa/LICENSE-APACHE-2.0
COPY src/npa/workbench/cosmos/nano_video.py \
     src/npa/workbench/cosmos/nano_video_augment.py \
     src/npa/workbench/cosmos/nano_video_server.py \
     src/npa/workbench/cosmos/nano_video_engine.py \
     src/npa/workbench/cosmos/nano_video_auth.py \
     src/npa/workbench/cosmos/nano_video_router.py \
     src/npa/workbench/cosmos/nano_video_stage.py /opt/npa/npa/workbench/cosmos/
COPY src/npa/workbench/cosmos/nano_video_golden.py /opt/npa/npa/workbench/cosmos/nano_video_golden.py

ENV PYTHONPATH=/opt/npa \
    HOME=/home/npa-video \
    NPA_COSMOS3_NANO_CLOSURE_SHA256=31a0a9da0fe55b7098fc02e275027c465b1e1a24f29c029fb211c7548a685bad \
    NPA_COSMOS3_SERVE_GUARDRAILS=off \
    NPA_COSMOS3_SERVE_MODEL=nvidia/Cosmos3-Nano \
    NPA_COSMOS3_SERVE_MODEL_REVISION=7a312c868bcce8e40b3eb40861300a9d0ba3fde1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/huggingface \
    RAY_USAGE_STATS_ENABLED=0 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    NPA_COSMOS3_MODEL_PATH=/models/Cosmos3-Nano \
    NPA_COSMOS3_VIDEO_OUTPUT_ROOT=/outputs
ARG NPA_SOURCE_SHA
LABEL org.opencontainers.image.source="https://github.com/nebius/nebius-physical-ai" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      npa.tool="cosmos3-nano-video" \
      npa.weights="runtime-prestage-only" \
      npa.runtime="operator-fetch" \
      npa.vllm_omni.revision="${VLLM_OMNI_REVISION}" \
      npa.version="0.1.0" \
      npa.redistribution="public"
USER 10001:10001
RUN test -w /opt/npa-cosmos3-serving/runtime \
    && sha256sum -c /opt/npa-cosmos3-serving/bootstrap-source-sha256s.txt \
    && python /opt/npa-cosmos3-serving/verify_env.py
WORKDIR /opt/npa
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s \
  CMD python -c 'import os,http.client; c=http.client.HTTPConnection("127.0.0.1",8000,timeout=8); c.request("GET","/health",headers={"Authorization":"Bearer "+os.environ["NPA_COSMOS3_VIDEO_TOKEN"]}); r=c.getresponse(); assert r.status == 200'
ENTRYPOINT ["/usr/local/bin/npa-cosmos3-nano-bootstrap"]
CMD ["python", "-m", "npa.workbench.cosmos.nano_video_server", "--serve"]
