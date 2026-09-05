# Operator-controlled registry only. Inherits the official vendor runtime;
# this is not a zero-payload image eligible for the NPA public mirror.
ARG BASE_IMAGE=vllm/vllm-omni:cosmos3@sha256:970dee6658ea223f615b2438ce41e47f1d5322225482546e6e6bc5d8134f757c
FROM ${BASE_IMAGE}

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*
COPY docker/workbench/cosmos3-nano-video/requirements.txt /opt/npa/requirements.txt
RUN uv pip install --system --no-cache -r /opt/npa/requirements.txt \
    && useradd --create-home --uid 10001 --shell /bin/bash npa-video \
    && mkdir -p /opt/npa/npa/workbench/cosmos /models /outputs /tmp/ray \
    && chown -R 10001:10001 /models /outputs /tmp/ray
COPY src/npa/workbench/cosmos/nano_video.py \
     src/npa/workbench/cosmos/nano_video_server.py \
     src/npa/workbench/cosmos/nano_video_router.py \
     src/npa/workbench/cosmos/nano_video_stage.py /opt/npa/npa/workbench/cosmos/
COPY src/npa/workbench/cosmos/nano_video_golden.py /opt/npa/npa/workbench/cosmos/nano_video_golden.py

ENV PYTHONPATH=/opt/npa \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/huggingface \
    RAY_USAGE_STATS_ENABLED=0 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    NPA_COSMOS3_MODEL_PATH=/models/Cosmos3-Nano \
    NPA_COSMOS3_VIDEO_OUTPUT_ROOT=/outputs
ARG NPA_SOURCE_COMMIT
LABEL org.opencontainers.image.source="https://github.com/nebius/nebius-physical-ai" \
      org.opencontainers.image.revision="${NPA_SOURCE_COMMIT}" \
      npa.tool="cosmos3-nano-video" \
      npa.weights="runtime-prestage-only" \
      npa.runtime.license="inherited-from-upstream-vllm-omni-cosmos3" \
      npa.redistribution="restricted"
USER 10001:10001
WORKDIR /opt/npa
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s \
  CMD python -c 'import ray; ray.init(address="auto", logging_level="ERROR"); ray.shutdown()'
ENTRYPOINT []
CMD ["serve", "run", "npa.workbench.cosmos.nano_video_server:app"]
