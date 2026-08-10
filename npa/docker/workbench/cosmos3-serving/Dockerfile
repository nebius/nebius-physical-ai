# npa-cosmos3-serving: nvidia/Cosmos3-Super served as an OpenAI-compatible
# endpoint, on one 8-GPU node.
#
# This is the serving sibling of npa-cosmos3. That image runs cosmos-framework
# as a batch job and pays model load on every invocation; for the 64B Super
# checkpoint that load is measured in minutes. This image loads once and serves
# many requests against resident weights.
#
# WEIGHTS ARE NEVER BAKED, the same rule npa-cosmos3 follows. The base image is
# NVIDIA's own vLLM-Omni serving runtime, pinned by digest, and it carries no
# checkpoint bytes. nvidia/Cosmos3-Super, the guardrail models, and everything
# else gated download AT RUN TIME with the operator's own Hugging Face token,
# under that operator's own license acceptance. The build gate below asserts
# that no weight file landed in a layer, which is what keeps this image
# redistributable (packaging-contract.yaml redistribution: public).
#
# Pinned by digest, not by tag: `vllm/vllm-omni:cosmos3` is a moving tag, and
# every measured claim in docs/workbench/cosmos3-super-serving.md was taken
# against this exact digest.
ARG BASE_IMAGE=vllm/vllm-omni@sha256:6d2630c7d637b699557573f2c3fee8df5d4d0cd718977aa22549ed6a6ef30587
FROM ${BASE_IMAGE}

ARG COSMOS3_SERVING_VERSION=0.1.0

LABEL npa.tool="cosmos3-serving" \
      npa.version="${COSMOS3_SERVING_VERSION}"

USER root

# HF_HUB_DISABLE_XET is set here, and that is a deliberate departure from
# `npa workbench cosmos3 generate`, which only warns. The difference is who
# controls the pins. That path runs in whatever environment the operator built,
# so setting the variable for them would silently disable a faster download path
# on environments that do not need it. Here the pins are baked into the base
# image this Dockerfile chose: hf-xet 1.5.1 with huggingface_hub 1.23.0, the
# exact pair huggingface/xet-core#895 reproduces on. On that pair the gated
# guardrail download dies with "Unable to parse string as hex hash value"
# several minutes into startup. verify_env.py re-checks the installed pair at
# build time and fails the build once a base bump moves off it, so this line
# cannot outlive the defect it works around.
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_XET=1 \
    HF_HOME=/opt/npa-cosmos3-serving/hf-cache \
    NPA_COSMOS3_SERVE_MODEL=nvidia/Cosmos3-Super \
    NPA_COSMOS3_SERVE_HOST=0.0.0.0 \
    NPA_COSMOS3_SERVE_PORT=8000 \
    NPA_COSMOS3_SERVE_GUARDRAILS=on \
    NPA_COSMOS3_SERVE_INIT_TIMEOUT=1800 \
    NPA_COSMOS3_SERVE_GPUS=8

COPY docker/workbench/cosmos3-serving/entrypoint.sh /opt/npa-cosmos3-serving/entrypoint.sh
COPY docker/workbench/cosmos3-serving/verify_env.py /opt/npa-cosmos3-serving/verify_env.py

# The runtime user owns its home and its cache before the image drops to it. The
# cache directory is a mount point in practice; it exists here so a run without a
# mount still starts instead of failing on a missing path.
RUN id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu \
    && mkdir -p /opt/npa-cosmos3-serving/hf-cache /home/ubuntu \
    && chmod 0755 /opt/npa-cosmos3-serving/entrypoint.sh \
    && chown -R ubuntu:ubuntu /opt/npa-cosmos3-serving /home/ubuntu

USER ubuntu
ENV HOME=/home/ubuntu

# Build gate. Runs with no GPU and no weights: it confirms the serving stack
# imports, that the xet workaround above still matches the installed pins, that
# the entrypoint assembles the serve command it claims to, and that no weight
# file is present in any layer.
RUN python3 /opt/npa-cosmos3-serving/verify_env.py

EXPOSE 8000

# The readiness window is minutes, not seconds. HSDP-sharded configs reached
# 'Application startup complete' in roughly 280-290 s with the weights already in
# the page cache, and about 320 s longer on a cold one. start-period covers the
# cold case; a tighter probe would report a healthy boot as a failure and restart
# it into another one.
HEALTHCHECK --interval=30s --timeout=10s --start-period=1200s --retries=3 \
  CMD python3 -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('NPA_COSMOS3_SERVE_PORT','8000')+'/v1/models', timeout=8)" || exit 1

ENTRYPOINT ["/opt/npa-cosmos3-serving/entrypoint.sh"]
