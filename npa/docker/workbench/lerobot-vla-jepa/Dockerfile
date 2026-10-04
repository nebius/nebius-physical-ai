# VLA-JEPA is deliberately a distinct candidate from the generic LeRobot image.
# Its exact upstream policy is available only in LeRobot v0.6.1, not the accepted
# generic 0.5.1/0.6.0 surface.
ARG LEROBOT_SOURCE_REVISION=7e241bd630a3719a56157a497ce5d08f244784f1

FROM ghcr.io/nebius/nebius-physical-ai/npa-lerobot@sha256:8d513f8558253fc484808a1e53a63a5da5a0c280ff973c4e590dd3e04b228643 AS vla_jepa_build

ARG LEROBOT_SOURCE_REVISION

USER ubuntu
WORKDIR /opt/lerobot

# This image intentionally contains the pinned LeRobot implementation and its
# open dependency extras, but no checkpoint, Qwen, V-JEPA2, dataset, or cache.
# Those artifacts are revision-addressed runtime fetches in the workflow.
RUN /opt/lerobot/venv/bin/pip uninstall -y lerobot \
    && /opt/lerobot/venv/bin/pip install --no-cache-dir \
      "lerobot[training,evaluation,libero,vla_jepa] @ git+https://github.com/huggingface/lerobot.git@${LEROBOT_SOURCE_REVISION}" \
    && /opt/lerobot/venv/bin/pip uninstall -y wandb \
    && /opt/lerobot/venv/bin/python - <<'PY'
import importlib.metadata
from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig

assert importlib.metadata.version("lerobot") == "0.6.1"
assert VLAJEPAConfig.__name__ == "VLAJEPAConfig"
PY

# The generic 0.6.0 service image carries only its service-side NPA subset.
# Bake this candidate's reviewed workflow adapter into the same vendor venv so
# the worker never depends on an ambient or mutable source overlay at GPU time.
COPY --chown=ubuntu:ubuntu src/npa /opt/vla-jepa-npa/src/npa
# The inherited service image previously retained this populated pip cache.
# Delete it after the final install so runtime-only package payloads cannot be
# mistaken for an intentionally baked model/data cache.
RUN /opt/lerobot/venv/bin/pip install --no-cache-dir "rerun-sdk==0.38.1" \
    && rm -rf /home/ubuntu/.cache/pip \
    && test ! -e /home/ubuntu/.cache/pip \
    && PYTHONPATH=/opt/vla-jepa-npa/src /opt/lerobot/venv/bin/python - <<'PY'
import rerun
from npa.workflows import lerobot_vla_jepa

assert rerun.__version__ == "0.38.1"
assert lerobot_vla_jepa.LEROBOT_VERSION == "0.6.1"
PY

# Copying the merged build filesystem into a scratch final stage means cache
# bytes inherited in a parent image are not retained in the published image's
# layer history. A later rm alone would only create a whiteout and leave those
# bytes recoverable from the parent layer.
FROM scratch AS vla_jepa_runtime

ARG LEROBOT_SOURCE_REVISION
ARG NPA_ADAPTER_REVISION=local-uncommitted

COPY --from=vla_jepa_build / /

LABEL npa.tool="lerobot-vla-jepa" \
      npa.lerobot.version="0.6.1" \
      npa.lerobot.source-revision="${LEROBOT_SOURCE_REVISION}" \
      npa.adapter.source-revision="${NPA_ADAPTER_REVISION}" \
      org.opencontainers.image.source="https://github.com/huggingface/lerobot" \
      org.opencontainers.image.licenses="Apache-2.0"

USER ubuntu
WORKDIR /opt/lerobot
ENV WANDB_MODE=disabled \
    WANDB_DISABLED=true \
    HF_HUB_DISABLE_TELEMETRY=1 \
    NPA_LEROBOT_VERSION=0.6.1 \
    PYTHONPATH=/opt/vla-jepa-npa/src:$PYTHONPATH \
    PATH=/opt/lerobot/venv/bin:$PATH

# The parent service entrypoint would start FastAPI rather than the workflow's
# native CLI.  A job image leaves command selection to the NPA stage renderer.
HEALTHCHECK NONE
ENTRYPOINT []
CMD ["lerobot-train", "--help"]
