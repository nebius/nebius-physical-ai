ARG BASE_IMAGE=npa-genesis:cuda13-b300-0.4.6-sm80-sm90-sm100-sm103-sm120-20260803T034152Z
FROM ${BASE_IMAGE}

ARG LEROBOT_VLM_RL_VERSION=0.1.1
ARG NPA_SOURCE_SHA

LABEL npa.tool="lerobot-vlm-rl" \
      npa.version="${LEROBOT_VLM_RL_VERSION}" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}"

ENV PYTHONUNBUFFERED=1 \
    NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    PYTHONPATH=/opt/npa/src

WORKDIR /workspace

USER root

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu workflows /opt/npa/workflows
COPY docker/workbench/common/sim2real-control-requirements.txt \
    /opt/npa/sim2real-control-requirements.txt
COPY docker/workbench/common/sim2real-genesis-requirements.txt \
    /opt/npa/sim2real-genesis-requirements.txt

RUN test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40 \
    && find /opt/npa/src /opt/npa/workflows -type d -exec chmod a+rx {} + \
    && find /opt/npa/src /opt/npa/workflows -type f -exec chmod a+r {} + \
    && chmod a+r /opt/npa/pyproject.toml \
    && python -m pip install --no-deps \
      -r /opt/npa/sim2real-control-requirements.txt \
      -r /opt/npa/sim2real-genesis-requirements.txt \
    && python -m pip install --no-deps -e /opt/npa \
    && python -m pip check \
    && python - <<'PY'
from npa.workbench.lerobot.policy_container import parse_vlm_signal_batch

payload = {
    "schema": "npa.sim2real.rl_signal.v1",
    "rollout_id": "build-smoke",
    "per_step": [{"step": 0, "reward": 0.5, "advantage": 0.0}],
}
assert parse_vlm_signal_batch(payload)[0].rollout_id == "build-smoke"
PY

RUN chown -R ubuntu:ubuntu /workspace /opt/npa/src /opt/npa/pyproject.toml

USER ubuntu

ENTRYPOINT ["python", "-m", "npa.workbench.lerobot.policy_container"]
CMD ["vlm-signal-step", "--help"]
