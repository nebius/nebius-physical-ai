ARG BASE_IMAGE=ghcr.io/nebius/nebius-physical-ai/npa-envgen@sha256:08eb75118f5a04194d33a60308212db7706dd9c339d74afc5471a58608bf0422
FROM ${BASE_IMAGE}

ARG LEROBOT_VLM_RL_VERSION=0.1.1
ARG NPA_SOURCE_SHA

LABEL npa.tool="lerobot-vlm-rl" \
      npa.version="${LEROBOT_VLM_RL_VERSION}" \
      npa.base.image="npa-envgen" \
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
    && python -m pip uninstall -y transformers \
    && python -m pip install --no-deps \
      -r /opt/npa/sim2real-control-requirements.txt \
      -r /opt/npa/sim2real-genesis-requirements.txt \
    && python -m pip check \
    && env -u PYTHONPATH python -c \
      'from npa.workbench.lerobot.policy_container import parse_vlm_signal_batch; assert callable(parse_vlm_signal_batch)' \
    && python -c 'import importlib.metadata as m; assert not any(d.metadata["Name"].lower() == "transformers" for d in m.distributions())' \
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
