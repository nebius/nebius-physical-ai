ARG BASE_IMAGE=npa-genesis:0.4.6-sm80-sm90-sm120-latest
FROM ${BASE_IMAGE}

ARG LEROBOT_VLM_RL_VERSION=0.1.0

LABEL npa.tool="lerobot-vlm-rl" \
      npa.version="${LEROBOT_VLM_RL_VERSION}"

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/npa/src

WORKDIR /workspace

USER root

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa

RUN python -m pip install tomli \
    && python -m pip install --no-deps -e /opt/npa \
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
