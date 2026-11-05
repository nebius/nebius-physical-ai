ARG BASE_IMAGE=ghcr.io/nebius/nebius-physical-ai/npa-envgen@sha256:38a9c229218600dca1e4071561e51c50a76cb0c538e60c9a70b80e81f06fc726
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

# The immutable EnvGen base registers this source root via a ``.pth`` file.
# Remove it before copying the child so an omitted/renamed module cannot be
# imported from an earlier parent revision.
RUN rm -rf /opt/npa/src/npa

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu workflows /opt/npa/workflows

# Envgen registers /opt/npa/src through npa-exact-source.pth, so the build
# smoke deliberately proves imports work without an editable child install.
RUN test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40 \
    && find /opt/npa/src /opt/npa/workflows -type d -exec chmod a+rx {} + \
    && find /opt/npa/src /opt/npa/workflows -type f -exec chmod a+r {} + \
    && chmod a+r /opt/npa/pyproject.toml \
    && python -m pip check \
    && env -u PYTHONPATH python -c \
      'from npa.workbench.lerobot.policy_container import parse_vlm_signal_batch; assert callable(parse_vlm_signal_batch)' \
    && python -c 'import importlib.metadata as m; names = {d.metadata["Name"].lower() for d in m.distributions()}; assert names.isdisjoint({"datasets", "lerobot", "genesis-world", "torch"})' \
    && env -u PYTHONPATH python - <<'PY'
from pathlib import Path

import npa.workbench.lerobot.policy_container as policy_container

payload = {
    "schema": "npa.sim2real.rl_signal.v1",
    "rollout_id": "build-smoke",
    "per_step": [{"step": 0, "reward": 0.5, "advantage": 0.0}],
}
assert Path(policy_container.__file__).resolve() == Path(
    "/opt/npa/src/npa/workbench/lerobot/policy_container.py"
)
assert policy_container.parse_vlm_signal_batch(payload)[0].rollout_id == "build-smoke"
PY

RUN chown -R ubuntu:ubuntu /workspace /opt/npa/src /opt/npa/pyproject.toml

USER ubuntu

ENTRYPOINT ["python", "-m", "npa.workbench.lerobot.policy_container"]
CMD ["vlm-signal-step", "--help"]
