ARG BASE_IMAGE=npa-genesis:cuda13-b300-0.4.6-sm80-sm90-sm100-sm103-sm120-20260803T034152Z
FROM ${BASE_IMAGE}

ARG SIM2REAL_EVAL_VERSION=cuda13-b300-0.1.3-sm80-sm90-sm100-sm103-sm120-20260803T034152Z
ARG NPA_SOURCE_SHA
ARG UBUNTU_SNAPSHOT=20260801T053000Z

LABEL npa.tool="loop-eval" \
      npa.version="${SIM2REAL_EVAL_VERSION}" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}"

ENV PYTHONUNBUFFERED=1 \
    NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    NPA_SIM2REAL_GENESIS_BATCH_SIZE=16 \
    NPA_SIM2REAL_GENESIS_MAX_STEPS=240 \
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
COPY --chmod=0755 docker/workbench/common/workflow_runtime_entrypoint.sh \
    /usr/local/bin/npa-workflow-entrypoint
COPY --chmod=0755 docker/workbench/common/install_workflow_runtime_prereqs.sh \
    /usr/local/bin/install-workflow-runtime-prereqs

USER root

RUN /usr/local/bin/install-workflow-runtime-prereqs "${UBUNTU_SNAPSHOT}" \
    && test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40 \
    && find /opt/npa/src /opt/npa/workflows -type d -exec chmod a+rx {} + \
    && find /opt/npa/src /opt/npa/workflows -type f -exec chmod a+r {} + \
    && chmod a+r /opt/npa/pyproject.toml \
    && python -m pip install --no-deps \
      -r /opt/npa/sim2real-control-requirements.txt \
      -r /opt/npa/sim2real-genesis-requirements.txt \
    && python -m pip check \
    && python -m pip install --no-deps -e /opt/npa \
    && python -m py_compile /opt/npa/src/npa/workflows/sim2real_loop.py \
    && python -c "from npa.genesis.env_pick_place import EnvConfig, FrankaPickPlaceEnv; assert EnvConfig(action_space='cartesian').action_space == 'cartesian'; assert FrankaPickPlaceEnv.__name__ == 'FrankaPickPlaceEnv'" \
    && chown -R ubuntu:ubuntu /workspace /opt/npa/src /opt/npa/pyproject.toml

USER ubuntu

ENTRYPOINT ["/usr/local/bin/npa-workflow-entrypoint"]
