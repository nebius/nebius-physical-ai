ARG BASE_IMAGE=npa-genesis:0.4.6-sm80-sm90-sm120-latest
FROM ${BASE_IMAGE}

ARG SIM2REAL_EVAL_VERSION=0.1.1-genuine-sm120

LABEL npa.tool="loop-eval" \
      npa.version="${SIM2REAL_EVAL_VERSION}"

ENV PYTHONUNBUFFERED=1 \
    NPA_SIM2REAL_GENESIS_BATCH_SIZE=16 \
    NPA_SIM2REAL_GENESIS_MAX_STEPS=240 \
    PYTHONPATH=/opt/npa/src:$PYTHONPATH

WORKDIR /workspace

USER root

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa

USER root

RUN python -m pip install tomli \
    && python -m pip install --no-deps -e /opt/npa \
    && python -m py_compile /opt/npa/src/npa/workflows/sim2real_loop.py \
    && python -c "from npa.genesis.env_pick_place import EnvConfig, FrankaPickPlaceEnv; assert EnvConfig(action_space='cartesian').action_space == 'cartesian'; assert FrankaPickPlaceEnv.__name__ == 'FrankaPickPlaceEnv'" \
    && chown -R ubuntu:ubuntu /workspace /opt/npa/src /opt/npa/pyproject.toml

USER ubuntu

ENTRYPOINT ["python", "-m", "npa.workflows.sim2real_loop"]
CMD ["full-loop", "--help"]
