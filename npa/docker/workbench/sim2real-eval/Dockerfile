ARG BASE_IMAGE=npa-genesis:0.4.6-sm80-sm90-sm120-latest
FROM ${BASE_IMAGE}

ARG SIM2REAL_EVAL_VERSION=0.1.0

LABEL npa.tool="sim2real-eval" \
      npa.version="${SIM2REAL_EVAL_VERSION}"

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/npa/src:$PYTHONPATH

WORKDIR /workspace

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa

RUN python -m pip install tomli \
    && python -m pip install --no-deps -e /opt/npa \
    && python -m py_compile /opt/npa/src/npa/workflows/sim2real_loop.py \
    && chown -R ubuntu:ubuntu /workspace /opt/npa

USER ubuntu

ENTRYPOINT ["python", "-m", "npa.workflows.sim2real_loop"]
CMD ["full-loop", "--help"]
