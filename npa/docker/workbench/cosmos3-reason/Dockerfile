ARG BASE_IMAGE=npa-base:cuda13-b300-sm80-sm90-sm120-latest
FROM ${BASE_IMAGE}

ARG COSMOS3_REASON_VERSION=3.0.0

LABEL npa.tool="cosmos3-reason" \
      npa.version="${COSMOS3_REASON_VERSION}"

ENV PYTHONUNBUFFERED=1 \
    NPA_COSMOS3_REASON_MODEL=nvidia/Cosmos-Predict2.5-2B \
    NPA_COSMOS3_TRANSFER_MODEL=nvidia/Cosmos-Transfer2.5-2B \
    NPA_COSMOS_GUARDRAIL_MODEL=nvidia/Cosmos-Guardrail1 \
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
CMD ["inner-loop", "--help"]
