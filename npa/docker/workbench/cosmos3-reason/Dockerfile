ARG BASE_IMAGE=npa-base:cuda13-b300-sm80-sm90-sm120-latest
FROM ${BASE_IMAGE}

ARG COSMOS3_REASON_VERSION=3.0.1-genuine-sm120

LABEL npa.tool="cosmos3-reason" \
      npa.version="${COSMOS3_REASON_VERSION}"

ENV PYTHONUNBUFFERED=1 \
    NPA_COSMOS3_REASON_MODEL=nvidia/Cosmos-Predict2.5-2B \
    NPA_COSMOS3_TRANSFER_MODEL=nvidia/Cosmos-Transfer2.5-2B \
    NPA_COSMOS_GUARDRAIL_MODEL=nvidia/Cosmos-Guardrail1 \
    NPA_COSMOS_REASON2_MODEL_ID=nvidia/Cosmos-Reason2-8B \
    NPA_COSMOS_REASON3_MODEL_ID=nvidia/Cosmos-Reason1-7B \
    NPA_COSMOS_REASON_MODEL_ID=nvidia/Cosmos-Reason2-8B \
    NPA_COSMOS_REASON2_CACHE=/tmp/hf_home/cosmos-reason2 \
    NPA_COSMOS_REASON3_CACHE=/tmp/hf_home/cosmos-reason3 \
    NPA_COSMOS_REASON_CACHE=/tmp/hf_home/cosmos-reason2 \
    HF_HOME=/tmp/hf_home \
    PYTHONPATH=/opt/npa/src:$PYTHONPATH

WORKDIR /workspace

USER root

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa

USER root

RUN mkdir -p /tmp/hf_home/cosmos-reason2 /tmp/hf_home/cosmos-reason3 \
    && python -m pip install tomli \
    && python -m pip install \
      "accelerate>=0.33,<2" \
      "huggingface_hub>=0.24,<2" \
      "qwen-vl-utils>=0.0.8,<1" \
      "safetensors>=0.4,<1" \
      "transformers>=4.51,<5" \
    && python -m pip install --no-deps -e /opt/npa \
    && python -m py_compile /opt/npa/src/npa/workflows/sim2real_loop.py \
    && chown -R ubuntu:ubuntu /workspace /tmp/hf_home /opt/npa/src /opt/npa/pyproject.toml

USER ubuntu

ENTRYPOINT ["python", "-m", "npa.workflows.sim2real_loop"]
CMD ["inner-loop", "--help"]
