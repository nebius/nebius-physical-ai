ARG BASE_IMAGE=npa-base:cuda13-b300-sm80-sm90-sm100-sm103-sm120-v2-latest
FROM ${BASE_IMAGE}

ARG COSMOS3_REASON_VERSION=cuda13-b300-3.0.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z

LABEL npa.tool="cosmos3-reason" \
      npa.version="${COSMOS3_REASON_VERSION}"

ENV PYTHONUNBUFFERED=1 \
    NPA_COSMOS3_REASON_MODEL=nvidia/Cosmos-Predict2.5-2B \
    NPA_COSMOS3_TRANSFER_MODEL=nvidia/Cosmos-Transfer2.5-2B \
    NPA_COSMOS_GUARDRAIL_MODEL=nvidia/Cosmos-Guardrail1 \
    NPA_COSMOS_REASON2_MODEL_ID=nvidia/Cosmos-Reason2-8B \
    NPA_COSMOS_REASON3_MODEL_ID=nvidia/Cosmos-Reason2-2B \
    NPA_COSMOS_REASON_MODEL_ID=nvidia/Cosmos-Reason2-8B \
    NPA_COSMOS_REASON2_CACHE=/tmp/hf_home/cosmos-reason2 \
    NPA_COSMOS_REASON3_CACHE=/tmp/hf_home/cosmos-reason2-2b \
    NPA_COSMOS_REASON_CACHE=/tmp/hf_home/cosmos-reason2 \
    HF_HOME=/tmp/hf_home \
    PYTHONPATH=/opt/npa/src:$PYTHONPATH

WORKDIR /workspace

USER root

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa

USER root

# Do not recursively chown the inherited /opt/npa/venv: overlayfs would copy
# the complete CUDA/PyTorch environment into this child layer.
RUN mkdir -p /tmp/hf_home/cosmos-reason2 /tmp/hf_home/cosmos-reason2-2b \
    && python -m pip install tomli \
    && python -m pip install \
      "accelerate>=0.33,<2" \
      "huggingface_hub>=0.24,<2" \
      "qwen-vl-utils>=0.0.8,<1" \
      "safetensors>=0.4,<1" \
      "transformers>=4.51,<5" \
    && python -m pip install --no-deps -e /opt/npa \
    && python -m py_compile /opt/npa/src/npa/workflows/sim2real_loop.py \
    && chown -R ubuntu:ubuntu \
      /workspace /tmp/hf_home /opt/npa/src /opt/npa/pyproject.toml

# --- Schedulable-on-Kubernetes prerequisites -----------------------------------
#
# SkyPilot's Kubernetes runtime bootstrap execs into the container and needs a system python3,
# rsync, and passwordless sudo. Without them provisioning never starts and the only symptom is
#   KubernetesError: Failed to get ssh user for pod ...: container not found ("ray-node")
# which says nothing about what is missing (live job 304). The runtime user also needs a
# writable HOME, because `npa workbench cosmos3 text-to-image` clones the Cosmos framework and
# lets uv build an environment there before it can generate anything.
# npa-base deliberately purges linux-libc-dev as final-image CVE hardening,
# leaving its build-time libc6-dev dependency unsatisfied. Repair that
# relationship temporarily, install everything, then reapply the purge.
RUN apt-get update \
    && apt-get install -y --no-install-recommends --fix-broken

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         python3 python3-venv python3-pip rsync curl openssh-client sudo netcat-openbsd \
    && rm -rf /var/lib/apt/lists/* \
    && printf 'ubuntu ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/99-npa-runtime-user \
    && chmod 0440 /etc/sudoers.d/99-npa-runtime-user \
    && mkdir -p /home/ubuntu && chown -R ubuntu:ubuntu /home/ubuntu \
    && python3 --version \
    && dpkg --purge --force-depends linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*

# NOTE: deliberately NOT `ENV PATH=/usr/bin:$PATH`. The Isaac images need that, because their
# default python3 is a kit interpreter that cannot import its own site-packages outside
# python.sh. Here the opposite holds: this image's own python is the one carrying npa and its
# dependencies, and forcing the bare system python in front of it made setup fail with
# "npa is not importable after setup" (live job 307). SkyPilot needs a system python3 to EXIST,
# not to be first.

USER ubuntu

ENTRYPOINT ["python", "-m", "npa.workflows.sim2real_loop"]
CMD ["inner-loop", "--help"]
