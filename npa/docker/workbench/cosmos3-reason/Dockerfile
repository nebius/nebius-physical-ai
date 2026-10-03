ARG BASE_IMAGE=ghcr.io/nebius/nebius-physical-ai/npa-cosmos3-reason@sha256:e708aa32b9247eaedf1a7a5b0d82adbe65b95a9af50721b9d09fddbfd508bbc9
FROM ${BASE_IMAGE}

ARG COSMOS3_REASON_VERSION=cuda13-b300-3.0.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z
ARG NPA_SOURCE_SHA
ARG UBUNTU_SNAPSHOT=20260801T053000Z

LABEL npa.tool="cosmos3-reason" \
      npa.version="${COSMOS3_REASON_VERSION}" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

ENV PYTHONUNBUFFERED=1 \
    NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    NPA_COSMOS3_REASON_MODEL=nvidia/Cosmos-Predict2.5-2B \
    NPA_COSMOS3_TRANSFER_MODEL=nvidia/Cosmos-Transfer2.5-2B \
    NPA_COSMOS_GUARDRAIL_MODEL=nvidia/Cosmos-Guardrail1 \
    NPA_COSMOS_REASON2_MODEL_ID=nvidia/Cosmos-Reason2-8B \
    NPA_COSMOS_REASON3_MODEL_ID=nvidia/Cosmos-Reason2-2B \
    NPA_COSMOS_REASON_MODEL_ID=nvidia/Cosmos-Reason2-8B \
    NPA_COSMOS_REASON2_CACHE=/tmp/hf_home/cosmos-reason2 \
    NPA_COSMOS_REASON3_CACHE=/tmp/hf_home/cosmos-reason2-2b \
    NPA_COSMOS_REASON_CACHE=/tmp/hf_home/cosmos-reason2 \
    NPA_SKIP_EAGER_IMPORTS=1 \
    NPA_BAKED_PYTHON=/opt/npa/venv/bin/python \
    HF_HOME=/tmp/hf_home \
    PYTHONPATH=/opt/npa/src

WORKDIR /workspace

USER root

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu workflows /opt/npa/workflows
COPY --chmod=0755 docker/workbench/common/workflow_runtime_entrypoint.sh \
    /usr/local/bin/npa-workflow-entrypoint

USER root

# Do not recursively chown the inherited /opt/npa/venv: overlayfs would copy
# the complete CUDA/PyTorch environment into this child layer. The public
# parent also carries its historical editable NPA distribution; this child
# supplies an exact-source .pth, so remove stale package metadata before pip
# check instead of inheriting dependencies for unrelated tools.
RUN mkdir -p /tmp/hf_home/cosmos-reason2 /tmp/hf_home/cosmos-reason2-2b \
    && find /opt/npa/src /opt/npa/workflows -type d -exec chmod a+rx {} + \
    && find /opt/npa/src /opt/npa/workflows -type f -exec chmod a+r {} + \
    && chmod a+r /opt/npa/pyproject.toml \
    && test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40 \
    && python -m pip uninstall -y npa \
    && python -m pip install --no-deps \
      "tomli==2.4.1" \
      "accelerate==1.14.0" \
      "huggingface_hub==0.36.0" \
      "qwen-vl-utils==0.0.14" \
      "safetensors==0.8.0" \
      "transformers==4.57.6" \
    && python -m pip check \
    && python -c "import site; from pathlib import Path; Path(site.getsitepackages()[0], 'npa-exact-source.pth').write_text('/opt/npa/src\\n', encoding='utf-8')" \
    && python -c 'import npa.workflows.sim2real.runtime_attestation' \
    && env -u PYTHONPATH /opt/npa/venv/bin/python -c \
      'import boto3; from npa.workflows.sim2real.workflow_stage import main; assert callable(main)' \
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
RUN rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*.list \
      /etc/apt/sources.list.d/*.sources \
    && printf '%s\n' \
      'Types: deb' \
      "URIs: https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/" \
      'Suites: jammy jammy-updates jammy-backports jammy-security' \
      'Components: main restricted universe multiverse' \
      'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg' \
      > /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends --fix-broken

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         python3 python3-venv python3-pip rsync curl openssh-client openssh-server \
         procps sudo netcat-openbsd \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/ssh/ssh_host_* \
    && test -z "$(find /etc/ssh -maxdepth 1 -type f -name 'ssh_host_*' -print -quit)" \
    && printf 'ubuntu ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/99-npa-runtime-user \
    && chmod 0440 /etc/sudoers.d/99-npa-runtime-user \
    && mkdir -p /home/ubuntu && chown -R ubuntu:ubuntu /home/ubuntu \
    && rm -f /etc/ssh/ssh_host_* \
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

ENTRYPOINT ["/usr/local/bin/npa-workflow-entrypoint"]
