ARG BASE_IMAGE=ghcr.io/nebius/nebius-physical-ai/npa-cosmos3-reason@sha256:e708aa32b9247eaedf1a7a5b0d82adbe65b95a9af50721b9d09fddbfd508bbc9
FROM ${BASE_IMAGE} AS sanitized

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
COPY --chmod=0444 docker/workbench/common/repair_accelerate_checkpoints.py /opt/npa/repair_accelerate_checkpoints.py

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
    && python -m pip install --no-deps --upgrade \
      "pip==26.2.2" \
      "certifi==2026.10.19" \
      "Pillow==13.0.0" \
      "cryptography==50.0.2" \
      "cffi==2.1.1" \
      "pycparser==3.0" \
      "tomli==2.4.1" \
      "accelerate==1.14.0" \
      "huggingface_hub==0.36.0" \
      "qwen-vl-utils==0.0.14" \
      "safetensors==0.8.0" \
      "transformers==4.57.6" \
    && python /opt/npa/repair_accelerate_checkpoints.py \
    && python -m pip check \
    && python -c "import site; from pathlib import Path; Path(site.getsitepackages()[0], 'npa-exact-source.pth').write_text('/opt/npa/src\\n', encoding='utf-8')" \
    && python -c 'import npa.workflows.sim2real.runtime_attestation' \
    && env -u PYTHONPATH /opt/npa/venv/bin/python -c \
      'import boto3; from npa.workflows.sim2real.workflow_stage import main; assert callable(main)' \
    && env -u PYTHONPATH /opt/npa/venv/bin/python -c \
      'from cryptography.exceptions import InvalidSignature; from npa.workflows.sim2real.cosmos3_augment import run_cosmos3_structural_augmentation; assert InvalidSignature and callable(run_cosmos3_structural_augmentation)' \
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
    && rm -rf /opt/nvidia/nsight-compute \
    && test ! -e /opt/nvidia/nsight-compute \
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

# The parent is a trusted build input, but its historical layers are also
# scanner-visible.  Emit only the sanitized final filesystem so package upgrades
# above replace the inherited bytes instead of depending on scanner exceptions.
FROM scratch AS runtime

ARG COSMOS3_REASON_VERSION=cuda13-b300-3.0.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z
ARG NPA_SOURCE_SHA

LABEL npa.tool="cosmos3-reason" \
      npa.version="${COSMOS3_REASON_VERSION}" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

COPY --from=sanitized / /

ENV PATH=/opt/npa/venv/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    NVARCH=x86_64 \
    NVIDIA_REQUIRE_CUDA="cuda>=13.0 brand=unknown,driver>=535,driver<536 brand=grid,driver>=535,driver<536 brand=tesla,driver>=535,driver<536 brand=nvidia,driver>=535,driver<536 brand=quadro,driver>=535,driver<536 brand=quadrortx,driver>=535,driver<536 brand=nvidiartx,driver>=535,driver<536 brand=vapps,driver>=535,driver<536 brand=vpc,driver>=535,driver<536 brand=vcs,driver>=535,driver<536 brand=vws,driver>=535,driver<536 brand=cloudgaming,driver>=535,driver<536 brand=unknown,driver>=550,driver<551 brand=grid,driver>=550,driver<551 brand=tesla,driver>=550,driver<551 brand=nvidia,driver>=550,driver<551 brand=quadro,driver>=550,driver<551 brand=quadrortx,driver>=550,driver<551 brand=nvidiartx,driver>=550,driver<551 brand=vapps,driver>=550,driver<551 brand=vpc,driver>=550,driver<551 brand=vcs,driver>=550,driver<551 brand=vws,driver>=550,driver<551 brand=cloudgaming,driver>=550,driver<551 brand=unknown,driver>=565,driver<566 brand=grid,driver>=565,driver<566 brand=tesla,driver>=565,driver<566 brand=nvidia,driver>=565,driver<566 brand=quadro,driver>=565,driver<566 brand=quadrortx,driver>=565,driver<566 brand=nvidiartx,driver>=565,driver<566 brand=vapps,driver>=565,driver<566 brand=vpc,driver>=565,driver<566 brand=vcs,driver>=565,driver<566 brand=vws,driver>=565,driver<566 brand=cloudgaming,driver>=565,driver<566 brand=unknown,driver>=570,driver<571 brand=grid,driver>=570,driver<571 brand=tesla,driver>=570,driver<571 brand=nvidia,driver>=570,driver<571 brand=quadro,driver>=570,driver<571 brand=quadrortx,driver>=570,driver<571 brand=nvidiartx,driver>=570,driver<571 brand=vapps,driver>=570,driver<571 brand=vpc,driver>=570,driver<571 brand=vcs,driver>=570,driver<571 brand=vws,driver>=570,driver<571 brand=cloudgaming,driver>=570,driver<571 brand=unknown,driver>=575,driver<576 brand=grid,driver>=575,driver<576 brand=tesla,driver>=575,driver<576 brand=nvidia,driver>=575,driver<576 brand=quadro,driver>=575,driver<576 brand=quadrortx,driver>=575,driver<576 brand=nvidiartx,driver>=575,driver<576 brand=vapps,driver>=575,driver<576 brand=vpc,driver>=575,driver<576 brand=vcs,driver>=575,driver<576 brand=vws,driver>=575,driver<576 brand=cloudgaming,driver>=575,driver<576" \
    CUDA_VERSION=13.0.1 \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST="8.0 9.0 10.0 10.3 12.0" \
    CUDA_HOME=/usr/local/cuda \
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
    HF_HUB_CACHE=/opt/npa/hf_cache/hub \
    TRANSFORMERS_CACHE=/opt/npa/hf_cache/transformers \
    NPA_INPUT_PATH=/input \
    NPA_OUTPUT_PATH=/output \
    INPUT_PATH=/input \
    OUTPUT_PATH=/output \
    PYTHONPATH=/opt/npa/src

WORKDIR /workspace
USER ubuntu

ENTRYPOINT ["/usr/local/bin/npa-workflow-entrypoint"]
