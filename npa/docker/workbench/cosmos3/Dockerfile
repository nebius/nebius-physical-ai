# npa-cosmos3: Cosmos 3 omni-model generation runtime (text2image / image2image /
# text2video / image2video / video2video) for the workbench.
#
# WEIGHTS ARE NEVER BAKED. This image ships only the OpenMDW-1.1 cosmos-framework
# SOURCE at a pinned commit plus its inference environment. Every Cosmos 3
# checkpoint (nvidia/Cosmos3-*, the Wan VAE it pulls, the guardrail models) is
# downloaded AT RUN TIME with the operator's OWN Hugging Face token, and with
# their own NGC key when NGC-hosted artifacts are required. That is what keeps the
# built image redistributable (packaging-contract.yaml redistribution: public):
# NVIDIA delivers the gated weights to each operator under that operator's own
# license acceptance, and we ship no weight bytes. npa.workbench.cosmos.generate
# (require_model_access) refuses to run without those credentials, and the build
# asserts below that no checkpoint file landed in a layer.
ARG BASE_IMAGE=nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

# cosmos-framework release that COSMOS3_REF points at.
ARG COSMOS3_VERSION=1.2.2
ARG COSMOS3_SOURCE_REPO=https://github.com/NVIDIA/cosmos-framework.git
# Pinned so a build is reproducible and the golden eval means something; bumping
# this is a deliberate, tested change (upstream main moves and re-locks deps).
ARG COSMOS3_REF=5e67049cd94acb667786f1e6dd0dab821cb90c97
# cu130 matches the CUDA 13 base and is the wheel set upstream pins for it.
# Generation needs the `guardrail` extra (guardrails are on by default). Upstream's
# own Dockerfile takes `--all-extras`, which drags in the whole `train` closure
# (433 vs 149 resolved packages: megatron-core, torchtitan, DALI, jupyterlab...).
# This image is generate-only, so it installs the `guardrail` extra plus the two
# lock-pinned `train` members the generate path actually imports, and runs
# verify_env.py at build time so a framework bump that needs more fails the build
# instead of the first GPU run.
ARG COSMOS3_UV_GROUP=cu130
ARG COSMOS3_UV_EXTRA=guardrail
ARG COSMOS3_INFERENCE_DEPS="qwen-vl-utils==0.0.14 boto3==1.40.61"

LABEL npa.tool="cosmos3" \
      npa.version="${COSMOS3_VERSION}" \
      npa.cosmos3.ref="${COSMOS3_REF}"

# UV_PYTHON_INSTALL_DIR keeps uv's interpreters out of /root (mode 700) so the
# non-root runtime user can execute the venvs that link against them.
# UV_LINK_MODE is left at uv's default (hardlink): the venv then shares inodes
# with the build-time cache, so deleting that cache in the same layer keeps peak
# build disk at roughly one copy of the ~9 GB torch/CUDA closure instead of two.
# TRITON_PTXAS_PATH: Triton's bundled ptxas does not cover the newest architectures.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    COSMOS3_REPO=/opt/cosmos3/cosmos-framework \
    NPA_COSMOS3_CHECKPOINT=Cosmos3-Nano \
    NPA_COSMOS3_OUTPUT_DIR=/tmp/npa-cosmos3-generate \
    COSMOS_TRAINING=0 \
    HF_HOME=/tmp/hf_home \
    COSMOS_DOWNLOAD_CACHE_DIR=/tmp/hf_home/cosmos-downloads \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    PATH=/opt/npa/.venv/bin:${PATH}

# openssh-server + sudo are here for the SkyPilot Kubernetes bootstrap, which
# overrides the entrypoint with its own shell and then installs an SSH runtime as
# the pod user. Without them a workbench image cannot host a SkyPilot k8s task
# (it dies with `sudo: command not found`), which is why the cosmos3-generate
# workflow needs them. The final user stays non-root; the sudoers grant below is
# scoped to that bootstrap.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        git-lfs \
        openssh-server \
        sudo \
        wget \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /usr/local/bin/

# The runtime user owns its trees BEFORE the heavy layers, and every later layer
# runs as that user. A trailing `chown -R` would instead rewrite the whole 9.3 GB
# venv into a second layer, taking the image from 27.3 GB to ~37 GB.
RUN id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu
RUN mkdir -p /opt/cosmos3 /opt/npa /opt/uv /workspace /tmp/hf_home \
        "${NPA_COSMOS3_OUTPUT_DIR}" \
    && chown -R ubuntu:ubuntu /opt/cosmos3 /opt/npa /opt/uv /workspace \
       /tmp/hf_home "${NPA_COSMOS3_OUTPUT_DIR}" \
    && echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-skypilot \
    && chmod 0440 /etc/sudoers.d/90-skypilot

USER ubuntu

# Framework source at the pinned commit. History is dropped and the upstream
# license/notice files are kept alongside for attribution.
RUN git clone --filter=blob:none "${COSMOS3_SOURCE_REPO}" "${COSMOS3_REPO}" \
    && git -C "${COSMOS3_REPO}" checkout --detach "${COSMOS3_REF}" \
    && mkdir -p /opt/cosmos3/licenses \
    && cp "${COSMOS3_REPO}/LICENSE" "${COSMOS3_REPO}/NOTICE" \
       "${COSMOS3_REPO}/ATTRIBUTIONS.md" /opt/cosmos3/licenses/ \
    && rm -rf "${COSMOS3_REPO}/.git"

# The real inference environment, in the framework's own venv (torch cu130 +
# flash-attn + guardrail deps). verify_env.py then walks the inference graph for
# every advertised generation mode, so the trimmed dependency set is proven at
# build time rather than assumed.
COPY --chown=ubuntu:ubuntu docker/workbench/cosmos3/verify_env.py /opt/cosmos3/verify_env.py
RUN cd "${COSMOS3_REPO}" \
    && UV_CACHE_DIR=/tmp/uv-cache uv python install \
    && chmod -R a+rX /opt/uv \
    && UV_CACHE_DIR=/tmp/uv-cache uv sync --frozen \
       --extra "${COSMOS3_UV_EXTRA}" --group "${COSMOS3_UV_GROUP}" \
    && UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv/bin/python \
       ${COSMOS3_INFERENCE_DEPS} \
    && COSMOS_DEVICE=cpu .venv/bin/python /opt/cosmos3/verify_env.py \
    && rm -rf /tmp/uv-cache

# npa CLI/SDK in its own venv so `npa workbench cosmos3 generate` is the single
# entrypoint for the CLI, the SDK, and the SkyPilot workflow.
COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu workflows /opt/npa/workflows
RUN UV_CACHE_DIR=/tmp/uv-cache uv venv --python 3.12 /opt/npa/.venv \
    && UV_CACHE_DIR=/tmp/uv-cache uv pip install \
       --python /opt/npa/.venv/bin/python -e /opt/npa \
    && /opt/npa/.venv/bin/npa workbench cosmos3 generate --help >/dev/null \
    && rm -rf /tmp/uv-cache

COPY --chown=ubuntu:ubuntu --chmod=755 docker/workbench/cosmos3/smoke_functional.sh /opt/cosmos3/smoke_functional.sh

# Distribution guard: fail the build if any model weight was baked into a layer.
RUN baked="$(find /opt/cosmos3 /opt/npa /tmp/hf_home -type f \
        \( -name '*.safetensors' -o -name '*.ckpt' -o -name '*.pth' -o -name '*.pt' \) \
        -size +50M 2>/dev/null | head -5)" \
    && if [ -n "${baked}" ]; then \
         echo "ERROR: model weights baked into image (must download at runtime):" >&2; \
         echo "${baked}" >&2; \
         exit 1; \
       fi

WORKDIR /workspace

ENTRYPOINT ["npa", "workbench", "cosmos3"]
CMD ["generate", "--help"]
