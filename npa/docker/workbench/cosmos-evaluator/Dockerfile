# NVIDIA Cosmos Evaluator workbench image.
#
# Upstream (https://github.com/nvidia-cosmos/cosmos-evaluator, Apache-2.0) builds
# with Bazel and publishes no wheel, so this image bakes a pinned source checkout
# and the dependencies its checkers import. `npa workbench cosmos-evaluator` then
# runs upstream's own HallucinationProcessor rather than a reimplementation.
#
# MODEL WEIGHTS AND WHY THIS IMAGE NEEDS NONE:
#
#  * Hallucination check — no weights at all. It is classical computer vision:
#    per-frame dynamic masks from frame differencing, then a distance transform.
#  * Attribute verification check — no local weights. Upstream calls an
#    OpenAI-compatible endpoint for question generation and answering; NPA points
#    it at Nebius Token Factory, so the credential it needs at run time is
#    NEBIUS_TOKEN_FACTORY_KEY, not a model download.
#  * Objects/obstacle check — DOES need weights (a SegFormer ONNX and a CWIP
#    checkpoint), and upstream ships them as Git-LFS objects behind an EULA. This
#    image runs the checkout with GIT_LFS_SKIP_SMUDGE=1, so those objects are never
#    fetched or redistributed here, and that check is not wired into NPA. Anyone
#    who needs it must accept upstream's EULA and fetch the weights themselves.
#
# A build-time check asserts no LFS payload and no weight file landed in the image.
#
# Digest pinned 2026-07-30. Tag: python:3.12-slim-bookworm
FROM python:3.12-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b

ARG COSMOS_EVALUATOR_VERSION=0.1.2
# Upstream commit this image was validated against; bump together with the tag.
ARG COSMOS_EVALUATOR_REF=e901afea61d70d2cfe94817c8a757989b47ebee4
ARG OPENCV_VERSION=4.13.0.90
ARG NUMPY_VERSION=2.3.5

LABEL npa.tool="cosmos-evaluator" \
      npa.version="${COSMOS_EVALUATOR_VERSION}" \
      npa.model_weights="none; the wired checks are CV + a hosted VLM endpoint" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

ENV DEBIAN_FRONTEND=noninteractive \
    COSMOS_EVALUATOR_VENV=/opt/cosmos-evaluator/venv \
    NPA_COSMOS_EVALUATOR_SRC=/opt/cosmos-evaluator/src \
    PATH=/opt/cosmos-evaluator/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

# ffmpeg decodes the clips under test and extracts the frame the VLM answers about.
# Debian's build carries libx264, which is all this image encodes with (the smoke's
# synthetic clip); the curator's libopenh264 requirement does not apply here.
#
# openssh-server, rsync, and sudo are for SkyPilot's Kubernetes provisioner, whose
# per-pod setup runs `$(prefix_cmd) apt install openssh-server rsync -y` and
# `service ssh restart`, where prefix_cmd is `sudo` for a non-root user (see
# sky/provision/kubernetes/instance.py). Without them the setup script fails, the
# container exits, and SkyPilot reports the misleading `container not found
# ("ray-node")`. Pre-installing them also makes that step a no-op instead of an
# apt round trip on every pod.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      ffmpeg \
      git \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      netcat-openbsd \
      openssh-server \
      procps \
      rsync \
      sudo \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash -u 1000 ubuntu \
    && echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu \
    && install -d -m 0755 -o ubuntu -g ubuntu \
      /opt/cosmos-evaluator \
      /opt/cosmos-evaluator/src \
      /opt/cosmos-evaluator/venv \
      /opt/npa/src/npa \
      /opt/npa/docker/workbench/cosmos-evaluator

USER ubuntu
WORKDIR /opt/cosmos-evaluator

# Upstream source checkout, pinned. GIT_LFS_SKIP_SMUDGE keeps the EULA-gated model
# objects (SegFormer ONNX, CWIP checkpoint) as pointer files: the code is
# Apache-2.0 and redistributable, those weights are not.
RUN set -eux; \
    git init -q "${NPA_COSMOS_EVALUATOR_SRC}"; \
    cd "${NPA_COSMOS_EVALUATOR_SRC}"; \
    git remote add origin https://github.com/nvidia-cosmos/cosmos-evaluator.git; \
    GIT_LFS_SKIP_SMUDGE=1 git fetch -q --depth 1 origin "${COSMOS_EVALUATOR_REF}"; \
    GIT_LFS_SKIP_SMUDGE=1 git checkout -q FETCH_HEAD; \
    rm -rf .git checks/sample_data; \
    find . -type d -name data -path '*tests*' -prune -exec rm -rf {} + ; \
    test -f checks/hallucination/processor.py; \
    test -f checks/cosmos_evaluator.yaml

# What upstream's hallucination and attribute-verification checkers import.
# bazel-runfiles supplies the `runfiles` module its config loader reaches for;
# NPA aliases it under the Bazel module path so the checkers import outside Bazel.
RUN python -m venv "${COSMOS_EVALUATOR_VENV}" \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
      "bazel-runfiles==2.2.0" \
      "multi-storage-client[boto3]==1.0.1" \
      "numpy==${NUMPY_VERSION}" \
      "openai==2.52.0" \
      "opencv-python-headless==${OPENCV_VERSION}" \
      "pydantic==2.13.4" \
      "python-dotenv==1.2.2" \
      "pyyaml==6.0.3"

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu workflows /opt/npa/workflows
COPY --chown=ubuntu:ubuntu docker/workbench/cosmos-evaluator/smoke_functional.py /opt/npa/docker/workbench/cosmos-evaluator/smoke_functional.py
COPY --chown=ubuntu:ubuntu docker/workbench/cosmos-evaluator/entrypoint.sh /opt/npa/docker/workbench/cosmos-evaluator/entrypoint.sh

RUN python -m pip install --no-cache-dir -e /opt/npa \
    && python -m py_compile /opt/npa/docker/workbench/cosmos-evaluator/smoke_functional.py \
    && chmod +x /opt/npa/docker/workbench/cosmos-evaluator/entrypoint.sh

# Build-time proof that upstream's checker is the engine this image resolves to,
# and that no model weight or LFS payload was baked in.
RUN python - <<'PY'
from pathlib import Path

from npa.workbench.cosmos_evaluator.evaluate import evaluator_engine_summary
from npa.workbench.cosmos_evaluator.upstream import ensure_upstream_importable

summary = evaluator_engine_summary()
if summary["engine"] != "cosmos-evaluator-upstream":
    raise RuntimeError(f"expected the upstream engine, resolved {summary}")

ensure_upstream_importable()
from checks.hallucination.processor import HallucinationProcessor  # noqa: E402
from checks.utils.frame_processing import compute_dynamic_mask  # noqa: E402

root = Path(summary["upstream_source"])
weights = sorted(
    str(path.relative_to(root))
    for pattern in ("*.onnx", "*.safetensors", "*.pth", "*.pt", "*.bin", "*.ckpt")
    for path in root.rglob(pattern)
    if path.is_file() and path.stat().st_size > 1_000_000
)
if weights:
    raise RuntimeError(f"model weights must never be baked into this image; found {weights}")
print(f"COSMOS_EVALUATOR_BUILD_OK source={root} upstream_processor={HallucinationProcessor.__name__}")
PY

WORKDIR /opt/cosmos-evaluator
ENTRYPOINT ["/opt/npa/docker/workbench/cosmos-evaluator/entrypoint.sh"]
CMD ["engine"]
