# NVIDIA Cosmos Curator workbench image.
#
# Upstream (https://github.com/nvidia-cosmos/cosmos-curate, Apache-2.0) ships its
# curation pipelines inside its own pixi-managed container and publishes no wheel
# for them. This image instead bakes an upstream source checkout plus the subset
# of its dependencies the GPU-free stages need, so `npa workbench cosmos-curate`
# can drive upstream's real stage classes in-process:
#
#   VideoDownloader -> FixedStrideExtractorStage -> ClipTranscodingStage
#   -> MotionVectorDecodeStage -> MotionFilterStage -> ClipWriterStage
#
# NO MODEL WEIGHTS ARE BAKED IN. Those stages need none — they are ffmpeg and
# OpenCV work. The curator's GPU stages (TransNetV2 shot detection, InternVideo2 /
# Cosmos-Embed1 embeddings, CLIP + aesthetic head, Qwen VLM captioning) do need
# weights, and those are third-party and NVIDIA models under their own licenses.
# They are downloaded at RUN time into ${NPA_COSMOS_CURATE_WEIGHTS_DIR} with the
# operator's own Hugging Face token:
#
#   docker run -e HF_TOKEN=... -v curator-weights:/config/models <image> fetch-models
#
# The model ids and their pinned revisions come from upstream's own registry
# (cosmos_curator/configs/all_models.json), and the download is upstream's own
# huggingface_hub call. A build-time check asserts the weights directory is empty,
# so a weight can never be redistributed inside this image by accident.
#
# Two other build details matter:
#
#  * Python 3.12 — upstream requires >=3.12,<3.13.
#  * ffmpeg with libopenh264 — upstream's ClipTranscodingStage accepts only
#    `libopenh264` (CPU) or `h264_nvenc` (GPU), and Debian's ffmpeg is built with
#    neither. Conda-forge's LGPL ffmpeg build carries libopenh264, which is also
#    how upstream's own pixi environment gets it.
#
# Digest pinned 2026-07-30. Tag: python:3.12-slim-bookworm
FROM python:3.12-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b

ARG COSMOS_CURATE_VERSION=0.1.0
# Upstream commit this image was validated against; bump together with the tag.
ARG COSMOS_CURATE_REF=de89c5e4f76219c3f2985dc8fda790a74ab8bab0
ARG COSMOS_XENNA_VERSION=0.5.5
ARG MICROMAMBA_VERSION=2.0.5
ARG FFMPEG_VERSION=7.1.1
ARG TORCH_VERSION=2.9.0

LABEL npa.tool="cosmos-curate" \
      npa.version="${COSMOS_CURATE_VERSION}" \
      npa.model_weights="none-baked-in; fetched at runtime with HF_TOKEN"

# CODE_DIR is upstream's own in-container code directory
# (cosmos_curator.core.utils.environment.CONTAINER_PATHS_CODE_DIR) and WORKSPACE is
# its workspace root, so upstream's internal path resolution works unmodified.
ENV DEBIAN_FRONTEND=noninteractive \
    COSMOS_CURATE_CODE_DIR=/opt/cosmos-curator \
    COSMOS_CURATE_VENV=/opt/cosmos-curate/venv \
    COSMOS_CURATOR_LOCAL_DOCKER_JOB=1 \
    FFMPEG_PREFIX=/opt/ffmpeg \
    NPA_COSMOS_CURATE_SRC=/opt/cosmos-curator \
    NPA_COSMOS_CURATE_WEIGHTS_DIR=/config/models \
    PATH=/opt/cosmos-curate/venv/bin:/opt/ffmpeg/bin:$PATH \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      ca-certificates \
      curl \
      git \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash -u 1000 ubuntu \
    && install -d -m 0755 -o ubuntu -g ubuntu \
      /config \
      /config/models \
      /config/tmp \
      /opt/cosmos-curate \
      /opt/cosmos-curate/venv \
      /opt/cosmos-curator \
      /opt/npa/src/npa \
      /opt/npa/docker/workbench/cosmos-curate

# ffmpeg + ffprobe with libopenh264, from conda-forge's LGPL build. The prefix must
# not exist yet: micromamba refuses to create an environment over a non-conda
# directory, so FFMPEG_PREFIX is deliberately absent from the install -d list above.
RUN set -eux; \
    cd /tmp; \
    curl -sSL "https://micro.mamba.pm/api/micromamba/linux-64/${MICROMAMBA_VERSION}" \
      | tar -xvj bin/micromamba; \
    ./bin/micromamba create -y -p "${FFMPEG_PREFIX}" -c conda-forge \
      "ffmpeg==${FFMPEG_VERSION}" "openh264"; \
    ./bin/micromamba clean -y --all; \
    rm -rf /tmp/bin; \
    "${FFMPEG_PREFIX}/bin/ffmpeg" -hide_banner -encoders | grep -q libopenh264; \
    "${FFMPEG_PREFIX}/bin/ffprobe" -version | head -n 1; \
    chown -R ubuntu:ubuntu "${FFMPEG_PREFIX}"

USER ubuntu
WORKDIR /opt/cosmos-curate

# Upstream source checkout, pinned to COSMOS_CURATE_REF. GIT_LFS_SKIP_SMUDGE keeps
# any large-file objects out of the image; only the pipeline source is needed.
RUN set -eux; \
    git init -q "${COSMOS_CURATE_CODE_DIR}"; \
    cd "${COSMOS_CURATE_CODE_DIR}"; \
    git remote add origin https://github.com/nvidia-cosmos/cosmos-curate.git; \
    GIT_LFS_SKIP_SMUDGE=1 git fetch -q --depth 1 origin "${COSMOS_CURATE_REF}"; \
    GIT_LFS_SKIP_SMUDGE=1 git checkout -q FETCH_HEAD; \
    rm -rf .git; \
    test -d cosmos_curator/pipelines/video; \
    test -f cosmos_curator/configs/all_models.json

# The dependency set upstream's GPU-free stages import. torch is CPU-only here:
# the motion stage runs its CPU code path when no GPU is requested, and the CUDA
# wheels would add several GB for kernels this image never calls.
RUN python -m venv "${COSMOS_CURATE_VENV}" \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
      "torch==${TORCH_VERSION}" \
      --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir \
      "cosmos-xenna==${COSMOS_XENNA_VERSION}" \
      "attrs" \
      "av" \
      "azure-identity" \
      "azure-storage-blob" \
      "boto3" \
      "huggingface_hub" \
      "loguru" \
      "ngcsdk" \
      "numpy" \
      "nvidia-ml-py" \
      "nvtx" \
      "opencv-python-headless" \
      "pandas" \
      "pyarrow" \
      "pydantic" \
      "pylance" \
      "tqdm" \
      "webdataset"

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu docker/workbench/cosmos-curate/smoke_functional.py /opt/npa/docker/workbench/cosmos-curate/smoke_functional.py
COPY --chown=ubuntu:ubuntu docker/workbench/cosmos-curate/entrypoint.sh /opt/npa/docker/workbench/cosmos-curate/entrypoint.sh

RUN python -m pip install --no-cache-dir -e /opt/npa \
    && python -m py_compile /opt/npa/docker/workbench/cosmos-curate/smoke_functional.py \
    && chmod +x /opt/npa/docker/workbench/cosmos-curate/entrypoint.sh

# Build-time proof that the real upstream stages import, that the transcoding
# encoder this image relies on is present, that the model registry resolves, and
# that NO model weights were baked in.
RUN python - <<'PY'
import os
from pathlib import Path

from npa.workbench.cosmos_curate import MODEL_SETS, probe_availability, resolve_models, weights_dir

availability = probe_availability()
if not availability.can_run_in_process:
    raise RuntimeError(f"cosmos-curate image cannot run the curator: {availability.reason()}")

specs = resolve_models([])
if not specs:
    raise RuntimeError("upstream model registry resolved no models for the default set")

root = weights_dir()
baked = [str(path) for path in root.rglob("*") if path.is_file()] if root.is_dir() else []
if baked:
    raise RuntimeError(
        f"model weights must never be baked into this image; found {len(baked)} file(s) under {root}"
    )
print(
    "COSMOS_CURATE_BUILD_OK "
    f"source={availability.source} encoder={availability.encoder} "
    f"model_sets={len(MODEL_SETS)} default_set={[spec.key for spec in specs]} weights_dir={root} (empty)"
)
PY

# Weights and scratch live here; mount a volume to persist downloads across runs.
VOLUME ["/config/models"]
WORKDIR /opt/cosmos-curate
ENTRYPOINT ["/opt/npa/docker/workbench/cosmos-curate/entrypoint.sh"]
CMD ["engine"]
