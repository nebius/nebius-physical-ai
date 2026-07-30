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
# Two build details matter:
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
      npa.version="${COSMOS_CURATE_VERSION}"

ENV DEBIAN_FRONTEND=noninteractive \
    COSMOS_CURATE_HOME=/opt/cosmos-curate \
    COSMOS_CURATE_VENV=/opt/cosmos-curate/venv \
    FFMPEG_PREFIX=/opt/ffmpeg \
    NPA_COSMOS_CURATE_SRC=/opt/cosmos-curate/src \
    COSMOS_CURATOR_LOCAL_DOCKER_JOB=1 \
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
    &&     install -d -m 0755 -o ubuntu -g ubuntu \
      /opt/cosmos-curate \
      /opt/cosmos-curate/src \
      /opt/cosmos-curate/venv \
      /opt/ffmpeg \
      /opt/npa/src/npa \
      /opt/npa/docker/workbench/cosmos-curate

# ffmpeg + ffprobe with libopenh264, from conda-forge's LGPL build.
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

# Upstream source checkout, pinned to COSMOS_CURATE_REF. Only the pipeline
# packages are needed; the pixi environment and container tooling are not.
RUN set -eux; \
    git init -q "${NPA_COSMOS_CURATE_SRC}"; \
    cd "${NPA_COSMOS_CURATE_SRC}"; \
    git remote add origin https://github.com/nvidia-cosmos/cosmos-curate.git; \
    git fetch -q --depth 1 origin "${COSMOS_CURATE_REF}"; \
    git checkout -q FETCH_HEAD; \
    rm -rf .git; \
    test -d cosmos_curator/pipelines/video

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
      "webdataset"

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu docker/workbench/cosmos-curate/smoke_functional.py /opt/npa/docker/workbench/cosmos-curate/smoke_functional.py

RUN python -m pip install --no-cache-dir -e /opt/npa \
    && python -m py_compile /opt/npa/docker/workbench/cosmos-curate/smoke_functional.py

# Build-time proof that the real upstream stages import and that the transcoding
# encoder this image relies on is actually present.
RUN python - <<'PY'
from npa.workbench.cosmos_curate import probe_availability

availability = probe_availability()
if not availability.can_run_in_process:
    raise RuntimeError(f"cosmos-curate image cannot run the curator: {availability.reason()}")
print(f"COSMOS_CURATE_BUILD_OK source={availability.source} encoder={availability.encoder}")
PY

WORKDIR /opt/cosmos-curate
ENTRYPOINT ["/bin/bash"]
