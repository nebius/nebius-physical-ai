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

ARG COSMOS_CURATE_VERSION=0.1.2
# Upstream commit this image was validated against; bump together with the tag.
ARG COSMOS_CURATE_REF=de89c5e4f76219c3f2985dc8fda790a74ab8bab0
ARG COSMOS_XENNA_VERSION=0.5.5
ARG MICROMAMBA_VERSION=2.0.5
# Checksum of the linux-64 tarball for MICROMAMBA_VERSION. Piping curl into tar would
# execute whatever the endpoint served; verify it first and bump both together.
ARG MICROMAMBA_SHA256=ff1a10b46e9ee673bb672ac464d4787b4e9c3569b75816a56f76dddc0b1f07fc
ARG FFMPEG_VERSION=7.1.1
ARG TORCH_VERSION=2.9.0

LABEL npa.tool="cosmos-curate" \
      npa.version="${COSMOS_CURATE_VERSION}" \
      npa.model_weights="none-baked-in; fetched at runtime with HF_TOKEN" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

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

# openssh-server, rsync, and sudo are for SkyPilot's Kubernetes provisioner, whose
# per-pod setup runs `$(prefix_cmd) apt install openssh-server rsync -y` and
# `service ssh restart`, where prefix_cmd is `sudo` for a non-root user (see
# sky/provision/kubernetes/instance.py). Without them the setup script fails, the
# container exits, and SkyPilot reports the misleading `container not found
# ("ray-node")`. Pre-installing them also makes that step a no-op instead of an
# apt round trip on every pod.
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      ca-certificates \
      curl \
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

# SkyPilot's Kubernetes provisioner gates on `sudo -l` and then runs, as this image's
# own user, `sudo apt install`, `sudo sed -i` on /etc/ssh/sshd_config and
# /etc/pam.d/sshd, `sudo ssh-keygen -A`, `sudo chown`/`chmod` on ~/.ssh, `sudo cat` of
# the mounted key, and `sudo service ssh restart`. Enumerating those commands in
# sudoers would look narrower without being narrower — arbitrary `sudo sed -i`,
# `sudo cat`, and `sudo chown` already amount to root — and would break provisioning
# the moment SkyPilot adds one more. The grant is blanket and the reason recorded.
RUN useradd -m -s /bin/bash -u 1000 ubuntu \
    && echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu \
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
      -o micromamba.tar.bz2; \
    echo "${MICROMAMBA_SHA256}  micromamba.tar.bz2" | sha256sum -c -; \
    tar -xj -f micromamba.tar.bz2 bin/micromamba; \
    rm -f micromamba.tar.bz2; \
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
    # Record what was checked out before .git goes away. npa's probe compares this to
    # the commit its stage-construction code was written against, so bumping the pin
    # without revisiting that code fails at the boundary rather than as a TypeError
    # from inside an upstream constructor.
    git rev-parse HEAD > .npa-upstream-revision; \
    rm -rf .git; \
    test -d cosmos_curator/pipelines/video; \
    test -f cosmos_curator/configs/all_models.json; \
    test "$(cat .npa-upstream-revision)" = "${COSMOS_CURATE_REF}"

# The dependency set upstream's GPU-free stages import. torch is CPU-only here:
# the motion stage runs its CPU code path when no GPU is requested, and the CUDA
# wheels would add several GB for kernels this image never calls.
#
# Every version is pinned to what the verified image resolved. Unpinned, the same
# Dockerfile builds a different image every day and a bad upstream release lands
# without anyone choosing it — and this image's whole job is to be the one place the
# curator's environment is known good.
RUN python -m venv "${COSMOS_CURATE_VENV}" \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
      "torch==${TORCH_VERSION}" \
      --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir \
      "cosmos-xenna==${COSMOS_XENNA_VERSION}" \
      "attrs==26.1.0" \
      "av==18.0.0" \
      "azure-identity==1.25.3" \
      "azure-storage-blob==12.30.0" \
      "boto3==1.43.61" \
      "huggingface_hub==1.26.0" \
      "loguru==0.7.3" \
      "ngcsdk==4.34.7" \
      "numpy==2.5.1" \
      "nvidia-ml-py==13.610.43" \
      "nvtx==0.2.15" \
      "opencv-python-headless==5.0.0.93" \
      "pandas==3.0.5" \
      "pyarrow==25.0.0" \
      "pydantic==2.13.4" \
      "pylance==9.0.0" \
      "tqdm==4.70.0" \
      "webdataset==1.0.2"

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

# The runtime mount being empty proves nothing about the layers: a weight committed
# into upstream's tree, or pulled by an LFS smudge that slipped through, lands in the
# baked checkout instead. Scan there too, the way the evaluator image does.
WEIGHT_SUFFIXES = {".onnx", ".safetensors", ".pth", ".pt", ".bin", ".ckpt", ".gguf"}
source = Path(availability.source)
in_source = sorted(
    str(path.relative_to(source))
    for path in source.rglob("*")
    if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES
)
if in_source:
    raise RuntimeError(
        f"model weights must never be baked into this image; found {in_source} under {source}"
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
