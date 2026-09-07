# CUDA 13.0.1 is paired with the PyTorch 2.9.0 cu130 wheel index.
# The Docker Hub tag uses "cudnn" rather than the non-existent "cudnn9" suffix.
ARG CUDA_BASE_TAG=13.0.1-cudnn-devel-ubuntu22.04
FROM nvidia/cuda:${CUDA_BASE_TAG}

ARG CUDA_BASE_TAG=13.0.1-cudnn-devel-ubuntu22.04
ARG BUILD_TS=unknown
ARG TORCH_VERSION=2.9.0
ARG TORCHVISION_VERSION=0.24.0
ARG TORCHAUDIO_VERSION=2.9.0
ARG FLASH_ATTN_COMMIT=0409f9adcbdebff6cc19eb95f370d40e896980bc
# flash-attn-4 requires nvidia-cutlass-dsl>=4.4.2 and quack-kernels>=0.4.0, both
# unbounded, so pip resolves to whatever is newest. CUTLASS DSL 4.6.x dropped
# cutlass.cute.core.ThrMma, which this flash-attn commit still references, so an
# unpinned build installs successfully and then fails on `import flash_attn`
# with "module 'cutlass.cute.core' has no attribute 'ThrMma'". 4.5.3 is the
# newest DSL that still exposes it; quack 0.5.0 is the newest quack that accepts
# a 4.5.x DSL (0.5.3+ pin an exact 4.6 release). Bump all three together.
ARG CUTLASS_DSL_VERSION=4.5.3
ARG QUACK_KERNELS_VERSION=0.5.0
# SASS is per-arch with no cross-major compatibility, so every architecture a
# source-compiled extension must run on has to be listed here explicitly:
#   8.0  = A100 (sm_80)          9.0  = H100/H200 (sm_90)
#   10.0 = B200 (sm_100)         10.3 = B300 / Blackwell Ultra (sm_103)
#   12.0 = RTX PRO 6000 (sm_120)
# Minor-version forward compatibility holds inside a major, so sm_100 SASS also
# runs on sm_103 - but sm_120 (major 12) and sm_100/sm_103 (major 10) are
# mutually incompatible, which is why both majors are listed.
ARG TORCH_CUDA_ARCH_LIST="8.0 9.0 10.0 10.3 12.0"
# Space-separated torch arch tokens (sm_XX) that torch.cuda.get_arch_list() must
# report for the build to succeed. Empty disables the assertion.
ARG REQUIRE_TORCH_ARCHS="sm_80 sm_90 sm_100 sm_120"

LABEL npa.image="cuda13-b300" \
      npa.cuda.base="${CUDA_BASE_TAG}" \
      npa.torch.version="${TORCH_VERSION}+cu130" \
      npa.flash_attn_4.commit="${FLASH_ATTN_COMMIT}" \
      npa.cutlass_dsl.version="${CUTLASS_DSL_VERSION}" \
      npa.quack_kernels.version="${QUACK_KERNELS_VERSION}" \
      npa.cuda_arch_list="${TORCH_CUDA_ARCH_LIST}" \
      npa.build_ts="${BUILD_TS}" \
      npa.build_info="cuda=${CUDA_BASE_TAG};torch=${TORCH_VERSION}+cu130;torchvision=${TORCHVISION_VERSION}+cu130;torchaudio=${TORCHAUDIO_VERSION}+cu130;flash_attn_4_commit=${FLASH_ATTN_COMMIT};cutlass_dsl=${CUTLASS_DSL_VERSION};quack_kernels=${QUACK_KERNELS_VERSION};torch_cuda_arch_list=${TORCH_CUDA_ARCH_LIST}"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CUDA_HOME=/usr/local/cuda \
    HF_HOME=/opt/npa/hf_cache \
    HF_HUB_CACHE=/opt/npa/hf_cache/hub \
    TRANSFORMERS_CACHE=/opt/npa/hf_cache/transformers \
    NPA_INPUT_PATH=/input \
    NPA_OUTPUT_PATH=/output \
    INPUT_PATH=/input \
    OUTPUT_PATH=/output \
    IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg \
    PATH=/opt/npa/venv/bin:/usr/local/cuda/bin:$PATH

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      gnupg \
      software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      ffmpeg \
      git-lfs \
      libegl1 \
      libffi-dev \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libssl-dev \
      libxext6 \
      libxrender1 \
      ninja-build \
      pkg-config \
      python3.11 \
      python3.11-dev \
      python3.11-venv \
      wget \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash -u 1000 ubuntu \
    && install -d -m 0755 -o ubuntu -g ubuntu \
      /input \
      /output \
      /opt/npa \
      /opt/npa/hf_cache \
      /opt/npa/src

RUN python3.11 -m venv /opt/npa/venv \
    && python -m pip install --upgrade pip setuptools wheel packaging

RUN python -m pip install \
      torch==${TORCH_VERSION} \
      torchvision==${TORCHVISION_VERSION} \
      torchaudio==${TORCHAUDIO_VERSION} \
      --index-url https://download.pytorch.org/whl/cu130

RUN python -m pip install \
      accelerate \
      av \
      boto3 \
      datasets \
      einops \
      huggingface_hub \
      numpy \
      opencv-python-headless \
      pandas \
      pillow \
      pyarrow \
      safetensors \
      scipy \
      tensorboard \
      transformers \
      wandb \
    && find /opt/npa/venv -type f \
      -path '*/imageio_ffmpeg/binaries/ffmpeg*' -delete \
    && test -z "$(find /opt/npa/venv -type f \
      -path '*/imageio_ffmpeg/binaries/ffmpeg*' -print -quit)" \
    && python -c 'import importlib.util; assert importlib.util.find_spec("imageio_ffmpeg") is None or __import__("imageio_ffmpeg").get_ffmpeg_exe() == "/usr/bin/ffmpeg"'

ARG CUTLASS_DSL_VERSION
ARG QUACK_KERNELS_VERSION
RUN git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention \
    && cd /tmp/flash-attention \
    && git checkout "${FLASH_ATTN_COMMIT}" \
    && python -m pip install \
      "nvidia-cutlass-dsl[cu13]==${CUTLASS_DSL_VERSION}" \
      "quack-kernels==${QUACK_KERNELS_VERSION}" \
      "flash_attn/cute[cu13]" \
    && rm -rf /tmp/flash-attention

# flash-attn-4 exposes flash_attn.cute. Provide the legacy root import used by
# existing smoke scripts while still routing calls to the FA4 CuTe implementation.
RUN python - <<'PY'
from importlib import metadata
from pathlib import Path
import site

site_dir = Path(site.getsitepackages()[0])
pkg = site_dir / "flash_attn"
pkg.mkdir(exist_ok=True)
(pkg / "__init__.py").write_text(
    "from importlib import metadata as _metadata\n"
    "__version__ = _metadata.version('flash-attn-4')\n"
    "from flash_attn.cute import flash_attn_func as _cute_flash_attn_func\n"
    "from flash_attn.cute import flash_attn_varlen_func as _cute_flash_attn_varlen_func\n"
    "def _tensor_result(result):\n"
    "    return result[0] if isinstance(result, tuple) else result\n"
    "def flash_attn_func(*args, **kwargs):\n"
    "    return _tensor_result(_cute_flash_attn_func(*args, **kwargs))\n"
    "def flash_attn_varlen_func(*args, **kwargs):\n"
    "    return _tensor_result(_cute_flash_attn_varlen_func(*args, **kwargs))\n"
    "__all__ = ['flash_attn_func', 'flash_attn_varlen_func']\n",
    encoding="utf-8",
)
print("flash-attn-4", metadata.version("flash-attn-4"))
PY

# Baked in so every derived image can validate itself on whatever GPU it lands
# on, without the caller having to inject a script.
COPY scripts/check_torch_gpu_arch.py /npa/check_torch_gpu_arch.py
COPY scripts/gpu_capability_smoke.py /npa/gpu_capability_smoke.py

ARG REQUIRE_TORCH_ARCHS
RUN python /npa/check_torch_gpu_arch.py $(printf -- '--require-arch %s ' ${REQUIRE_TORCH_ARCHS}) \
    && python - <<'PY'
import importlib.metadata as metadata

import flash_attn
from flash_attn import flash_attn_func

# Import only: the flash-attn-4 CuTe kernel needs a GPU with TMA, so whether it
# actually runs is decided by /npa/gpu_capability_smoke.py at deploy time.
print("nvidia-nccl-cu13", metadata.version("nvidia-nccl-cu13"))
print("flash-attn-4", flash_attn.__version__)
print("flash_attn_func", flash_attn_func)
PY

WORKDIR /workspace

USER root

RUN dpkg --purge --force-depends linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*

USER ubuntu
