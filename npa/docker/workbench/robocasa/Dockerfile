# npa-robocasa: RoboCasa kitchen-task simulation as a first-class workbench tool.
#
# Built independently on the CUDA 12.4 base that the live-accepted RoboCasa BYOF
# evidence used (nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04), NOT on the
# purpose-built SONIC MuJoCo application image. RoboCasa and robosuite are
# Apache-2.0; the pinned versions below match the accepted BYOF capability runs.
#
# What IS baked, and is fine to redistribute:
#   * CUDA 12.4 Ubuntu 22.04 base
#   * RoboCasa source at the pinned v1.0 tag (Apache-2.0)
#   * robosuite at the pinned commit (Apache-2.0)
#   * MuJoCo 3.3.1, Gymnasium 0.29.1, and the OSS Python closure
#
# What is NOT baked:
#   * Kitchen assets (textures, fixtures, objects) — downloaded at run time by
#     the documented download_kitchen_assets script under the operator's own
#     network access. No model weights are baked.
#   * Any credentials, tokens, or EULA acceptance.
ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
FROM --platform=linux/amd64 ${BASE_IMAGE}

ARG ROBOCASA_REPO_URL=https://github.com/robocasa/robocasa.git
ARG ROBOCASA_REPO_REF=v1.0
ARG ROBOSUITE_COMMIT=85abee228d1c43ab1939bce33028099945d453b4
ARG ROBOCASA_VERSION=0.1.0

LABEL npa.tool="robocasa" \
      npa.version="${ROBOCASA_VERSION}" \
      npa.base_image="${BASE_IMAGE}" \
      npa.runtime_user="ubuntu" \
      npa.cuda_architectures="sm80,sm90,sm100,sm103,sm120"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    ROBOCASA_PORT=8791 \
    PYTHONPATH=/app \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    PATH=/opt/robocasa/venv/bin:$PATH

USER root

# System deps for EGL rendering and the Python build toolchain.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python3-venv build-essential git \
        software-properties-common ca-certificates curl \
        fuse netcat-openbsd pciutils procps wget \
        libegl1 libgl1 libglvnd0 libegl1-mesa-dev \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/robocasa/venv

WORKDIR /opt/robocasa

# Pin the exact dependency closure from the live-accepted BYOF evidence.
# RoboCasa requires Gymnasium 0.29.1. LeRobot's broad environment dependency
# requests Gymnasium >=1.1.1, but exact-checkpoint ACT inference uses neither
# LeRobot environment wrappers nor that newer API. Keep RoboCasa's known-good
# pin and install only the LeRobot policy package here.
RUN python -m pip install --no-cache-dir --upgrade pip "setuptools>=71,<81" wheel \
    && python -m pip install --no-cache-dir \
        "git+https://github.com/ARISE-Initiative/robosuite.git@${ROBOSUITE_COMMIT}" \
    && python -m pip install --no-cache-dir \
        numpy==2.2.5 numba==0.61.2 scipy==1.15.3 mujoco==3.3.1 pygame Pillow \
        "opencv-python-headless>=4.9,<4.14" pyyaml pynput tqdm termcolor "imageio[ffmpeg]" \
        h5py lxml hidapi "gymnasium==0.29.1" huggingface_hub \
        "fastapi==0.136.1" "uvicorn[standard]==0.38.0" "pydantic==2.13.4" \
        "boto3==1.42.9" "httpx>=0.27" \
        "datasets>=4.0.0,<5.0.0" "accelerate>=1.10.0,<2.0.0" \
        "packaging>=24.2,<26.0" "einops>=0.8.0,<0.9.0" \
        "draccus==0.10.0" "safetensors>=0.4.3,<1.0.0" \
        "av>=15.0.0,<16.0.0" "jsonlines>=4.0.0,<5.0.0" \
        "deepdiff>=7.0.1,<9.0.0" "diffusers>=0.27.2,<0.36.0" \
        "pyserial>=3.5,<4.0" \
    && python -m pip install --no-cache-dir --no-deps "lerobot==0.5.1" \
    && python -m pip install --no-cache-dir --upgrade \
        "torch==2.12.1" "torchvision==0.27.1"

# Install RoboCasa source (no deps; the closure above is pinned).
RUN git clone --depth 1 --branch ${ROBOCASA_REPO_REF} \
      ${ROBOCASA_REPO_URL} /opt/robocasa/source \
    && python -m pip install --no-cache-dir --no-deps -e /opt/robocasa/source

# Fail the image build if the exact held-out evaluator import path is incomplete.
RUN python -c "import gymnasium; assert gymnasium.__version__ == '0.29.1'; from lerobot.policies.act.modeling_act import ACTPolicy; from lerobot.configs.policies import PreTrainedConfig; from lerobot.policies.factory import make_pre_post_processors"

# Copy the shared workbench implementation into the image.
RUN mkdir -p /app/npa/workbench
RUN touch /app/npa/__init__.py /app/npa/workbench/__init__.py
COPY src/npa/workbench/robocasa /app/npa/workbench/robocasa
COPY src/npa/smoke/test_robocasa_functional.py /app/smoke_functional.py

RUN id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu \
    && chown -R ubuntu:ubuntu /app /opt/robocasa

EXPOSE 8791

USER ubuntu

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python3 -c "import os,urllib.request; p=os.environ.get('ROBOCASA_PORT','8791'); urllib.request.urlopen(f'http://127.0.0.1:{p}/health', timeout=3)" || exit 1

CMD ["uvicorn", "npa.workbench.robocasa.service:app", "--host", "0.0.0.0", "--port", "8791"]
