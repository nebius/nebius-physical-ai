# npa-robocasa: RoboCasa kitchen-task simulation as a first-class workbench tool.
#
# Built independently on the exact CUDA 12.4 base that the live-accepted
# RoboCasa BYOF evidence used, NOT on the purpose-built SONIC MuJoCo application
# image. RoboCasa and robosuite are Apache-2.0; their immutable revisions and
# the reviewed Python closure below match the accepted capability path.
#
# What IS baked, and is fine to redistribute:
#   * CUDA 12.4 Ubuntu 22.04 base
#   * RoboCasa source at the pinned commit (Apache-2.0)
#   * robosuite at the pinned commit (Apache-2.0)
#   * MuJoCo 3.3.1, Gymnasium 0.29.1, and the OSS Python closure
#
# What is NOT baked:
#   * Kitchen assets (textures, fixtures, objects) — downloaded at run time by
#     the documented download_kitchen_assets script under the operator's own
#     network access. No model weights are baked.
#   * Any credentials, tokens, or EULA acceptance.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04@sha256:0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4

ARG ROBOCASA_VERSION=0.1.1
ARG NPA_SOURCE_SHA
ARG LEROBOT_SOURCE_COMMIT=7e241bd630a3719a56157a497ce5d08f244784f1

LABEL npa.tool="robocasa" \
      npa.version="${ROBOCASA_VERSION}" \
      npa.base_image="nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04@sha256:0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4" \
      npa.robocasa.revision="8f3c96ec8d1bfcd8126cad2bca887da98d30e997" \
      npa.robosuite.revision="85abee228d1c43ab1939bce33028099945d453b4" \
      npa.lerobot.revision="${LEROBOT_SOURCE_COMMIT}" \
      npa.lerobot.derivative="0.6.1+npa1" \
      npa.lerobot.archive.sha256="869026b70a9488f11ae25c92573f68a03e04e45b4ffda7d63586582ac8d506ee" \
      npa.lerobot.patch.sha256="7d90538cc6c66351256f394d555797a6aacf34cb17dad2d0c9fc781c7e22404c" \
      npa.runtime_user="ubuntu" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.opencontainers.image.base.name="nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04" \
      org.opencontainers.image.base.digest="sha256:0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4"

ENV NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    ROBOCASA_REQUIRE_IMAGE_SOURCE_SHA=1 \
    ROBOCASA_SOURCE_COMMIT=8f3c96ec8d1bfcd8126cad2bca887da98d30e997 \
    ROBOSUITE_SOURCE_COMMIT=85abee228d1c43ab1939bce33028099945d453b4 \
    LEROBOT_SOURCE_COMMIT=${LEROBOT_SOURCE_COMMIT} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    ROBOCASA_PORT=8791 \
    PYTHONPATH=/app \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    PATH=/opt/robocasa/venv/bin:$PATH

USER root

# System deps for EGL rendering and the Python build toolchain.
COPY --chmod=0644 docker/workbench/robocasa/deadsnakes-ppa.gpg.b64 \
    /tmp/deadsnakes-ppa.gpg.b64
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python3-venv build-essential git \
        ca-certificates curl \
        ffmpeg fuse netcat-openbsd pciutils procps wget \
        libegl1 libgl1 libglvnd0 libegl1-mesa-dev \
    && install -d -m 0755 /etc/apt/keyrings \
    && base64 --decode /tmp/deadsnakes-ppa.gpg.b64 \
        > /etc/apt/keyrings/deadsnakes-ppa.gpg \
    && echo \
        "5caefc9c4bb1dadde2e03579a1c9ac1ae0d3ae9c15350de361a89782f814c438  /etc/apt/keyrings/deadsnakes-ppa.gpg" \
        | sha256sum --check --strict - \
    && printf '%s\n' \
        "deb [signed-by=/etc/apt/keyrings/deadsnakes-ppa.gpg] https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main" \
        > /etc/apt/sources.list.d/deadsnakes-ubuntu-ppa-jammy.list \
    && rm -f /tmp/deadsnakes-ppa.gpg.b64 \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
    && python3.12 -m venv /opt/robocasa/venv \
    && rm -rf /var/lib/apt/lists/* /run/systemd/resolve \
    && test ! -e /run/systemd/resolve \
    && test ! -L /run/systemd/resolve

WORKDIR /opt/robocasa

# Install the reviewed CPython 3.12 linux/amd64 closure from hash locks generated
# only against anonymous PyPI and PyTorch indexes. RoboCasa requires Gymnasium
# 0.29.1. The exact upstream LeRobot revision is rebuilt as a locally versioned
# NPA ACT derivative whose metadata declares this closure. The patch changes no
# Python implementation bytes; the image validates the resolver and real ACT path.
COPY --chmod=0644 \
    docker/workbench/robocasa/build-requirements.lock \
    docker/workbench/robocasa/requirements.lock \
    /opt/robocasa/locks/
COPY --chmod=0644 \
    docker/workbench/robocasa/lerobot-npa-act.patch.b64 \
    docker/workbench/robocasa/lerobot-npa-act.NOTICE \
    docker/workbench/robocasa/verify_lerobot_act_derivative.py \
    /opt/robocasa/derivative/

RUN PIP_CONFIG_FILE=/dev/null python -m pip install \
        --disable-pip-version-check --no-cache-dir \
        --index-url https://pypi.org/simple \
        --require-hashes --only-binary=:all: \
        -r /opt/robocasa/locks/build-requirements.lock \
    && PIP_CONFIG_FILE=/dev/null python -m pip install \
        --disable-pip-version-check --no-cache-dir \
        --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
        --require-hashes --no-build-isolation \
        -r /opt/robocasa/locks/requirements.lock \
    && curl --fail --silent --show-error --location \
        "https://github.com/huggingface/lerobot/archive/${LEROBOT_SOURCE_COMMIT}.tar.gz" \
        --output /tmp/lerobot-source.tar.gz \
    && echo \
        "869026b70a9488f11ae25c92573f68a03e04e45b4ffda7d63586582ac8d506ee  /tmp/lerobot-source.tar.gz" \
        | sha256sum --check --strict - \
    && mkdir -p /opt/robocasa/lerobot-source \
    && tar --extract --gzip --file /tmp/lerobot-source.tar.gz \
        --strip-components=1 --directory /opt/robocasa/lerobot-source \
    && rm -f /tmp/lerobot-source.tar.gz \
    && base64 --decode /opt/robocasa/derivative/lerobot-npa-act.patch.b64 \
        > /tmp/lerobot-npa-act.patch \
    && echo \
        "7d90538cc6c66351256f394d555797a6aacf34cb17dad2d0c9fc781c7e22404c  /tmp/lerobot-npa-act.patch" \
        | sha256sum --check --strict - \
    && git -C /opt/robocasa/lerobot-source apply --check \
        /tmp/lerobot-npa-act.patch \
    && git -C /opt/robocasa/lerobot-source apply \
        /tmp/lerobot-npa-act.patch \
    && rm -f /tmp/lerobot-npa-act.patch \
    && mkdir -p /opt/robocasa/lerobot-wheel \
    && PIP_CONFIG_FILE=/dev/null python -m pip wheel \
        --disable-pip-version-check --no-cache-dir --no-build-isolation --no-deps \
        --wheel-dir /opt/robocasa/lerobot-wheel \
        /opt/robocasa/lerobot-source \
    && test "$(find /opt/robocasa/lerobot-wheel -maxdepth 1 -type f -name 'lerobot-0.6.1+npa1-*.whl' | wc -l)" = 1 \
    && PIP_CONFIG_FILE=/dev/null python -m pip install \
        --disable-pip-version-check --no-cache-dir \
        --index-url https://pypi.org/simple \
        /opt/robocasa/lerobot-wheel/lerobot-0.6.1+npa1-*.whl \
    && python -m pip check \
    && python /opt/robocasa/derivative/verify_lerobot_act_derivative.py \
    && install -D -m 0644 /opt/robocasa/derivative/lerobot-npa-act.NOTICE \
        /usr/share/doc/lerobot-npa-act/NOTICE \
    && rm -rf /opt/robocasa/lerobot-source /opt/robocasa/lerobot-wheel \
    && PIP_CONFIG_FILE=/dev/null python -m pip install \
        --disable-pip-version-check --no-cache-dir \
        --index-url https://pypi.org/simple \
        --no-build-isolation --no-deps \
        "robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git@${ROBOSUITE_SOURCE_COMMIT}" \
    && python -c "import json; from importlib.metadata import distribution; direct=json.loads(distribution('robosuite').read_text('direct_url.json')); vcs=direct.get('vcs_info', {}); assert vcs.get('vcs') == 'git' and vcs.get('commit_id') == '${ROBOSUITE_SOURCE_COMMIT}', direct" \
    && find /opt/robocasa/venv -type f \
        -path '*/imageio_ffmpeg/binaries/ffmpeg*' -delete \
    && rm -f \
        /opt/robocasa/venv/lib/python3.12/site-packages/robosuite/scripts/render_dataset_with_omniverse.py \
    && find /opt/robocasa/venv -type f \
        -path '*/robosuite/scripts/__pycache__/render_dataset_with_omniverse*.pyc' \
        -delete \
    && test -z "$(find /opt/robocasa/venv -type f \
        -path '*/imageio_ffmpeg/binaries/ffmpeg*' -print -quit)" \
    && test ! -e \
        /opt/robocasa/venv/lib/python3.12/site-packages/robosuite/scripts/render_dataset_with_omniverse.py \
    && test -z "$(find /opt/robocasa/venv -type f \
        -path '*/robosuite/scripts/__pycache__/render_dataset_with_omniverse*.pyc' \
        -print -quit)" \
    && python -c "import imageio_ffmpeg; assert imageio_ffmpeg.get_ffmpeg_exe() == '/usr/bin/ffmpeg'" \
    && python -c "import itertools, tempfile; from pathlib import Path; import imageio.v2 as imageio; import numpy as np; tmp = tempfile.TemporaryDirectory(); video = Path(tmp.name) / 'probe.mp4'; imageio.mimsave(video, [np.zeros((64, 64, 3), dtype=np.uint8), np.full((64, 64, 3), 255, dtype=np.uint8)], fps=2); reader = imageio.get_reader(video); frames = list(itertools.islice(reader, 3)); reader.close(); assert len(frames) == 2 and frames[0].shape == (64, 64, 3); tmp.cleanup()" \
    && rm -rf /root/.cache/pip /tmp/pip-*

# Install RoboCasa source (no deps; the closure above is pinned).
RUN git init /opt/robocasa/source \
    && git -C /opt/robocasa/source remote add origin \
      https://github.com/robocasa/robocasa.git \
    && git -C /opt/robocasa/source fetch --depth 1 origin \
      "${ROBOCASA_SOURCE_COMMIT}" \
    && git -C /opt/robocasa/source checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/robocasa/source rev-parse HEAD)" = \
      "${ROBOCASA_SOURCE_COMMIT}" \
    && PIP_CONFIG_FILE=/dev/null python -m pip install \
      --disable-pip-version-check --no-cache-dir \
      --no-build-isolation --no-deps -e /opt/robocasa/source \
    && rm -rf /opt/robocasa/source/.git /root/.cache/pip /tmp/pip-*

# The pinned CUDA devel base retains stale userspace kernel headers. They are
# needed only by the build toolchain above; remove them and their reverse build
# dependencies before any final-runtime acceptance check.
RUN apt-get purge -y --auto-remove linux-libc-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ! dpkg-query --show linux-libc-dev >/dev/null 2>&1

# Fail the image build if the exact held-out evaluator import path is incomplete.
RUN printf '%s\n' "${NPA_SOURCE_SHA}" | grep -Eq '^[0-9a-f]{40}$' \
    && python -c "from importlib.metadata import version; assert version('lerobot') == '0.6.1+npa1'" \
    && python -m pip check \
    && python -c "from importlib.metadata import distributions; forbidden_names = {'nvidia-cublas', 'nvidia-cuda-cupti', 'nvidia-cuda-nvrtc', 'nvidia-cuda-runtime', 'nvidia-cudnn-cu13', 'nvidia-cufft', 'nvidia-cufile', 'nvidia-curand', 'nvidia-cusolver', 'nvidia-cusparse', 'nvidia-cusparselt-cu13', 'nvidia-nccl-cu13', 'nvidia-nvjitlink', 'nvidia-nvshmem-cu13', 'nvidia-nvtx'}; opencv_names = {'opencv-python', 'opencv-python-headless', 'opencv-contrib-python', 'opencv-contrib-python-headless'}; installed = {str(dist.metadata['Name']).lower(): str(dist.version) for dist in distributions()}; forbidden = sorted(name for name in installed if name in forbidden_names or name.endswith('-cu13')); opencv = sorted(set(installed) & opencv_names); cuda_helpers = {name: installed[name] for name in installed if name.startswith('cuda-')}; assert not forbidden, forbidden; assert opencv == ['opencv-python'], opencv; assert cuda_helpers == {'cuda-bindings': '12.9.8', 'cuda-pathfinder': '1.8.2', 'cuda-toolkit': '12.9.1'}, cuda_helpers" \
    && python -c "import torch, torchvision; assert torch.__version__ == '2.13.0+cu129'; assert torch.version.cuda == '12.9'; assert torchvision.__version__ == '0.28.0+cu129'" \
    && python -c "import gymnasium; assert gymnasium.__version__ == '0.29.1'; from lerobot.policies.act.modeling_act import ACTPolicy; from lerobot.configs.policies import PreTrainedConfig; from lerobot.policies.factory import make_pre_post_processors"

# Copy the shared workbench implementation into the image.
RUN mkdir -p /app/npa/workbench /app/npa/clients /app/npa/cli
RUN touch /app/npa/__init__.py /app/npa/workbench/__init__.py \
    /app/npa/clients/__init__.py /app/npa/cli/__init__.py
COPY src/npa/clients/storage.py /app/npa/clients/storage.py
COPY src/npa/cli/path_contract.py /app/npa/cli/path_contract.py
COPY src/npa/workbench/robocasa /app/npa/workbench/robocasa
COPY src/npa/smoke/test_robocasa_functional.py /app/smoke_functional.py

RUN chmod -R u=rwX,go=rX /app/npa /app/smoke_functional.py

RUN python -c "from npa.workbench.robocasa.service import app; assert app is not None"

RUN id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu \
    && chown -R ubuntu:ubuntu /opt/robocasa/source/robocasa/models/assets

EXPOSE 8791

USER ubuntu

# Exercise the final runtime user and real upstream registration before the
# candidate can leave the build. Assets remain runtime-fetched; only the owned
# destination's writability is checked here. The golden smoke also executes
# registration through the production spawn/process-group boundary.
RUN test "$(id -u)" != "0" \
    && python /app/smoke_functional.py

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python3 -c "import os,urllib.request; p=os.environ.get('ROBOCASA_PORT','8791'); urllib.request.urlopen(f'http://127.0.0.1:{p}/health', timeout=3)" || exit 1

CMD ["uvicorn", "npa.workbench.robocasa.service:app", "--host", "0.0.0.0", "--port", "8791"]
