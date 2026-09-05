# Digest pinned 2026-05-12. Tag: nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:da6791294b0b04d7e65d87b7451d6f2390b4d36225ab0701ee7dfec5769829f5

ENV DEBIAN_FRONTEND=noninteractive \
    HF_LEROBOT_HOME=/opt/lerobot/hf_cache \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    PYTHONUNBUFFERED=1 \
    NPA_POLICY_HOST=0.0.0.0 \
    NPA_POLICY_PORT=8080 \
    NPA_DEFAULT_POLICY_CHECKPOINT=lerobot/diffusion_pusht

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
      libegl1 \
      libgl1 \
      libglib2.0-0 \
      python3.12 \
      python3.12-dev \
      python3.12-venv \
      python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash -u 1000 ubuntu \
    && install -d -m 0755 -o ubuntu -g ubuntu /opt/lerobot /opt/npa/src/npa /opt/npa/docker

USER ubuntu
WORKDIR /opt/lerobot

COPY --chown=ubuntu:ubuntu docker/workbench/lerobot-policy/requirements.txt /tmp/npa-lerobot-policy-requirements.txt

# LeRobot 0.5.1 pulls torchcodec<0.11.0 which ships libraries linked against
# FFmpeg 4-7 sonames.  Ubuntu 22.04 provides only FFmpeg 4, and torchcodec's
# FFmpeg-4 variant crashes with torch 2.12 (ABI mismatch).  Install FFmpeg 8
# shared libraries from the official BtbN build so torchcodec 0.16.0 can find
# its matching shared libraries.
#
# The imageio-ffmpeg wheel bundles a GPL/BSD-3-licensed static FFmpeg binary
# whose licensing is incompatible with public redistribution.  Remove it inside
# this same RUN layer so that the OCI history never records the forbidden bytes;
# a later RUN with a whiteout delete would leave the payload reachable in an
# earlier layer.  LeRobot uses the system ffmpeg provided by apt instead.
RUN python3.12 -m venv /opt/lerobot/venv \
    && /opt/lerobot/venv/bin/python -m pip install --upgrade pip setuptools wheel \
    && /opt/lerobot/venv/bin/pip install --no-cache-dir -r /tmp/npa-lerobot-policy-requirements.txt \
    && /opt/lerobot/venv/bin/pip install --no-cache-dir --upgrade \
      "torch==2.12.1" \
      "torchvision==0.27.1" \
      "diffusers>=0.38.0" \
      "torchcodec>=0.16.0" \
    && /opt/lerobot/venv/bin/pip uninstall -y wandb \
    && find /opt/lerobot/venv -type f -path '*/imageio_ffmpeg/binaries/ffmpeg*' -delete \
    && export IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg \
    && test "$(/opt/lerobot/venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" = /usr/bin/ffmpeg \
    && test -z "$(find /opt/lerobot/venv -type f -path '*/imageio_ffmpeg/binaries/ffmpeg*' -print -quit)"

USER root

# FFmpeg 8 shared libraries from BtbN official build.  torchcodec 0.16.0 ships
# libtorchcodec_core8.so linked against libavutil.so.61 / libavcodec.so.63 etc.
# The BtbN master shared build provides these exact sonames.  Add them to the
# system linker path and purge the download tools afterward.
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/ffmpeg8 \
    && cd /opt/ffmpeg8 \
    && wget -q 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl-shared.tar.xz' \
    && tar -xf ffmpeg-master-latest-linux64-gpl-shared.tar.xz --strip-components=1 \
    && rm ffmpeg-master-latest-linux64-gpl-shared.tar.xz \
    && echo /opt/ffmpeg8/lib > /etc/ld.so.conf.d/ffmpeg8.conf \
    && ldconfig

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ffmpeg \
    && apt-get install -y --no-install-recommends --only-upgrade gpgv openssl libssl3 \
    && apt-get purge -y --auto-remove \
      dirmngr \
      gnupg \
      gnupg-utils \
      gnupg2 \
      gpg \
      gpg-agent \
      software-properties-common \
      wget \
      xz-utils \
    && dpkg --purge --force-depends linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*

USER ubuntu

ENV PATH=/opt/lerobot/venv/bin:$PATH \
    PYTHONPATH=/opt/npa/src \
    NPA_SKIP_EAGER_IMPORTS=1

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu docker/workbench/lerobot-policy/entrypoint.sh /opt/lerobot/entrypoint.sh

RUN chmod +x /opt/lerobot/entrypoint.sh \
    && python - <<'PY'
from npa.workbench.lerobot.policy_container import assert_lerobot_importable, parse_feedback_batch
result = assert_lerobot_importable()
assert result.dataset_class.endswith(".LeRobotDataset")
assert parse_feedback_batch({"success": True, "score": 0.9, "rationale": "ok"})
PY

EXPOSE 8080

ENTRYPOINT ["/opt/lerobot/entrypoint.sh"]
CMD ["serve"]
