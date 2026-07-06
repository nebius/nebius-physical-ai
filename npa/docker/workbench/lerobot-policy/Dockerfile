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

RUN python3.12 -m venv /opt/lerobot/venv \
    && /opt/lerobot/venv/bin/python -m pip install --upgrade pip setuptools wheel \
    && /opt/lerobot/venv/bin/pip install --no-cache-dir -r /tmp/npa-lerobot-policy-requirements.txt \
    && /opt/lerobot/venv/bin/pip install --no-cache-dir --upgrade \
      "torch==2.12.1" \
      "torchvision==0.27.1" \
      "diffusers>=0.38.0" \
    && /opt/lerobot/venv/bin/pip uninstall -y wandb

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
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
