# Digest pinned 2026-06-02. Tag: python:3.11-slim-trixie
# To update: re-resolve via registry API and update the digest below.
FROM python:3.11-slim-trixie@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0

ARG RERUN_SDK_VERSION=0.31.4

LABEL npa.tool="rerun-viewer" \
      npa.version="${RERUN_SDK_VERSION}" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/rerun/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    RERUN_SDK_VERSION=${RERUN_SDK_VERSION}

# This image is both the interactive Rerun server and the CPU worker for
# workbench.nurec.visualize. SkyPilot's Kubernetes provisioner bootstraps SSH
# through sudo as the image user, so the immutable worker has to provide the
# bootstrap closure before a task starts.
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      ffmpeg \
      netcat-openbsd \
      openssh-server \
      procps \
      rsync \
      sudo \
    && install -d -m 0755 /run/sshd \
    && rm -f /etc/ssh/ssh_host_* \
    && printf '%s\n' \
      'PasswordAuthentication no' \
      'PermitRootLogin no' \
      > /etc/ssh/sshd_config.d/99-npa-container.conf \
    && rm -rf /var/lib/apt/lists/*

# SkyPilot invokes apt and service through sudo. The grant is confined to the
# dedicated non-root workflow user and the image starts no SSH service itself.
RUN useradd -m -s /bin/bash -u 1000 ubuntu \
    && printf 'ubuntu ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/99-npa-runtime-user \
    && chmod 0440 /etc/sudoers.d/99-npa-runtime-user \
    && install -d -m 0755 -o ubuntu -g ubuntu \
      /opt/npa/docker/workbench/rerun-viewer \
      /opt/rerun \
      /opt/rerun/venv

USER ubuntu
WORKDIR /opt/rerun

RUN python -m venv /opt/rerun/venv \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir "rerun-sdk==${RERUN_SDK_VERSION}" \
    && python -c 'import os, rerun; assert rerun.__version__ == os.environ["RERUN_SDK_VERSION"]'

COPY --chown=ubuntu:ubuntu --chmod=0755 docker/workbench/rerun-viewer/entrypoint.sh /opt/npa/docker/workbench/rerun-viewer/entrypoint.sh

EXPOSE 9090 9876
ENTRYPOINT ["/opt/npa/docker/workbench/rerun-viewer/entrypoint.sh"]
