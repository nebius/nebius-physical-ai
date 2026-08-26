# Digest pinned 2026-06-02. Tag: python:3.11-slim-trixie.
FROM python:3.11-slim-trixie@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0

ARG RERUN_SDK_VERSION=0.31.4
ARG NPA_SOURCE_SHA
ARG DEBIAN_SNAPSHOT=20260801T000000Z

LABEL npa.tool="rerun-viewer" \
      npa.version="${RERUN_SDK_VERSION}" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/rerun/venv/bin:$PATH \
    NPA_BAKED_PYTHON=/opt/rerun/venv/bin/python \
    NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    NPA_SKIP_EAGER_IMPORTS=1 \
    PYTHONPATH=/opt/npa/src \
    PYTHONUNBUFFERED=1 \
    RERUN_SDK_VERSION=${RERUN_SDK_VERSION}

# This image is both the exact-source Sim2Real Stage 14 worker and the interactive
# Rerun server. Bake the non-root SkyPilot bootstrap closure so provisioning does
# not depend on mutable pod-time package downloads.
RUN rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*.list \
      /etc/apt/sources.list.d/*.sources \
    && printf '%s\n' \
      'Types: deb' \
      "URIs: https://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}/" \
      'Suites: trixie trixie-updates' \
      'Components: main' \
      'Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg' \
      > /etc/apt/sources.list.d/debian.sources \
    && printf '%s\n' \
      'Types: deb' \
      "URIs: https://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}/" \
      'Suites: trixie-security' \
      'Components: main' \
      'Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg' \
      > /etc/apt/sources.list.d/debian-security.sources \
    && printf 'Acquire::Check-Valid-Until "false";\n' \
      > /etc/apt/apt.conf.d/99snapshot \
    && apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
      ca-certificates curl ffmpeg netcat-openbsd openssh-client openssh-server \
      procps rsync sudo \
    && install -d -m 0755 /run/sshd /workspace /opt/npa \
    && rm -f /etc/ssh/ssh_host_* \
    && printf '%s\n' \
      'PasswordAuthentication no' \
      'PermitRootLogin no' \
      > /etc/ssh/sshd_config.d/99-npa-container.conf \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/bash -u 1000 ubuntu \
    && printf 'ubuntu ALL=(ALL) NOPASSWD:ALL\n' \
      > /etc/sudoers.d/99-npa-runtime-user \
    && chmod 0440 /etc/sudoers.d/99-npa-runtime-user \
    && install -d -m 0755 -o ubuntu -g ubuntu \
      /opt/npa/docker/workbench/rerun-viewer \
      /opt/rerun \
      /opt/rerun/venv \
    && chown ubuntu:ubuntu /workspace

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu workflows /opt/npa/workflows
COPY --chown=ubuntu:ubuntu docker/workbench/common/sim2real-viewer-requirements.txt \
    /opt/npa/sim2real-viewer-requirements.txt
COPY --chown=ubuntu:ubuntu --chmod=0755 docker/workbench/rerun-viewer/entrypoint.sh \
    /opt/npa/docker/workbench/rerun-viewer/entrypoint.sh

USER ubuntu
WORKDIR /opt/rerun

RUN test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40 \
    && python -m venv /opt/rerun/venv \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir "rerun-sdk==${RERUN_SDK_VERSION}" \
    && python -m pip install --no-cache-dir --no-deps \
      -r /opt/npa/sim2real-viewer-requirements.txt \
    && python -m pip check \
    && python -c "import site; from pathlib import Path; Path(site.getsitepackages()[0], 'npa-exact-source.pth').write_text('/opt/npa/src\n', encoding='utf-8')" \
    && env -u PYTHONPATH python -c \
      'import boto3, mcap, rerun; from npa.workflows.sim2real.workflow_stage import main; assert callable(main)' \
    && python -c 'import os, rerun; assert rerun.__version__ == os.environ["RERUN_SDK_VERSION"]'

WORKDIR /workspace
EXPOSE 9090 9876
ENTRYPOINT ["/opt/npa/docker/workbench/rerun-viewer/entrypoint.sh"]
