# Digest pinned 2026-06-02. Tag: python:3.11-slim-trixie.
FROM python:3.11-slim-trixie@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0

ARG RERUN_SDK_VERSION=0.31.4
ARG NPA_SOURCE_SHA
ARG DEBIAN_SNAPSHOT=20260801T000000Z

LABEL npa.tool="rerun-viewer" \
      npa.version="${RERUN_SDK_VERSION}" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}"

ENV NPA_BAKED_PYTHON=/usr/local/bin/python \
    NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    NPA_SKIP_EAGER_IMPORTS=1 \
    PYTHONPATH=/opt/npa/src \
    PYTHONUNBUFFERED=1

# Stage 14 is a standard SkyPilot managed task, not only a `rerun serve` image.
# Bake the same non-root bootstrap closure as the controller so provisioning does
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
    && apt-get install -y --no-install-recommends \
      curl netcat-openbsd openssh-client openssh-server procps rsync sudo \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/bash -u 1000 ubuntu \
    && printf 'ubuntu ALL=(ALL) NOPASSWD:ALL\n' \
      > /etc/sudoers.d/99-npa-runtime-user \
    && chmod 0440 /etc/sudoers.d/99-npa-runtime-user \
    && install -d -m 0755 /var/run/sshd /workspace /opt/npa

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu workflows /opt/npa/workflows
COPY docker/workbench/common/sim2real-viewer-requirements.txt \
    /opt/npa/sim2real-viewer-requirements.txt
COPY --chmod=0755 docker/workbench/common/workflow_runtime_entrypoint.sh \
    /usr/local/bin/npa-workflow-entrypoint

RUN test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40 \
    && find /opt/npa/src /opt/npa/workflows -type d -exec chmod a+rx {} + \
    && find /opt/npa/src /opt/npa/workflows -type f -exec chmod a+r {} + \
    && chmod a+r /opt/npa/pyproject.toml \
    && python -m pip install --no-cache-dir --no-deps \
      -r /opt/npa/sim2real-viewer-requirements.txt \
    && python -m pip check \
    && python -c "import site; from pathlib import Path; Path(site.getsitepackages()[0], 'npa-exact-source.pth').write_text('/opt/npa/src\n', encoding='utf-8')" \
    && env -u PYTHONPATH python -c \
      'import boto3, mcap, rerun; from npa.workflows.sim2real.workflow_stage import main; assert callable(main)' \
    && chown -R ubuntu:ubuntu /workspace

WORKDIR /workspace
EXPOSE 9090
USER ubuntu
ENTRYPOINT ["/usr/local/bin/npa-workflow-entrypoint"]
