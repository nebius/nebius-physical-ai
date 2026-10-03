# Digest pinned 2026-06-02. Tag: python:3.11-slim-trixie.
FROM python:3.11-slim-trixie@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0

ARG SIM2REAL_CONTROL_VERSION=0.1.2
ARG NPA_SOURCE_SHA
ARG DEBIAN_SNAPSHOT=20260801T000000Z

LABEL npa.tool="sim2real-control" \
      npa.version="${SIM2REAL_CONTROL_VERSION}" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

ENV NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    NPA_SKIP_EAGER_IMPORTS=1 \
    PYTHONPATH=/opt/npa/src \
    PYTHONUNBUFFERED=1

RUN useradd -m -s /bin/bash -u 1000 ubuntu \
    && install -d -m 0755 -o ubuntu -g ubuntu /workspace /opt/npa

# SkyPilot's Kubernetes bootstrap replaces the image command, installs/starts sshd,
# and uses passwordless sudo before a workflow stage can run.  Bake that closure here:
# relying on a pod-time apt download makes a transient package mirror part of every
# stage's availability, while omitting sudo makes the container exit before SkyPilot can
# even inspect it (live compositional attempt 6).  The Debian snapshot keeps the system
# dependency closure reproducible; the final workflow still pins this image by digest.
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
    && rm -f /etc/ssh/ssh_host_* \
    && test -z "$(find /etc/ssh -maxdepth 1 -type f -name 'ssh_host_*' -print -quit)" \
    && printf '%s\n' \
      'PasswordAuthentication no' \
      'PermitRootLogin no' \
      > /etc/ssh/sshd_config.d/99-npa-runtime.conf \
    && rm -rf /var/lib/apt/lists/* \
    && printf 'ubuntu ALL=(ALL) NOPASSWD:ALL\n' \
      > /etc/sudoers.d/99-npa-runtime-user \
    && chmod 0440 /etc/sudoers.d/99-npa-runtime-user \
    && install -d -m 0755 /var/run/sshd \
    && python3 --version \
    && rsync --version >/dev/null

WORKDIR /workspace

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu workflows /opt/npa/workflows
COPY docker/workbench/common/sim2real-controller-requirements.txt \
    /opt/npa/sim2real-controller-requirements.txt
COPY docker/workbench/common/sim2real-control-requirements.txt \
    /opt/npa/sim2real-control-requirements.txt
COPY --chmod=0755 docker/workbench/common/workflow_runtime_entrypoint.sh \
    /usr/local/bin/npa-workflow-entrypoint

RUN test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40 \
    && find /opt/npa/src /opt/npa/workflows -type d -exec chmod a+rx {} + \
    && find /opt/npa/src /opt/npa/workflows -type f -exec chmod a+r {} + \
    && chmod a+r /opt/npa/pyproject.toml \
    && python -m pip install --no-cache-dir --no-deps \
      -r /opt/npa/sim2real-controller-requirements.txt \
      -r /opt/npa/sim2real-control-requirements.txt \
    && python -m pip check \
    && python -c "import site; from pathlib import Path; Path(site.getsitepackages()[0], 'npa-exact-source.pth').write_text('/opt/npa/src\n', encoding='utf-8')" \
    && python -m py_compile \
      /opt/npa/src/npa/workflows/sim2real/stage9_replay.py \
      /opt/npa/src/npa/workflows/sim2real/stage14_finalize.py \
      /opt/npa/src/npa/workflows/sim2real/workflow_io.py \
      /opt/npa/src/npa/workflows/sim2real/workflow_stage.py \
    && env -u PYTHONPATH python -c "import os; os.environ['NEBIUS_TOKEN_FACTORY_KEY']='build-smoke'; import boto3, httpx, yaml; from npa.clients.token_factory import TokenFactoryClient; from npa.workbench.cosmos.reason import run_token_factory_rollout_vlm; from npa.workflows.sim2real.stage8_cosmos3 import run; from npa.workflows.sim2real.workflow_stage import main; assert callable(run_token_factory_rollout_vlm) and callable(run) and callable(main); TokenFactoryClient()" \
    && chown -R ubuntu:ubuntu /workspace

USER ubuntu

ENTRYPOINT ["/usr/local/bin/npa-workflow-entrypoint"]
