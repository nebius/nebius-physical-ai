# CPU point-cloud registration and surface reconstruction. Open3D's geometry and
# `pipelines.registration` APIs are CPU implementations, so this image
# deliberately does not derive from the CUDA base: a registration stage that
# asked for a GPU would occupy an accelerator it cannot use.
#
# Digest pinned 2026-06-02. Tag: python:3.11-slim-trixie. Shared with
# npa-rerun-viewer so both reviewed CPU images track one base.
FROM python:3.11-slim-trixie@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0

ARG OPEN3D_VERSION=0.20.0
ARG NPA_SOURCE_SHA
ARG DEBIAN_SNAPSHOT=20260801T000000Z

LABEL npa.tool="open3d" \
      npa.version="${OPEN3D_VERSION}" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

# OPEN3D_DATA_ROOT: Open3D's data module caches the upstream fragment archives
# there. Keeping it under the runtime user's home leaves /opt read-only.
ENV DEBIAN_FRONTEND=noninteractive \
    NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    OPEN3D_VERSION=${OPEN3D_VERSION} \
    OPEN3D_DATA_ROOT=/home/ubuntu/.cache/open3d \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# SkyPilot's Kubernetes runtime bootstrap runs inside this container: it needs a
# system python3, rsync, and passwordless sudo, and it is capability-probed over
# SSH. Source OS packages from an immutable Debian snapshot so a rebuild of a
# published tag resolves the same bytes.
#
# libOpen3D links the GL/EGL loaders (libgl1, libegl1) at import time even when
# nothing renders; libgomp1 is its OpenMP runtime, and libusb/libidn2 come from
# its bundled RealSense and cURL closure.
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
      ca-certificates curl openssh-client openssh-server procps rsync sudo \
      libgl1 libegl1 libgomp1 libglib2.0-0 libusb-1.0-0 libidn2-0 \
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
    && chown ubuntu:ubuntu /workspace

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu workflows /opt/npa/workflows
COPY --chown=ubuntu:ubuntu docker/workbench/open3d/requirements.txt \
    /opt/npa/open3d-requirements.txt
COPY --chown=ubuntu:ubuntu --chmod=0755 docker/workbench/open3d/entrypoint.sh \
    /opt/npa/docker/workbench/open3d/entrypoint.sh

USER ubuntu
WORKDIR /workspace
ENV PATH=/home/ubuntu/.local/bin:$PATH

# The checks below prove the registration and reconstruction entrypoints this
# image exists for actually resolve, rather than only that the wheel imports.
RUN test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40 \
    && python -m pip install --no-cache-dir --user --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir --user /opt/npa \
    && python -m pip install --no-cache-dir --user -r /opt/npa/open3d-requirements.txt \
    && python -m pip check \
    && python -c "import open3d, os; assert open3d.__version__ == os.environ['OPEN3D_VERSION'], open3d.__version__" \
    && python -c "import open3d as o3d; r = o3d.pipelines.registration; assert callable(r.registration_icp); assert callable(r.registration_ransac_based_on_feature_matching); assert callable(r.compute_fpfh_feature); assert callable(r.global_optimization); assert callable(o3d.geometry.TriangleMesh.create_from_point_cloud_poisson)" \
    && python -c "from npa.workbench.open3d import runner; assert callable(runner.main)" \
    && npa workbench open3d --help > /dev/null \
    && install -d -m 0700 /home/ubuntu/.cache/open3d

# Command-passthrough: SkyPilot's Kubernetes bootstrap runs its own shell in this
# container, so argv must reach it unmodified. A stage supplies
# `npa workbench open3d <verb>`, and `runner.py` is the subprocess that verb
# launches.
ENTRYPOINT ["/opt/npa/docker/workbench/open3d/entrypoint.sh"]
