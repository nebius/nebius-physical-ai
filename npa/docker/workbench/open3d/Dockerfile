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
# Moved forward from 20260801T000000Z, which resolved trixie before the 13.7
# point release (2026-09-12) and so still carried libglib2.0-0t64
# 2.84.4-3~deb13u3 and perl-base 5.40.1-6. Those are the versions the image
# security policy failed on: CVE-2026-58016 for glib, CVE-2026-13221 /
# CVE-2026-42496 / CVE-2026-8376 for perl.
ARG DEBIAN_SNAPSHOT=20260913T000000Z

# The two packages whose vulnerable versions blocked this image, as the lowest
# version that carries the fix. `apt-get upgrade` below is what installs them;
# these are asserted afterwards so a snapshot that silently stops carrying the
# fix fails the build instead of shipping. Compared with dpkg rather than as
# strings, because `~` sorts *before* the empty string and `u10` is above `u5`:
# glib's actual fixed version in trixie is 2.84.4-3~deb13u5, above the
# 2.84.4-3~deb13u4 the scanner named, and no string comparison gets that right.
ARG MIN_LIBGLIB_VERSION=2.84.4-3~deb13u4
ARG MIN_PERL_BASE_VERSION=5.40.1-6+deb13u1

# The mcap release the retained notice was taken from, and that notice's bytes.
# mcap ships no licence file in its wheel or its sdist, so this image carries the
# grant; both are checked at build time. See THIRD_PARTY_NOTICES.md.
ARG MCAP_NOTICE_VERSION=1.4.0
ARG MCAP_NOTICE_SHA256=da11235665c17d4c1634072dae92b8ba1b38d6fdde2ccf19a6bbede33253f58d

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
    && for pin in "libglib2.0-0t64 ${MIN_LIBGLIB_VERSION}" \
                  "perl-base ${MIN_PERL_BASE_VERSION}"; do \
         set -- ${pin}; \
         installed="$(dpkg-query --show --showformat='${Version}' "$1")"; \
         dpkg --compare-versions "${installed}" ge "$2" || { \
           echo "$1 ${installed} is below the fixed $2" >&2; exit 1; }; \
       done \
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
# mcap is MIT but ships no licence file in either its wheel or its sdist, so the
# grant only reaches a recipient if the image carries it. See THIRD_PARTY_NOTICES.md.
COPY --chmod=0444 docker/workbench/open3d/THIRD_PARTY_NOTICES.md \
    /usr/share/doc/npa-open3d/THIRD_PARTY_NOTICES.md
COPY --chmod=0444 docker/workbench/open3d/notices/ \
    /usr/share/doc/npa-open3d/notices/

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
    && echo "${MCAP_NOTICE_SHA256}  /usr/share/doc/npa-open3d/notices/mcap-LICENSE.txt" \
      | sha256sum --check --status \
    && python -c "import importlib.metadata as m; v = m.version('mcap'); assert v == '${MCAP_NOTICE_VERSION}', f'mcap {v} no longer matches the retained notice; refresh docker/workbench/open3d/notices/mcap-LICENSE.txt'" \
    && install -d -m 0700 /home/ubuntu/.cache/open3d

# Command-passthrough: SkyPilot's Kubernetes bootstrap runs its own shell in this
# container, so argv must reach it unmodified. A stage supplies
# `npa workbench open3d <verb>`, and `runner.py` is the subprocess that verb
# launches.
ENTRYPOINT ["/opt/npa/docker/workbench/open3d/entrypoint.sh"]
