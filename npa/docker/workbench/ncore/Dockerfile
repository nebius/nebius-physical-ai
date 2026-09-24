# syntax=docker/dockerfile:1.7
# CPU NCore converter/V4 reader. Build context: committed npa/ from SOURCE_SHA.
FROM python:3.12.14-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254 AS runtime-base
ARG DEBIAN_SNAPSHOT=20260906T183022Z
# The base's CPython source-build key is unused by this recipe. Debian source
# authentication uses its independently hash-pinned archive keyring.
ENV GPG_KEY="" DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN sed -ri \
      -e "s|https?://deb.debian.org/debian-security|https://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}|g" \
      -e "s|https?://deb.debian.org/debian|https://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}|g" \
      /etc/apt/sources.list.d/debian.sources \
    && printf 'Acquire::Check-Valid-Until "false";\n' >/etc/apt/apt.conf.d/99snapshot \
    && apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
      ca-certificates curl netcat-openbsd openssh-server procps rsync sudo \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/ssh/ssh_host_* \
    && useradd -m -s /bin/bash -u 1000 ubuntu \
    && printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' >/etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu \
    && visudo -cf /etc/sudoers.d/ubuntu \
    && install -d -m 0755 /run/sshd \
    && printf '%s\n' 'PasswordAuthentication no' 'PermitRootLogin no' \
      >/etc/ssh/sshd_config.d/npa.conf \
    && install -d -o ubuntu -g ubuntu /workspace /home/ubuntu/.cache

# Build-only patch utility never enters the final layers.
FROM runtime-base AS source-stage
RUN apt-get update && apt-get install -y --no-install-recommends patch \
    && rm -rf /var/lib/apt/lists/*
COPY docker/workbench/ncore/source-lock.json docker/workbench/ncore/stage_upstream.py /build/
COPY src/npa/_public_https.py /build/_public_https.py
RUN python -B /build/stage_upstream.py --lock /build/source-lock.json --output /opt/ncore/src

# Deliver corresponding source only for selected covered base binaries.
# Permissive components carry their exact notices in the assembled filesystem.
FROM scratch AS base-source-cache
FROM runtime-base AS base-source-delivery
COPY docker/workbench/ncore/base-source-lock.json docker/workbench/ncore/base_sources.py \
     docker/workbench/ncore/BASE-SOURCES.md /opt/ncore/base-sources/recipes/
COPY src/npa/_public_https.py /opt/ncore/base-sources/recipes/_public_https.py
RUN --mount=type=bind,from=base-source-cache,target=/source-cache \
    python /opt/ncore/base-sources/recipes/base_sources.py assemble \
      --lock /opt/ncore/base-sources/recipes/base-source-lock.json \
      --annex /opt/ncore/base-sources --native /opt/ncore/base-sources --cache /source-cache \
      --metadata /build/base-metadata \
    && python /opt/ncore/base-sources/recipes/base_sources.py verify \
      --lock /opt/ncore/base-sources/recipes/base-source-lock.json \
      --annex /opt/ncore/base-sources --native /opt/ncore/base-sources \
      --metadata /build/base-metadata \
    && chmod -R a+rX /opt/ncore/base-sources

FROM runtime-base AS assembled
ARG NPA_SOURCE_SHA
ARG SOURCE_SHA=${NPA_SOURCE_SHA}
ENV PATH=/opt/venv/bin:/opt/ncore/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    HOME=/home/ubuntu XDG_CACHE_HOME=/home/ubuntu/.cache \
    NPA_NCORE_CONVERTER=/opt/ncore/bin/colmap-convert \
    NPA_NCORE_SOURCE_REVISION=59c698d206da92b406a4f72619fce3b3a2c64bfd \
    NPA_SOURCE_SHA=${SOURCE_SHA}
COPY --from=source-stage /opt/ncore/src /opt/ncore/src
COPY --from=base-source-delivery /opt/ncore/base-sources /opt/ncore/base-sources
# Explicit NPA import closure; no package install, CLI registry or SDK fanout.
COPY src/npa/__init__.py src/npa/_sdk.py src/npa/errors.py src/npa/lifecycle_intent.py /opt/npa/src/npa/
COPY src/npa/_public_https.py /opt/npa/src/npa/_public_https.py
COPY src/npa/clients/__init__.py src/npa/clients/storage.py /opt/npa/src/npa/clients/
COPY src/npa/cli/__init__.py src/npa/cli/path_contract.py /opt/npa/src/npa/cli/
COPY src/npa/cli/nurec/__init__.py /opt/npa/src/npa/cli/nurec/__init__.py
COPY src/npa/workbench/__init__.py /opt/npa/src/npa/workbench/__init__.py
COPY src/npa/workbench/ncore_staging.py /opt/npa/src/npa/workbench/ncore_staging.py
COPY src/npa/workbench/nurec/__init__.py src/npa/workbench/nurec/nurec.py \
     src/npa/workbench/nurec/colmap.py src/npa/workbench/nurec/ncore_rig.py /opt/npa/src/npa/workbench/nurec/
COPY src/npa/workflows/__init__.py src/npa/workflows/ncore.py src/npa/workflows/ncore_runtime.py /opt/npa/src/npa/workflows/
# The existing renderer invokes npa.cli.entry.main; both aliases use the same
# narrow adapter, while the public repository CLI/SDK remain unchanged.
COPY src/npa/workflows/ncore.py /opt/npa/src/npa/cli/entry.py
COPY src/npa/workflows/ncore.py /opt/npa/src/npa/__main__.py
RUN python -m venv --without-pip /opt/venv \
    && printf '%s\n' /opt/npa/src >/opt/venv/lib/python3.12/site-packages/npa-source.pth
COPY docker/workbench/ncore/runtime-lock.json docker/workbench/ncore/*requirements.lock \
     docker/workbench/ncore/source-lock.json docker/workbench/ncore/REDISTRIBUTION.md \
     docker/workbench/ncore/BASE-SOURCES.md /usr/share/doc/npa-ncore/
COPY docker/workbench/ncore/Dockerfile docker/workbench/ncore/stage_upstream.py /usr/share/doc/npa-ncore/recipes/
COPY src/npa/_public_https.py /usr/share/doc/npa-ncore/recipes/_public_https.py
COPY --from=source-stage /opt/ncore/src/ncore/LICENSE /usr/share/doc/npa-ncore/LICENSE-APACHE-2.0
COPY --chmod=0644 docker/workbench/ncore/notices/NPA-LICENSE \
     docker/workbench/ncore/notices/NOTICE-NVIDIA-NCORE-COLMAP \
     docker/workbench/ncore/notices/NOTICE-NVIDIA-SKILLS /usr/share/doc/npa-ncore/notices/
COPY --chmod=0644 docker/workbench/ncore/notices/cpython/ /usr/share/doc/npa-ncore/cpython/
COPY --chmod=0755 docker/workbench/ncore/colmap-convert docker/workbench/ncore/verify-packaging.py \
     docker/workbench/ncore/entrypoint.sh /opt/ncore/bin/
COPY --chmod=0755 docker/workbench/ncore/npa-entrypoint /opt/venv/bin/npa
COPY --chmod=0644 docker/workbench/ncore/native-bootstrap-lock.json \
     docker/workbench/ncore/native_bootstrap.py src/npa/_public_https.py /opt/ncore/native/
COPY --chmod=0644 docker/workbench/ncore/native-bootstrap.sh /opt/ncore/bin/
# Root installation uses the fixed base interpreter with -I -S, never the mutable
# application cache. The downloader/hook/lock and all ancestors stay root owned.
RUN chown -R root:root /opt/ncore/native /opt/ncore/bin /opt/venv \
    && chmod 0755 /opt/ncore/native \
    && chmod -R go-w /opt/ncore/native /opt/ncore/bin /opt/venv \
    && /usr/local/bin/python3.12 -I -S -B /opt/ncore/native/native_bootstrap.py verify-lock
# TEST: exercise non-root SkyPilot sudo/SSH bootstrap, as required by ncore's
# passwordless_root_exemption in packaging-contract.yaml; remove test host keys.
# trivy:ignore:AVD-DS-0010
RUN /opt/venv/bin/python -c 'import os,re; assert re.fullmatch("[0-9a-f]{40}", os.environ["NPA_SOURCE_SHA"])' \
    && printf '%s\n' "${SOURCE_SHA}" >/usr/share/doc/npa-ncore/npa-source-sha \
    && su ubuntu -s /bin/bash -c 'sudo -n ssh-keygen -A && sudo -n /usr/sbin/sshd -t && sudo -n service ssh restart && sudo -n service ssh stop && rsync --version >/dev/null && test -w /tmp && test -w "$HOME"' \
    && rm -f /etc/ssh/ssh_host_* \
    && test -z "$(find /etc/ssh -name 'ssh_host_*' -print -quit)"
# Copy the selected base files into an empty root; no builder ancestry survives.
# Merge documentation contents because selected CPython notices already created the directory.
RUN --mount=type=bind,from=base-source-delivery,source=/build/base-metadata,target=/build/base-metadata \
    python /opt/ncore/base-sources/recipes/base_sources.py assemble-root \
      --lock /opt/ncore/base-sources/recipes/base-source-lock.json --root / --output /public-root \
    && mkdir -p /public-root/opt \
    && cp -a /opt/ncore /opt/npa /opt/venv /public-root/opt/ \
    && cp -a /usr/share/doc/npa-ncore/. /public-root/usr/share/doc/npa-ncore/ \
    && printf '%s\n' '/usr/local/lib' >/public-root/etc/ld.so.conf \
    && ldconfig -r /public-root \
    && cp -a /dev/null /dev/urandom /public-root/dev/ \
    && (trap 'if test -s /public-root/run/sshd.pid; then kill "$(cat /public-root/run/sshd.pid)" 2>/dev/null || true; fi' EXIT; \
        chroot --userspec=1000:1000 /public-root /bin/bash -ec 'sudo -n ssh-keygen -A; sudo -n /usr/sbin/sshd -t; sudo -n service ssh restart; test -s /run/sshd.pid; sudo -n kill -0 "$(cat /run/sshd.pid)"; sudo -n service ssh stop; rsync --version >/dev/null; test -w /tmp; test -w "$HOME"; /opt/venv/bin/python /opt/ncore/bin/verify-packaging.py --image-only') \
    && chroot --userspec=1000:1000 /public-root /usr/local/bin/python3.12 -I -S -B /opt/ncore/native/native_bootstrap.py verify-lock \
    && chroot --userspec=1000:1000 /public-root /opt/venv/bin/python -c 'import ssl; assert ssl.create_default_context().cert_store_stats()["x509_ca"] > 0' \
    && rm -f /public-root/etc/ssh/ssh_host_* /public-root/run/sshd.pid /public-root/dev/null /public-root/dev/urandom \
    && python /opt/ncore/base-sources/recipes/base_sources.py verify \
      --lock /opt/ncore/base-sources/recipes/base-source-lock.json \
      --annex /public-root/opt/ncore/base-sources --native /public-root/opt/ncore/base-sources \
      --metadata /build/base-metadata --root /public-root \
    && test ! -e /public-root/usr/lib/x86_64-linux-gnu/libgnutls.so.30.34.3 \
    && test ! -L /public-root/usr/lib/x86_64-linux-gnu/libgnutls.so.30.34.3 \
    && test ! -L /public-root/usr/lib/x86_64-linux-gnu/libgnutls.so.30 \
    && test ! -e /public-root/usr/lib/x86_64-linux-gnu/libgnutls.so.30 \
    && test ! -e /public-root/usr/lib/x86_64-linux-gnu/libssh2.so.1.0.1 \
    && test ! -L /public-root/usr/lib/x86_64-linux-gnu/libssh2.so.1.0.1 \
    && test ! -L /public-root/usr/lib/x86_64-linux-gnu/libssh2.so.1 \
    && test ! -e /public-root/usr/lib/x86_64-linux-gnu/libssh2.so.1 \
    && test ! -e /public-root/home/ubuntu/.cache/npa \
    && test ! -L /public-root/home/ubuntu/.cache/npa \
    && test ! -e /public-root/run/npa-ncore-native \
    && test ! -L /public-root/run/npa-ncore-native

FROM scratch AS public-image
ARG NPA_SOURCE_SHA
ARG SOURCE_SHA=${NPA_SOURCE_SHA}
COPY --from=assembled /public-root/ /
ENV PATH=/opt/venv/bin:/opt/ncore/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    HOME=/home/ubuntu XDG_CACHE_HOME=/home/ubuntu/.cache \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 \
    NPA_NCORE_CONVERTER=/opt/ncore/bin/colmap-convert \
    NPA_NCORE_SOURCE_REVISION=59c698d206da92b406a4f72619fce3b3a2c64bfd \
    NPA_SOURCE_SHA=${SOURCE_SHA} \
    BASH_ENV=/opt/ncore/bin/native-bootstrap.sh
USER ubuntu
WORKDIR /workspace
LABEL org.opencontainers.image.source="https://github.com/nebius/nebius-physical-ai" \
      org.opencontainers.image.revision="${SOURCE_SHA}" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT AND LicenseRef-NCore-Base-Inventory" \
      npa.base.license-inventory="/opt/ncore/base-sources/recipes/base-source-lock.json" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1" \
      npa.source_revision="${SOURCE_SHA}" \
      npa.tool="ncore" npa.runtime="cpu" \
      npa.ncore.dependencies="hash-locked-runtime-fetch" \
      npa.ncore.revision="59c698d206da92b406a4f72619fce3b3a2c64bfd"
ENTRYPOINT ["/opt/ncore/bin/entrypoint.sh"]
CMD ["/opt/venv/bin/python", "/opt/ncore/bin/verify-packaging.py"]
