# syntax=docker/dockerfile:1.7
# Operator-private CUDA build image.  It deliberately contains no EmbodiedGen,
# TRELLIS, model, input, cache, output, or credential bytes.
ARG BASE_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66
FROM ${BASE_IMAGE}

ARG NPA_SOURCE_SHA
ARG UBUNTU_SNAPSHOT=20261004T000000Z
ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/ubuntu \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# The CUDA development image supplies nvcc for upstream TRELLIS extensions.
# These ordinary worker packages are acquired from an immutable Ubuntu snapshot;
# the application and every model byte remain a non-root runtime fetch.
# Do not mix that snapshot with inherited CUDA-base APT definitions. Ubuntu 24.04
# bases use deb822 ubuntu.sources, so remove both legacy and deb822 files first.
RUN set -eux; \
    rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; \
    printf '%s\n' \
      'Types: deb' \
      "URIs: https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/" \
      'Suites: noble noble-updates noble-security' \
      'Components: main universe' \
      'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg' \
      'Check-Valid-Until: no' > /etc/apt/sources.list.d/npa-snapshot.sources; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates git libegl1 libgl1 libglib2.0-0 linux-libc-dev openssh-server \
      python3 python3-dev python3-pip python3-venv rsync sudo util-linux; \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*; \
    rm -f /etc/ssh/ssh_host_*; \
    if getent passwd ubuntu >/dev/null; then \
      test "$(id -u ubuntu)" = 1000; \
      test "$(id -g ubuntu)" = 1000; \
      test "$(getent passwd ubuntu | cut -d: -f6)" = /home/ubuntu; \
      test "$(stat -c '%u:%g' /home/ubuntu)" = 1000:1000; \
    else \
      test ! -e /home/ubuntu; \
      useradd --uid 1000 --create-home --shell /bin/bash ubuntu; \
      test "$(id -g ubuntu)" = 1000; \
      test "$(stat -c '%u:%g' /home/ubuntu)" = 1000:1000; \
    fi; \
    install -d -m 0755 /workspace /run/sshd /opt/byof /opt/npa/embodiedgen; \
    chown ubuntu:ubuntu /workspace; \
    printf '%s\n' \
      '{"source":"operator-runtime-fetch","repo":"https://github.com/HorizonRobotics/EmbodiedGen.git","ref":"f0124197888c2b733e4eaa65acd81ad9cfda3b79"}' \
      > /opt/byof/npa_source_metadata.json; \
    chmod 0444 /opt/byof/npa_source_metadata.json; \
    printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/npa-skypilot; \
    chmod 0440 /etc/sudoers.d/npa-skypilot; \
    printf '%s\n' 'PasswordAuthentication no' 'PermitRootLogin no' \
      > /etc/ssh/sshd_config.d/npa-runtime.conf

COPY --chmod=0444 docker/workbench/embodiedgen/runtime-manifest.json \
     docker/workbench/embodiedgen/REDISTRIBUTION.md \
     docker/workbench/embodiedgen/THIRD_PARTY_NOTICES.md /opt/npa/embodiedgen/
COPY --chmod=0555 docker/workbench/embodiedgen/runtime-bootstrap.py \
    docker/workbench/embodiedgen/capability_smoke.py \
    docker/workbench/embodiedgen/asset_bundle.py /opt/npa/embodiedgen/
COPY --chmod=0555 docker/workbench/embodiedgen/entrypoint.sh \
     /usr/local/bin/npa-embodiedgen-entrypoint

RUN set -eux; \
    test -n "${NPA_SOURCE_SHA}"; \
    printf '%s' "${NPA_SOURCE_SHA}" | grep -Eq '^[0-9a-f]{40}$'; \
    chown -R root:root /opt/byof /opt/npa/embodiedgen /usr/local/bin/npa-embodiedgen-entrypoint; \
    chmod -R go-w /opt/byof /opt/npa/embodiedgen /usr/local/bin/npa-embodiedgen-entrypoint; \
    test ! -e /opt/npa/embodiedgen/runtime-cache; \
    test ! -e /opt/npa/embodiedgen/EmbodiedGen; \
    test ! -e /opt/npa/embodiedgen/TRELLIS

LABEL org.opencontainers.image.source="https://github.com/nebius/nebius-physical-ai" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.opencontainers.image.licenses="Apache-2.0 AND LicenseRef-NVIDIA-CUDA-EULA" \
      org.nebius.npa.delivery="operator-private-runtime-fetch" \
      org.nebius.npa.redistribution="operator-private" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

USER ubuntu
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/npa-embodiedgen-entrypoint"]
CMD ["health"]
