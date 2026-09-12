FROM python:3.10-slim-bookworm@sha256:999137905e8718de681744822ccd965e1950e1baba089035060418e05e1d7496

ARG NPA_SOURCE_SHA
ARG DEBIAN_SNAPSHOT=20260824T000000Z

LABEL org.opencontainers.image.title="NPA LIBERO neutral bootstrap" \
      org.opencontainers.image.description="Public neutral bootstrap; LIBERO and GPU/runtime payloads are fetched under an external manager decision" \
      org.opencontainers.image.source="https://github.com/nebius/nebius-physical-ai" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.nebius.npa.redistribution="public-neutral-bootstrap" \
      org.nebius.npa.validation-status="quarantined-unvalidated" \
      org.nebius.npa.base-manifest="sha256:999137905e8718de681744822ccd965e1950e1baba089035060418e05e1d7496" \
      org.nebius.npa.base-rootfs-material="sha256:5ae3c39ebd15e229dcedd5cee596b2497182493d41ff162e824ba13fc1b2b867" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

COPY docker/workbench/libero/debian-packages.lock /opt/npa/libero/debian-packages.lock

RUN set -eux; \
    test -n "${NPA_SOURCE_SHA}"; \
    test "${#NPA_SOURCE_SHA}" -eq 40; \
    case "${NPA_SOURCE_SHA}" in *[!0-9a-f]*) exit 64 ;; esac; \
    printf '%s\n' \
      'Types: deb' \
      "URIs: https://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}" \
      'Suites: bookworm bookworm-updates' \
      'Components: main' \
      'Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg' \
      '' \
      'Types: deb' \
      "URIs: https://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}" \
      'Suites: bookworm-security' \
      'Components: main' \
      'Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg' \
      > /etc/apt/sources.list.d/debian.sources; \
    apt-get -o Acquire::Check-Valid-Until=false update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      $(awk -F '\t' '$1 ~ /^(base|dependency|direct)$/ { printf "%s=%s ", $2, $3 }' /opt/npa/libero/debian-packages.lock); \
    while IFS="$(printf '\t')" read -r kind package version _; do \
      case "$kind" in \
        base|dependency|direct) test "$(dpkg-query -W -f='${Version}' "$package")" = "$version" ;; \
      esac; \
    done < /opt/npa/libero/debian-packages.lock; \
    for package in openssh-server rsync sudo; do \
      test "$(dpkg-query -W -f='${Status}' "$package")" = 'install ok installed'; \
    done; \
    test "$(dpkg-query -W -f='${Version}' linux-libc-dev)" = '6.1.180-1'; \
    rm -rf /var/lib/apt/lists/*; \
    useradd --uid 1000 --create-home --shell /bin/bash ubuntu; \
    install -d -m 0700 -o ubuntu -g ubuntu /workspace /workspace/.cache /workspace/.cache/npa /workspace/.cache/npa/libero /workspace/byof-runs; \
    install -d -m 0755 /run/sshd /opt/npa/libero; \
    printf '%s\n' \
      'Cmnd_Alias NPA_SKYPILOT_SSH = /usr/sbin/service ssh start, /usr/sbin/service ssh restart, /usr/sbin/sshd' \
      'ubuntu ALL=(root) NOPASSWD: NPA_SKYPILOT_SSH' \
      > /etc/sudoers.d/npa-skypilot-ssh; \
    chmod 0440 /etc/sudoers.d/npa-skypilot-ssh; \
    visudo -cf /etc/sudoers.d/npa-skypilot-ssh

COPY docker/workbench/libero/runtime-manifest.json /opt/npa/libero/runtime-manifest.json
COPY docker/workbench/libero/runtime-requirements.txt /opt/npa/libero/runtime-requirements.txt
COPY src/npa/deploy/libero_image_manifest.json /opt/npa/libero/image-manifest.json
COPY docker/workbench/libero/runtime-bootstrap.py /opt/npa/libero/runtime-bootstrap.py
COPY docker/workbench/libero/libero_smoke.py /opt/npa/libero/libero_smoke.py
COPY docker/workbench/libero/REDISTRIBUTION.md /opt/npa/libero/REDISTRIBUTION.md
COPY docker/workbench/libero/THIRD_PARTY_NOTICES.md /opt/npa/libero/THIRD_PARTY_NOTICES.md
COPY docker/workbench/libero/skypilot-bootstrap-guard.sh /usr/local/sbin/npa-skypilot-bootstrap-guard
COPY docker/workbench/libero/entrypoint.sh /usr/local/bin/npa-libero-entrypoint
COPY docker/workbench/libero/smoke.sh /opt/npa/libero/smoke.sh

RUN set -eux; \
    echo 'cc3556e776aeaca5c73e27be3bdcfea24b97848493eb1eae2883e662fd741c76  /opt/npa/libero/runtime-manifest.json' | sha256sum -c -; \
    chmod 0755 /opt/npa/libero/runtime-bootstrap.py /opt/npa/libero/smoke.sh /usr/local/sbin/npa-skypilot-bootstrap-guard /usr/local/bin/npa-libero-entrypoint; \
    ln -s npa-skypilot-bootstrap-guard /usr/local/sbin/apt-get; \
    ln -s /usr/local/sbin/npa-skypilot-bootstrap-guard /usr/local/bin/timeout; \
    ln -s /workspace/.cache/npa/libero/current/source /opt/byof; \
    /usr/local/sbin/npa-skypilot-bootstrap-guard verify; \
    /opt/npa/libero/runtime-bootstrap.py status | grep -F '"materialized": false'; \
    test ! -e /opt/npa/libero/libero; \
    test ! -e /opt/npa/libero/torch; \
    test ! -e /workspace/.cache/npa/libero/current

ENV HOME=/home/ubuntu \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

USER ubuntu
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/npa-libero-entrypoint"]
CMD ["/bin/bash"]
