FROM python:3.10-slim-bookworm@sha256:999137905e8718de681744822ccd965e1950e1baba089035060418e05e1d7496

ARG NPA_SOURCE_SHA
ARG NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY_SHA256
ARG DEBIAN_SNAPSHOT=20260824T000000Z
ARG SOURCE_DATE_EPOCH

LABEL org.opencontainers.image.title="NPA LIBERO neutral bootstrap" \
      org.opencontainers.image.description="Public neutral bootstrap; LIBERO and GPU/runtime payloads require customer/run authorization" \
      org.opencontainers.image.source="https://github.com/nebius/nebius-physical-ai" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.opencontainers.image.licenses="Apache-2.0 AND LicenseRef-NPA-LIBERO-Neutral-Third-Party" \
      org.nebius.npa.third-party-notices="/opt/npa/libero/THIRD_PARTY_NOTICES.md" \
      org.nebius.npa.redistribution="public-neutral-bootstrap" \
      org.nebius.npa.validation-status="quarantined-unvalidated" \
      org.nebius.npa.output-storage-authorization-public-key-sha256="${NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY_SHA256}" \
      org.nebius.npa.base-manifest="sha256:999137905e8718de681744822ccd965e1950e1baba089035060418e05e1d7496" \
      org.nebius.npa.base-rootfs-material="sha256:5ae3c39ebd15e229dcedd5cee596b2497182493d41ff162e824ba13fc1b2b867" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

COPY docker/workbench/libero/debian-packages.lock /opt/npa/libero/debian-packages.lock

RUN set -eux; \
    test -n "${NPA_SOURCE_SHA}"; \
    test "${#NPA_SOURCE_SHA}" -eq 40; \
    case "${NPA_SOURCE_SHA}" in *[!0-9a-f]*) exit 64 ;; esac; \
    case "${SOURCE_DATE_EPOCH}" in ''|*[!0-9]*) exit 64 ;; esac; \
    test "${SOURCE_DATE_EPOCH}" -gt 0; \
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
    rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub; \
    rm -rf /var/lib/apt/lists/*; \
    groupadd --gid 1001 npa-libero-exec; \
    useradd --no-log-init --uid 1000 --create-home --shell /bin/bash ubuntu; \
    useradd --no-log-init --uid 1001 --gid npa-libero-exec --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin npa-libero-exec; \
    usermod --append --groups npa-libero-exec ubuntu; \
    chage --lastday "$((SOURCE_DATE_EPOCH / 86400))" ubuntu; \
    chage --lastday "$((SOURCE_DATE_EPOCH / 86400))" npa-libero-exec; \
    rm -rf /var/log/apt/* /var/cache/apt/archives/*; \
    rm -f /var/log/dpkg.log /var/log/alternatives.log /var/log/lastlog /var/log/faillog /var/cache/ldconfig/aux-cache; \
    install -d -m 0750 -o ubuntu -g npa-libero-exec /workspace /workspace/.cache /workspace/.cache/npa /workspace/.cache/npa/libero; \
    install -d -m 1770 -o ubuntu -g npa-libero-exec /workspace/byof-runs; \
    install -d -m 0755 /run/sshd /run/npa/libero /opt/npa/libero; \
    printf '%s\n' \
      'Cmnd_Alias NPA_SKYPILOT_SSH = /usr/local/bin/ssh-keygen -A, /usr/sbin/service ssh start, /usr/sbin/service ssh restart' \
      'Cmnd_Alias NPA_SKYPILOT_APT = /usr/local/sbin/apt-get update, /usr/local/sbin/apt-get update *, /usr/local/sbin/apt-get install, /usr/local/sbin/apt-get install *' \
      'Cmnd_Alias NPA_LIBERO_EXEC = /opt/npa/libero/runtime-bootstrap.py execute' \
      'Defaults!NPA_LIBERO_EXEC closefrom_override' \
      'Defaults!NPA_LIBERO_EXEC env_keep += "NPA_BYOF_RUN_ID NPA_SMOKE_OUTPUT_DIR NPA_LIBERO_BOOTSTRAP_RECEIPT NPA_LIBERO_CUSTOMER_AUTHORIZATION_SHA256 NPA_LIBERO_AUTHENTICATED_CALLER_SHA256 NPA_LIBERO_CUSTOMER_IDENTITY_SHA256 NPA_LIBERO_EXPECTED_EXECUTABLE_PROFILE_SHA256 NPA_LIBERO_EXPECTED_CUSTOMER_AUTHORIZATION_EXPIRES_AT NPA_LIBERO_EXPECTED_CUSTOMER_SIGNER_PUBLIC_KEY_SHA256 NPA_LIBERO_EXPECTED_CANONICAL_BUILD_METADATA_SHA256 NPA_LIBERO_EXPECTED_PUBLICATION_BUNDLE_SHA256 NPA_LIBERO_EXPECTED_INFRASTRUCTURE_BUNDLE_SHA256 NPA_LIBERO_EXPECTED_ALLOWED_NODE_SHA256 NPA_LIBERO_EXPECTED_CLUSTER_IDENTITY_SHA256 NPA_LIBERO_EXPECTED_NAMESPACE_SHA256 NPA_LIBERO_EXPECTED_NAMESPACE_UID_SHA256 NPA_LIBERO_EXPECTED_NAMESPACE_INVENTORY_SHA256 NPA_LIBERO_EXPECTED_EXTERNAL_RBAC_INVENTORY_SHA256 NPA_LIBERO_EXPECTED_RBAC_SPEC_SHA256 NPA_LIBERO_EXPECTED_ROLE_BINDING_UID_SHA256 NPA_LIBERO_EXPECTED_ROLE_UID_SHA256 NPA_LIBERO_EXPECTED_SERVICE_ACCOUNT_UID_SHA256 NPA_LIBERO_EXPECTED_PAYLOAD_KUBECONFIG_SHA256 NPA_LIBERO_EXPECTED_EXECUTION_KUBECONFIG_SHA256 NPA_LIBERO_EXPECTED_SKYPILOT_CONFIG_SHA256 BYOF_REPO_ROOT BYOF_SMOKE_COMMAND BYOF_SOLUTION_NAME BYOF_CAPABILITY_NAME BYOF_SMOKE_ARTIFACT_NAME BYOF_IMAGE LIBERO_RUNTIME_ROOT NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES HOSTNAME KUBERNETES_SERVICE_HOST KUBERNETES_SERVICE_PORT"' \
      'ubuntu ALL=(root) NOPASSWD: NPA_SKYPILOT_SSH' \
      'ubuntu ALL=(root) NOPASSWD: NPA_SKYPILOT_APT' \
      'ubuntu ALL=(npa-libero-exec) NOPASSWD: NPA_LIBERO_EXEC' \
      > /etc/sudoers.d/npa-skypilot-ssh; \
    chmod 0440 /etc/sudoers.d/npa-skypilot-ssh; \
    visudo -cf /etc/sudoers.d/npa-skypilot-ssh

# The storage control plane uses a fixed public verification key. Customer
# signer material remains owner-private and runtime-only; no customer key is
# accepted by this build or written into an image layer.
RUN --mount=type=secret,id=npa_libero_output_storage_authorization_public_key_b64,required=true \
    set -eux; \
    tr -d '\r\n' \
      < /run/secrets/npa_libero_output_storage_authorization_public_key_b64 \
      > /opt/npa/libero/output-storage-authorization-public-key.b64; \
    base64 -d /opt/npa/libero/output-storage-authorization-public-key.b64 \
      > /tmp/npa-libero-output-storage-authorization-public-key.raw; \
    test "$(wc -c < /tmp/npa-libero-output-storage-authorization-public-key.raw)" = 32; \
    test -n "${NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY_SHA256}"; \
    test "$(printf '%s' "${NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY_SHA256}" | grep -Ec '^[0-9a-f]{64}$')" = 1; \
    test "$(sha256sum /tmp/npa-libero-output-storage-authorization-public-key.raw | cut -d' ' -f1)" = \
      "${NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY_SHA256}"; \
    test "$(base64 -w 0 < /tmp/npa-libero-output-storage-authorization-public-key.raw)" = \
      "$(cat /opt/npa/libero/output-storage-authorization-public-key.b64)"; \
    rm -f /tmp/npa-libero-output-storage-authorization-public-key.raw; \
    chown root:root /opt/npa/libero/output-storage-authorization-public-key.b64; \
    chmod 0444 /opt/npa/libero/output-storage-authorization-public-key.b64

COPY docker/workbench/libero/runtime-manifest.json /opt/npa/libero/runtime-manifest.json
COPY docker/workbench/libero/runtime-requirements.txt /opt/npa/libero/runtime-requirements.txt
COPY docker/workbench/libero/runtime-bootstrap.py /opt/npa/libero/runtime-bootstrap.py
COPY docker/workbench/libero/libero_smoke.py /opt/npa/libero/libero_smoke.py
COPY docker/workbench/libero/REDISTRIBUTION.md /opt/npa/libero/REDISTRIBUTION.md
COPY docker/workbench/libero/THIRD_PARTY_NOTICES.md /opt/npa/libero/THIRD_PARTY_NOTICES.md
COPY docker/workbench/libero/skypilot-bootstrap-guard.sh /usr/local/sbin/npa-skypilot-bootstrap-guard
COPY docker/workbench/libero/entrypoint.sh /usr/local/bin/npa-libero-entrypoint
COPY docker/workbench/libero/smoke.sh /opt/npa/libero/smoke.sh

RUN set -eux; \
    echo 'a319f5ac5fbd0eeb1390940eeda62828d621cd7dd0dd828789182024562c5b44  /opt/npa/libero/runtime-manifest.json' | sha256sum -c -; \
    echo '8504f236dcad67ad0e2f5959b916c93aa7ccbd02567c6e323e480366d0f23b99  /opt/npa/libero/runtime-requirements.txt' | sha256sum -c -; \
    chmod 0755 /opt/npa/libero/runtime-bootstrap.py /opt/npa/libero/smoke.sh /usr/local/sbin/npa-skypilot-bootstrap-guard /usr/local/bin/npa-libero-entrypoint; \
    ln -s npa-skypilot-bootstrap-guard /usr/local/sbin/apt-get; \
    ln -s /usr/local/sbin/npa-skypilot-bootstrap-guard /usr/local/bin/timeout; \
    ln -s /usr/local/sbin/npa-skypilot-bootstrap-guard /usr/local/bin/ssh-keygen; \
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
