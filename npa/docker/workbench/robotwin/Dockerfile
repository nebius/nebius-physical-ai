# syntax=docker/dockerfile:1.7
# Neutral public-development recipe. This exact official Ubuntu base and the
# snapshot-locked Ubuntu main closure contain no RoboTwin/NVIDIA/runtime data.
# The trusted build entrypoint remains fail-closed while the native-byte policy
# is unresolved; this Dockerfile is not a publication or runtime-use grant.
FROM ubuntu:22.04@sha256:281c5745f657873d78e5531fc5ba8575f46ab7769b94550ac99543f122679986 AS runtime

ARG NPA_SOURCE_SHA
ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/ubuntu \
    PATH=/opt/npa/robotwin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --chmod=0644 docker/workbench/robotwin/apt-packages.lock \
  docker/workbench/robotwin/runtime-requirements.lock /opt/npa/robotwin/

# Install only the exact Ubuntu main packages selected from the signed immutable
# snapshot. The locked direct set is ca-certificates, openssh-server,
# python3-minimal, rsync, and sudo. The base keyring, all three InRelease files,
# every installed version, and every retained package copyright file are bound
# to the committed lock.
RUN set -eux; \
    test "$(sha256sum /usr/share/keyrings/ubuntu-archive-keyring.gpg | cut -d ' ' -f1)" = "1a4dd63e5c76728960a2edddae22e2e0fc53df8e8b87806deb971030ac704eb0"; \
    test "$(sha256sum /opt/npa/robotwin/apt-packages.lock | cut -d ' ' -f1)" = "554ae7df404eb46788608a5b7d16e20c9d8f055e2678294b2f03b82a9caf666d"; \
    test "$(sha256sum /opt/npa/robotwin/runtime-requirements.lock | cut -d ' ' -f1)" = "cce5aad788880905cc7b6755c66d294577a68b3b8ea8eeb5b1eb96a997b041f2"; \
    rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; \
    printf '%s\n' \
      'Types: deb deb-src' \
      'URIs: https://snapshot.ubuntu.com/ubuntu/20260912T000000Z/' \
      'Suites: jammy jammy-updates jammy-security' \
      'Components: main' \
      'Architectures: amd64' \
      'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg' \
      > /etc/apt/sources.list.d/ubuntu.sources; \
    printf '%s\n' 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99npa-snapshot; \
    apt-get update; \
    awk -F '\t' '$1 == "inrelease" { print $3 "  /var/lib/apt/lists/snapshot.ubuntu.com_ubuntu_20260912T000000Z_dists_" $2 "_InRelease" }' \
      /opt/npa/robotwin/apt-packages.lock | sha256sum -c -; \
    packages="$(awk -F '\t' '$1 == "binary" { printf "%s=%s ", $2, $3 }' /opt/npa/robotwin/apt-packages.lock)"; \
    test "$(awk -F '\t' '$1 == "binary" { count += 1 } END { print count + 0 }' /opt/npa/robotwin/apt-packages.lock)" = 75; \
    apt-get install -y --no-install-recommends --allow-downgrades ${packages}; \
    while IFS="$(printf '\t')" read -r kind name version architecture archive_sha archive_size archive_path source source_version relationship copyright_sha copyright_path annotation; do \
      test "${kind}" = binary || continue; \
      test "$(dpkg-query -W -f='${Version}' "${name}")" = "${version}"; \
      test "$(dpkg-query -W -f='${Architecture}' "${name}")" = "${architecture}"; \
      test "$(sha256sum "/${copyright_path}" | cut -d ' ' -f1)" = "${copyright_sha}"; \
    done < /opt/npa/robotwin/apt-packages.lock; \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*.deb; \
    rm -f /etc/ssh/ssh_host_*; \
    install -d -m 0755 /run/sshd; \
    printf '%s\n' 'PasswordAuthentication no' 'PermitRootLogin no' > /etc/ssh/sshd_config.d/npa.conf; \
    useradd -m -s /bin/bash -u 1000 ubuntu; \
    install -d -o ubuntu -g ubuntu /workspace /home/ubuntu/.cache; \
    printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' >/etc/sudoers.d/ubuntu; \
    chmod 0440 /etc/sudoers.d/ubuntu; \
    visudo -cf /etc/sudoers.d/ubuntu

COPY --chmod=0644 docker/workbench/robotwin/runtime-lock.json \
  docker/workbench/robotwin/runtime-requirements.in \
  docker/workbench/robotwin/REDISTRIBUTION.md \
  docker/workbench/robotwin/THIRD_PARTY_NOTICES.md /opt/npa/robotwin/
COPY --chmod=0755 docker/workbench/robotwin/robotwin_runtime.py /opt/npa/robotwin/robotwin-runtime
COPY --chmod=0755 docker/workbench/robotwin/entrypoint.sh /opt/npa/robotwin/entrypoint.sh
COPY --chmod=0755 docker/workbench/robotwin/smoke.sh /opt/npa/robotwin/smoke.sh

LABEL org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1" \
      org.nebius.npa.source-sha="${NPA_SOURCE_SHA}" \
      org.nebius.npa.payload="zero-vendor-payload" \
      org.nebius.npa.publication-state="neutral-recipe-unbuilt-quarantined"

# TEST: exercise the declared non-root SkyPilot bootstrap contract, remove all
# generated host keys, and prove the real runtime gate still refuses before work.
# trivy:ignore:AVD-DS-0010
RUN su ubuntu -s /bin/bash -c 'sudo -n ssh-keygen -A && sudo -n /usr/sbin/sshd -t && sudo -n service ssh restart && sudo -n service ssh stop && rsync --version >/dev/null && test -w /tmp && test -w "$HOME"' \
    && rm -f /etc/ssh/ssh_host_* \
    && test -z "$(find /etc/ssh -name 'ssh_host_*' -print -quit)" \
    && env -u NPA_INTERNAL_BYOF_ROBOTWIN_RUNTIME_AUTH_V1 \
      /opt/npa/robotwin/robotwin-runtime assert-refusal

USER ubuntu
WORKDIR /workspace
ENTRYPOINT ["/opt/npa/robotwin/entrypoint.sh"]
CMD ["/bin/bash"]
