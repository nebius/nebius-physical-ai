# Operator-only compatibility layer; preserve the exact upstream service runtime.
# Source: NVIDIA/paidf-auto-labeling@36dc1114dea00d9986df97325a664520993964de.
# This recipe adds no weights, data, credentials, or access acceptance.
FROM nvcr.io/nvidia/paidf-visual-qa-service@sha256:e681c8dee849c7ac9fc5b182f51e9efd0da460972b08850d40f00aa9d5e3c97c

ARG NPA_SOURCE_SHA
LABEL org.opencontainers.image.source="https://github.com/nebius/nebius-physical-ai" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      npa.tool="paidf-visual-qa-sky" \
      npa.upstream.digest="sha256:e681c8dee849c7ac9fc5b182f51e9efd0da460972b08850d40f00aa9d5e3c97c" \
      npa.redistribution="operator-private-only"

USER root
# These commands are absent from the inspected parent. Git fetches the pinned
# orchestration configuration; SSH, rsync and sudo satisfy SkyPilot bootstrap.
# System Python/venv supplies a separate pip environment for NPA setup.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         openssh-server rsync sudo git python3 python3-venv \
    && /usr/bin/python3 -m venv /opt/npa-venv \
    && chown -R appuser:appuser /opt/npa-venv \
    && printf '%s\n' 'export PATH="/opt/npa-venv/bin:$PATH"' \
         > /etc/profile.d/npa-runtime-python.sh \
    && install -d -m 0755 /run/sshd \
    && sed -i '/^  start)$/a\    ssh-keygen -A' /etc/init.d/ssh \
    && grep -q 'ssh-keygen -A' /etc/init.d/ssh \
    && rm -f /etc/ssh/ssh_host_* \
    && printf '%s\n' 'PasswordAuthentication no' 'PermitRootLogin no' \
         > /etc/ssh/sshd_config.d/99-npa-container.conf \
    && printf 'appuser ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/npa-runtime \
    && chmod 0440 /etc/sudoers.d/npa-runtime \
    && command -v service >/dev/null \
    && test -x /app/.venv/bin/main \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

COPY paidf-visual-qa-sky/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/entrypoint.sh

# NPA installs into its own environment; never modify the vendor service venv.
# Retain upstream HOME, working directory, interpreter and non-root service user.
ENV PATH="/opt/npa-venv/bin:$PATH"
WORKDIR /app
USER appuser
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
