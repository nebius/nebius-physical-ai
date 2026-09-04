# Operator-built SkyPilot compatibility layer for the exact PAIDF runtime.
# Model weights, operator data, credentials, and access state remain runtime-only.
FROM docker.io/vllm/vllm-omni@sha256:5d8c7e742c98858f257d82307e378391f0e7d77065e141c733cc4778042128ab

ARG NPA_SOURCE_SHA
LABEL org.opencontainers.image.source="https://github.com/nebius/nebius-physical-ai" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      npa.tool="paidf-image-edit-sky" \
      npa.upstream.digest="sha256:5d8c7e742c98858f257d82307e378391f0e7d77065e141c733cc4778042128ab" \
      npa.redistribution="operator-private-only"

USER root
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         openssh-server rsync sudo \
    && (id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu) \
    && install -d -m 0755 /run/sshd \
    && install -d -m 0755 -o ubuntu -g ubuntu /workspace /home/ubuntu/.cache \
    && sed -i '/^  start)$/a\    ssh-keygen -A' /etc/init.d/ssh \
    && grep -q 'ssh-keygen -A' /etc/init.d/ssh \
    && rm -f /etc/ssh/ssh_host_* \
    && printf '%s\n' 'PasswordAuthentication no' 'PermitRootLogin no' \
         > /etc/ssh/sshd_config.d/99-npa-container.conf \
    && printf 'ubuntu ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/npa-runtime \
    && chmod 0440 /etc/sudoers.d/npa-runtime \
    && command -v service >/dev/null \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

COPY paidf-image-edit-sky/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/entrypoint.sh

ENV HOME=/home/ubuntu \
    UV_CACHE_DIR=/home/ubuntu/.cache/uv
WORKDIR /workspace
USER ubuntu
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
