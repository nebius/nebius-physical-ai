# Operator-built SkyPilot wrapper around the exact public benchmark runtime.
# The inherited vLLM-Omni/CUDA payload remains byte-identical; this layer adds
# only the worker bootstrap commands required by SkyPilot 0.12.2.
ARG BASE_IMAGE=docker.io/vllm/vllm-omni:cosmos3@sha256:6d2630c7d637b699557573f2c3fee8df5d4d0cd718977aa22549ed6a6ef30587
FROM ${BASE_IMAGE}

ARG NPA_SOURCE_COMMIT=unknown

LABEL org.opencontainers.image.source="https://github.com/nebius/nebius-physical-ai" \
      org.opencontainers.image.licenses="LicenseRef-NVIDIA-Software-License" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1" \
      npa.tool="cosmos3-super-benchmark" \
      npa.source.commit="${NPA_SOURCE_COMMIT}" \
      npa.upstream.image="docker.io/vllm/vllm-omni:cosmos3" \
      npa.upstream.digest="sha256:6d2630c7d637b699557573f2c3fee8df5d4d0cd718977aa22549ed6a6ef30587" \
      npa.redistribution="operator-private-only"

USER root
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      openssh-server rsync sudo \
    && rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* \
    && (id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu) \
    && install -d -m 0755 -o ubuntu -g ubuntu /workspace /home/ubuntu/.cache \
    && printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/npa-runtime \
    && chmod 0440 /etc/sudoers.d/npa-runtime \
    && dpkg-query -W openssh-server rsync sudo \
    && command -v service >/dev/null

COPY --chmod=0755 docker/workbench/cosmos3-super-benchmark/entrypoint.sh /usr/local/bin/npa-cosmos3-super-benchmark-entrypoint

ENV HOME=/home/ubuntu \
    UV_CACHE_DIR=/home/ubuntu/.cache/uv
WORKDIR /workspace
USER ubuntu
ENTRYPOINT ["/usr/local/bin/npa-cosmos3-super-benchmark-entrypoint"]
