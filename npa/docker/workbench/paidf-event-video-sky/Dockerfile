# Operator-built SkyPilot compatibility layer for the exact PAIDF runtime.
# Model weights, operator data, credentials, and access state remain runtime-only.
FROM docker.io/vllm/vllm-omni@sha256:970dee6658ea223f615b2438ce41e47f1d5322225482546e6e6bc5d8134f757c

ARG NPA_SOURCE_SHA
LABEL org.opencontainers.image.source="https://github.com/nebius/nebius-physical-ai" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      npa.tool="paidf-event-video-sky" \
      npa.upstream.digest="sha256:970dee6658ea223f615b2438ce41e47f1d5322225482546e6e6bc5d8134f757c" \
      npa.redistribution="operator-private-only" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

USER root
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         openssh-server rsync sudo linux-libc-dev \
    && (id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu) \
    && install -d -m 0755 /run/sshd \
    && install -d -m 0755 -o ubuntu -g ubuntu /workspace /home/ubuntu/.cache \
    && /usr/bin/python3 -m venv /opt/npa-venv \
    && chown -R ubuntu:ubuntu /opt/npa-venv \
    && printf '%s\n' 'export PATH="/opt/npa-venv/bin:$PATH"' \
         > /etc/profile.d/npa-runtime-python.sh \
    && sed -i '/^  start)$/a\    ssh-keygen -A' /etc/init.d/ssh \
    && grep -q 'ssh-keygen -A' /etc/init.d/ssh \
    && rm -f /etc/ssh/ssh_host_* \
    && printf '%s\n' 'PasswordAuthentication no' 'PermitRootLogin no' \
         > /etc/ssh/sshd_config.d/99-npa-container.conf \
    && printf 'ubuntu ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/npa-runtime \
    && chmod 0440 /etc/sudoers.d/npa-runtime \
    && command -v service >/dev/null \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# Patch the inherited NLTK JVM argument-injection issue (CVE-2026-79675).
# The 3.10.3 patch has the same required dependency set as inherited 3.10.0.
RUN uv pip install --python /usr/bin/python3 \
      'nltk @ https://files.pythonhosted.org/packages/b6/6d/ebd2af4640b12168fdf0cb74b6118df2f32a2f62ec7e0c06fbfd80706639/nltk-3.10.3-py3-none-any.whl#sha256=ff9598a8e20518ee0d557745890cc4435b9578489e2dcbc69c4f81fa060caf7c' \
    && /usr/bin/python3 -c 'import importlib.metadata as m; assert m.version("nltk") == "3.10.3"'

COPY paidf-event-video-sky/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/entrypoint.sh

ENV HOME=/home/ubuntu \
    PATH=/opt/npa-venv/bin:${PATH} \
    UV_CACHE_DIR=/home/ubuntu/.cache/uv
WORKDIR /workspace
USER ubuntu
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
