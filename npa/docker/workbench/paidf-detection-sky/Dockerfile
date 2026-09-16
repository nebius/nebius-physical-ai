# Operator-only compatibility layer; preserve the exact upstream service runtime.
# Source: NVIDIA/paidf-auto-labeling@36dc1114dea00d9986df97325a664520993964de.
# This recipe adds no weights, data, credentials, or access acceptance.
FROM nvcr.io/nvidia/paidf-detection-and-tracking-rfdetr-service@sha256:6b35e63b95cab7cd772906bcb08be978de7526427f0d1925ab84439dd4a9561e

ARG NPA_SOURCE_SHA
LABEL org.opencontainers.image.source="https://github.com/nebius/nebius-physical-ai" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      npa.tool="paidf-detection-sky" \
      npa.upstream.digest="sha256:6b35e63b95cab7cd772906bcb08be978de7526427f0d1925ab84439dd4a9561e" \
      npa.redistribution="operator-private-only" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

USER root
# RF-DETR's published inference extra uses prebuilt PyTorch operations. W&B is
# only a logger extra; its inherited torchtitan consumer is a training tool.
# Remove those unused packages and the dependency-linked development toolchain
# with normal package management, preserving the upstream service and CUDA
# runtime libraries. The exact service CLI and pretrained eager forward were
# checked after removal; training, export and native-extension builds are not
# capabilities of this detection worker.
RUN /usr/bin/python3 -m pip uninstall -y wandb torchtitan \
    && DEBIAN_FRONTEND=noninteractive apt-get purge -y linux-libc-dev \
    && /app/.venv/bin/python -c \
         'import importlib.util; assert importlib.util.find_spec("wandb") is None; assert importlib.util.find_spec("torchtitan") is None' \
    && ! dpkg-query -W -f='${db:Status-Status}' linux-libc-dev 2>/dev/null | grep -qx installed

# SSH, rsync and sudo are absent from the inspected parent. Its existing Git
# fetches the pinned orchestration configuration without replacing vendor code.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         openssh-server rsync sudo python3 python3-venv \
    && /usr/bin/python3 -m venv /opt/npa-venv \
    && chown -R ubuntu:ubuntu /opt/npa-venv \
    && printf '%s\n' 'export PATH="/opt/npa-venv/bin:$PATH"' \
         > /etc/profile.d/npa-runtime-python.sh \
    && install -d -m 0755 /run/sshd \
    && sed -i '/^  start)$/a\    ssh-keygen -A' /etc/init.d/ssh \
    && grep -q 'ssh-keygen -A' /etc/init.d/ssh \
    && rm -f /etc/ssh/ssh_host_* \
    && printf '%s\n' 'PasswordAuthentication no' 'PermitRootLogin no' \
         > /etc/ssh/sshd_config.d/99-npa-container.conf \
    && printf 'ubuntu ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/npa-runtime \
    && chmod 0440 /etc/sudoers.d/npa-runtime \
    && command -v service >/dev/null \
    && test -x /app/.venv/bin/main \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

COPY paidf-detection-sky/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/entrypoint.sh

# NPA installs into its own environment; never modify the vendor service venv.
# Retain upstream HOME, working directory, interpreter and non-root service user.
ENV PATH="/opt/npa-venv/bin:$PATH"
WORKDIR /app
USER ubuntu
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
