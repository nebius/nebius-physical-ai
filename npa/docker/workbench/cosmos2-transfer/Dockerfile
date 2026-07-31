# Capability wrapper for npa-cosmos2-transfer.
#
# Golden eval in this repo means "does this container actually do its job" — a
# real capability test, not a CUDA probe. So this wrapper bakes the REAL
# Cosmos-Transfer2.5 inference environment (Python 3.10 + torch cu128 +
# flash-attn + all deps via `uv sync`) on top of the upstream transfer runtime,
# and ships a golden-eval script that runs an actual video-to-video transfer on a
# bundled robot control example and asserts a generated video is produced.
#
# The upstream 2.5.x image ships the cosmos-transfer2.5 sources but does not bake
# a flash-attn-compatible venv, and pins .python-version=3.13 while the flash-attn
# wheel is cp310-only — both are corrected here.
ARG BASE_IMAGE=npa-cosmos2-transfer:2.5.0
FROM ${BASE_IMAGE}

USER root
WORKDIR /opt/cosmos/cosmos-transfer2.5

# UV_PYTHON_INSTALL_DIR keeps the interpreter out of /root. Built as root, uv
# otherwise puts it in /root/.local/share/uv/python and the venv's bin/python is a
# symlink into that tree; /root is 0700, so the image's own final USER ubuntu got
# `Permission denied` (exit 126) for every inference call and the venv was unusable
# as shipped.
ENV UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/cosmos/uv-python

# Build the real inference env. `uv sync --extra=cu128` resolves torch cu128 +
# flash-attn (cp310) + the full transfer2.5 dependency set from the upstream lock.
RUN uv python install 3.10 \
    && echo "3.10" > .python-version \
    && uv sync --extra=cu128 --python 3.10 \
    && .venv/bin/python -c "import torch, flash_attn; print('cosmos-transfer2.5 env torch', torch.__version__, 'flash_attn', flash_attn.__version__, 'cuda-build', torch.version.cuda)"

COPY --chmod=755 docker/workbench/cosmos2-transfer/smoke_functional.sh /opt/cosmos2-transfer/smoke_functional.sh
COPY --chmod=755 docker/workbench/cosmos2-transfer/entrypoint.sh /opt/cosmos2-transfer/entrypoint.sh

# openssh-server, rsync, and sudo are what SkyPilot's Kubernetes provisioner shells
# out to per pod (`$(prefix_cmd) apt install openssh-server rsync -y`, then
# `service ssh restart`, with prefix_cmd = sudo for a non-root user). Without them
# the setup script fails and the container exits before SkyPilot can exec into it.
RUN apt-get update && apt-get install -y --no-install-recommends \
      netcat-openbsd openssh-server procps rsync sudo \
    && rm -rf /var/lib/apt/lists/*

RUN id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu
RUN echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu \
    && chown -R ubuntu:ubuntu \
      /opt/cosmos2-transfer \
      /opt/cosmos/cosmos-transfer2.5/.venv \
      /opt/cosmos/uv-python

# The venv must be usable by the non-root user the image actually runs as.
RUN su ubuntu -s /bin/bash -c \
      '/opt/cosmos/cosmos-transfer2.5/.venv/bin/python -c "import torch, flash_attn; print(\"venv usable as ubuntu:\", torch.__version__)"'

USER ubuntu

# The inherited `ENTRYPOINT ["/bin/bash"]` swallowed the args Kubernetes passes, so
# SkyPilot's keep-alive command exited 126 and every GPU stage pinned to this image
# failed to provision. This entrypoint execs its arguments instead.
ENTRYPOINT ["/opt/cosmos2-transfer/entrypoint.sh"]
