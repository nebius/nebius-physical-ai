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
    UV_PYTHON_INSTALL_DIR=/opt/cosmos/uv-python \
    UV_LEGACY_PYTHON_DIR=/root/.local/share/uv/python

# Build the real inference env. `uv sync --extra=cu128` resolves torch cu128 +
# flash-attn (cp310) + the full transfer2.5 dependency set from the upstream lock.
#
# Two paths, because BASE_IMAGE may be either the upstream runtime or an
# already-built transfer image (the upstream 2.5.0 base is not always available to
# rebuild from):
#
#   fresh base      -> uv installs the interpreter into UV_PYTHON_INSTALL_DIR and
#                      syncs the venv against it.
#   built artifact  -> the venv already holds the right packages but its bin/python
#                      symlinks into /root, so relocate the interpreter and repoint
#                      the venv at it. Re-running `uv sync` here would re-download
#                      the whole cu128 torch set for no benefit.
#
# Either way the layer ends with the interpreter outside /root, which is what makes
# the venv usable by the non-root user this image runs as.
RUN set -eux; \
    resolved="$(readlink -f .venv/bin/python 2>/dev/null || true)"; \
    case "$resolved" in \
      "${UV_LEGACY_PYTHON_DIR}"/*) relocate=1 ;; \
      *) relocate=0 ;; \
    esac; \
    if [ "$relocate" = 1 ]; then \
      old_root="${resolved%/bin/*}"; \
      new_root="${UV_PYTHON_INSTALL_DIR}/$(basename "$old_root")"; \
      mkdir -p "${UV_PYTHON_INSTALL_DIR}"; \
      cp -a "${UV_LEGACY_PYTHON_DIR}/." "${UV_PYTHON_INSTALL_DIR}/"; \
      # uv's tree contains an absolute version symlink (cpython-3.10-... ->
      # cpython-3.10.20-...), and `cp -a` copies such links verbatim — so the copy
      # still points back into the legacy tree and resolving through it lands in
      # /root again. Repoint every absolute link that targets the old location.
      find "${UV_PYTHON_INSTALL_DIR}" -type l | while read -r link; do \
        target="$(readlink "$link")"; \
        case "$target" in \
          "${UV_LEGACY_PYTHON_DIR}"/*) \
            ln -sfn "${UV_PYTHON_INSTALL_DIR}/${target#"${UV_LEGACY_PYTHON_DIR}/"}" "$link" ;; \
        esac; \
      done; \
      for link in .venv/bin/python .venv/bin/python3 .venv/bin/python3.10; do \
        [ -L "$link" ] || continue; \
        target="$(readlink "$link")"; \
        case "$target" in \
          "${UV_LEGACY_PYTHON_DIR}"/*) ln -sfn "${new_root}/bin/$(basename "$target")" "$link" ;; \
        esac; \
      done; \
      # Rewrite by directory prefix, not by the resolved path: pyvenv.cfg records the
      # interpreter through uv's version symlink (cpython-3.10-...) while readlink -f
      # resolves to the real directory (cpython-3.10.20-...). Matching only the
      # resolved form leaves `home =` pointing into the legacy tree, and Python takes
      # sys._home from it — which surfaces much later as a PermissionError from
      # distutils.sysconfig rather than from the venv itself. `cp -a` copied both the
      # real directory and the symlink, so a prefix swap is correct for either form.
      sed -i "s|${UV_LEGACY_PYTHON_DIR}|${UV_PYTHON_INSTALL_DIR}|g" .venv/pyvenv.cfg; \
      rm -rf "${UV_LEGACY_PYTHON_DIR}"; \
    else \
      uv python install 3.10; \
      echo "3.10" > .python-version; \
      uv sync --extra=cu128 --python 3.10; \
    fi; \
    case "$(readlink -f .venv/bin/python)" in \
      "${UV_LEGACY_PYTHON_DIR}"/*) echo "interpreter still under ${UV_LEGACY_PYTHON_DIR}" >&2; exit 1 ;; \
    esac; \
    grep -q "${UV_LEGACY_PYTHON_DIR}" .venv/pyvenv.cfg \
      && { echo "pyvenv.cfg still points at ${UV_LEGACY_PYTHON_DIR}" >&2; exit 1; } || true; \
    if [ -d "${UV_PYTHON_INSTALL_DIR}" ] \
       && find "${UV_PYTHON_INSTALL_DIR}" -type l -lname "${UV_LEGACY_PYTHON_DIR}/*" | grep -q .; then \
      echo "relocated tree still links back into ${UV_LEGACY_PYTHON_DIR}" >&2; exit 1; \
    fi

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
      "${UV_PYTHON_INSTALL_DIR}"

# The venv must be usable by the non-root user the image actually runs as.
#
# `import torch` alone is too weak a check: it never reads sys._home, so a
# pyvenv.cfg still pointing into /root passes here and then fails deep in an
# inference run (megatron -> torch.utils.cpp_extension -> setuptools ->
# distutils.sysconfig -> PermissionError). Exercising distutils.sysconfig is what
# actually proves the relocation is complete.
RUN su ubuntu -s /bin/bash -c \
      '/opt/cosmos/cosmos-transfer2.5/.venv/bin/python -c "\
import sys, torch, flash_attn; \
from distutils.sysconfig import customize_compiler, get_config_var; \
home = sys._home or \"\"; \
assert \"/root\" not in home, home; \
get_config_var(\"prefix\"); \
print(\"venv usable as ubuntu:\", torch.__version__, \"home=\", home)"'

USER ubuntu

# The inherited `ENTRYPOINT ["/bin/bash"]` swallowed the args Kubernetes passes, so
# SkyPilot's keep-alive command exited 126 and every GPU stage pinned to this image
# failed to provision. This entrypoint execs its arguments instead.
ENTRYPOINT ["/opt/cosmos2-transfer/entrypoint.sh"]
