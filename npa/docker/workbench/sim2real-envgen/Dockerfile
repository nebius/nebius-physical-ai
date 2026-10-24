# EnvGen generates the canonical Stage 4 scenario records.  Its source never
# imports Genesis or LeRobot, so inheriting the Genesis policy/dataset venv
# would make an unrelated upstream incompatibility block this role.  The
# source-attested parent supplies only the CUDA/Kubernetes operating-system
# contract; this Dockerfile removes inherited NPA bytes and builds the exact
# secure Python 3.12 EnvGen closure below.
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS sanitized

ARG NPA_SOURCE_SHA
ARG UBUNTU_SNAPSHOT=20261009T000000Z
ARG ENVGEN_LINUX_LIBC_DEV_VERSION=5.15.0-198.208
ARG ENVGEN_PYTHON_VERSION=3.12.14

ENV ENVGEN_VENV=/opt/npa/envgen-venv

WORKDIR /opt/npa
USER root

# The source-attested parent may itself be an NPA image.  Remove its complete
# NPA tree rather than only selected subdirectories: copying source over an
# inherited venv would retain an unreviewed dependency closure.
RUN test -d /opt/npa && test ! -L /opt/npa \
    && rm -rf /opt/npa

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu workflows /opt/npa/workflows
COPY docker/workbench/common/sim2real-envgen-requirements.txt \
    /opt/npa/sim2real-envgen-requirements.txt
COPY --chmod=0755 docker/workbench/common/workflow_runtime_entrypoint.sh \
    /usr/local/bin/npa-workflow-entrypoint
COPY --chmod=0755 docker/workbench/common/configure_ubuntu_snapshot.sh \
    /usr/local/bin/configure-ubuntu-snapshot
COPY --chmod=0755 docker/workbench/common/install_workflow_runtime_prereqs.sh \
    /usr/local/bin/install-workflow-runtime-prereqs
# Use the same digest-pinned uv distribution already used by supported
# Workbench runtimes to obtain a current CPython on the Jammy CUDA parent.
# Jammy's system interpreter is 3.10, so relying on its absent python3.12 made
# the source-attested EnvGen image unbuildable.  uv itself is build-only and is
# removed after it creates the exact interpreter-backed venv.
COPY --from=ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 \
    /uv /uvx /usr/local/bin/

# Keep the same snapshot-pinned Kubernetes bootstrap contract as the prior
# role image, but make its Python runtime independently resolver-closed.  A
# source .pth avoids an unpinned build-backend download while still proving the
# exact checked-out NPA sources import with PYTHONPATH removed. uv creates a
# minimal venv by default, so seed pip before the security-pinned bootstrap;
# otherwise a clean builder cannot execute that mandatory upgrade.
RUN /usr/local/bin/install-workflow-runtime-prereqs "${UBUNTU_SNAPSHOT}" "${ENVGEN_LINUX_LIBC_DEV_VERSION}" \
    && test "$(dpkg-query --show --showformat='${Version}' linux-libc-dev)" = "${ENVGEN_LINUX_LIBC_DEV_VERSION}" \
    && test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40 \
    && UV_PYTHON_INSTALL_DIR=/opt/npa/python UV_CACHE_DIR=/tmp/uv-cache UV_NO_CACHE=1 UV_NO_MODIFY_PATH=1 \
      uv python install "${ENVGEN_PYTHON_VERSION}" \
    && UV_PYTHON_INSTALL_DIR=/opt/npa/python UV_CACHE_DIR=/tmp/uv-cache UV_NO_CACHE=1 UV_NO_MODIFY_PATH=1 \
      uv venv --seed --python "${ENVGEN_PYTHON_VERSION}" "${ENVGEN_VENV}" \
    && "${ENVGEN_VENV}/bin/python" -c \
      'import sys; assert sys.version_info[:3] == (3, 12, 14), sys.version' \
    && "${ENVGEN_VENV}/bin/python" -m pip install --no-cache-dir --upgrade 'pip==26.2.0' 'setuptools==84.1.0' \
    && "${ENVGEN_VENV}/bin/python" -m pip install --no-cache-dir --no-deps \
      -r /opt/npa/sim2real-envgen-requirements.txt \
    && find /opt/npa/src /opt/npa/workflows -type d -exec chmod a+rx {} + \
    && find /opt/npa/src /opt/npa/workflows -type f -exec chmod a+r {} + \
    && chmod a+r /opt/npa/pyproject.toml /opt/npa/sim2real-envgen-requirements.txt \
    && "${ENVGEN_VENV}/bin/python" -m pip check \
    && "${ENVGEN_VENV}/bin/python" -c "import site; from pathlib import Path; Path(site.getsitepackages()[0], 'npa-exact-source.pth').write_text('/opt/npa/src\\n', encoding='utf-8')" \
    && env -u PYTHONPATH "${ENVGEN_VENV}/bin/python" -c \
      'from importlib import metadata; from npa.workflows import sim2real_envgen; from npa.workflows.sim2real.workflow_stage import main; assert callable(sim2real_envgen.main) and callable(main); absent = {"datasets", "lerobot", "genesis-world", "torch"}; installed = {distribution.metadata["Name"].lower() for distribution in metadata.distributions()}; assert installed.isdisjoint(absent), installed & absent' \
    && "${ENVGEN_VENV}/bin/python" -m py_compile /opt/npa/src/npa/workflows/sim2real_envgen.py \
    && rm -rf /tmp/uv-cache \
    && rm -f /usr/local/bin/uv /usr/local/bin/uvx \
    && test ! -e /usr/local/bin/uv && test ! -e /usr/local/bin/uvx \
    && test ! -e /root/.local/bin/python3.12 \
    && chown -R ubuntu:ubuntu /opt/npa

# Flatten the sanitized filesystem.  The final manifest has no dependency on
# an inherited Genesis or LeRobot layer, and its runtime interpreter is
# declared explicitly for standard SkyPilot setup.
FROM scratch AS runtime

ARG NPA_SOURCE_SHA
LABEL npa.tool="envgen" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

COPY --from=sanitized / /

ENV PATH=/opt/npa/envgen-venv/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility \
    CUDA_HOME=/usr/local/cuda \
    CUDA_VERSION=13.0.1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NPA_INPUT_PATH=/input \
    NPA_OUTPUT_PATH=/output \
    INPUT_PATH=/input \
    OUTPUT_PATH=/output \
    NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    NPA_BAKED_PYTHON=/opt/npa/envgen-venv/bin/python

WORKDIR /opt/npa
USER ubuntu
ENTRYPOINT ["/usr/local/bin/npa-workflow-entrypoint"]
