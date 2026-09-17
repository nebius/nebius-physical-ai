# Enactic OpenArm: Apache-2.0 MuJoCo + runtime-fetched Isaac Sim/Isaac Lab.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@sha256:ad6d59a3bbf3e82c1c849c9ac09cfc2a3e0bbb8655042fd899be6681b3fe2a85

ARG OPENARM_MUJOCO_COMMIT=a8c979629f2591ad035d99d338ce114969e6cddc
ARG OPENARM_ISAAC_COMMIT=bad82e23716e6941c2de78ccb978f57c78b37734
ARG ISAAC_LAB_SRC_COMMIT=37ddf626871758333d6ed89cf64ad702aef127d0
ARG ISAAC_LAB_VERSION=2.3.2.post1
ARG ISAAC_SIM_VERSION=5.1.0.0
ARG NPA_SOURCE_SHA
ARG NPA_RUNTIME_USER=ubuntu
ARG OPENARM_GNUPG_VERSION=2.2.27-3ubuntu2.5
ARG OPENARM_OPENSSL_VERSION=3.0.2-0ubuntu1.26
ARG OPENARM_LINUX_LIBC_DEV_VERSION=5.15.0-190.200

LABEL npa.tool="openarm" \
      npa.version="2.2.0-isaac0.1.0-rtfetch" \
      npa.openarm_mujoco_commit="${OPENARM_MUJOCO_COMMIT}" \
      npa.openarm_isaac_commit="${OPENARM_ISAAC_COMMIT}" \
      npa.isaac_provisioning="runtime-fetch" \
      npa.architecture="no-baked-omniverse-kit" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

ENV NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    NPA_SKIP_EAGER_IMPORTS=1 \
    NPA_LIGHT_WORKBENCH_TOOL=openarm \
    NPA_BAKED_PYTHON=/opt/npa/sim/venv/bin/python \
    NPA_ISAAC_BASE_PYTHON=/opt/npa/sim/venv/bin/python \
    NPA_ISAAC_CACHE_DIR=/opt/isaac-cache \
    NPA_ISAAC_WHEELS_FILE=/opt/npa/docker/workbench/common/isaac-nvidia-wheels.txt \
    NPA_ISAAC_LAB_SRC_COMMIT=${ISAAC_LAB_SRC_COMMIT} \
    ISAAC_LAB_VERSION=${ISAAC_LAB_VERSION} \
    ISAAC_SIM_VERSION=${ISAAC_SIM_VERSION} \
    ISAAC_LAB_PYTHON=/isaac-sim/python.sh \
    NPA_ISAAC_LAB_WORKDIR=/workspace/isaaclab \
    OPENARM_MUJOCO_COMMIT=${OPENARM_MUJOCO_COMMIT} \
    OPENARM_ISAAC_COMMIT=${OPENARM_ISAAC_COMMIT} \
    OPENARM_ISAAC_ROOT=/opt/openarm/isaac \
    OPENARM_ISAAC_RUNNER=/opt/npa/openarm/isaac_runner.py \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_DRIVER_CAPABILITIES=all \
    IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg \
    PYTHONPATH=/opt/npa/src:/opt/openarm/isaac:/opt/openarm/isaac/source/openarm \
    PATH=/opt/npa/sim/venv/bin:${PATH} \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-lc"]
USER root

COPY docker/workbench/common /opt/npa/docker/workbench/common
COPY --chmod=0755 docker/workbench/common/workflow_runtime_entrypoint.sh /usr/local/bin/npa-workflow-entrypoint
COPY --chmod=0755 docker/workbench/isaac-lab/npa_cli.sh /usr/local/bin/npa
COPY docker/workbench/openarm/mujoco-requirements.txt /opt/npa/openarm/mujoco-requirements.txt
COPY docker/workbench/openarm/security-requirements.txt /opt/npa/openarm/security-requirements.txt
COPY --chmod=0755 docker/workbench/openarm/isaac_runner.py /opt/npa/openarm/isaac_runner.py
COPY --chmod=0755 docker/workbench/openarm/emit_license_inventory.py /opt/npa/openarm/emit_license_inventory.py
COPY --chmod=0755 docker/workbench/openarm/prune_python_vendor_devel.py /opt/npa/openarm/prune_python_vendor_devel.py
COPY --chmod=0444 docker/workbench/openarm/components.json /usr/share/doc/npa-openarm/components.json
COPY --chmod=0444 docker/workbench/openarm/REDISTRIBUTION.md /usr/share/doc/npa-openarm/REDISTRIBUTION.md
COPY --chmod=0444 docker/workbench/openarm/THIRD_PARTY_NOTICES.md /usr/share/doc/npa-openarm/THIRD_PARTY_NOTICES.md
COPY --chmod=0444 docker/workbench/openarm/ONBOARDING_CONTRACT.md /usr/share/doc/npa-openarm/ONBOARDING_CONTRACT.md
COPY pyproject.toml /opt/npa/pyproject.toml
COPY src/npa /opt/npa/src/npa
COPY workflows /opt/npa/workflows

RUN find /opt/npa/docker/workbench/common -type d -exec chmod 0755 {} + \
    && find /opt/npa/docker/workbench/common -type f -exec chmod 0644 {} + \
    && chmod 0755 /opt/npa/docker/workbench/common/*.sh \
    && test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40 \
    && NPA_RUNTIME_USER="${NPA_RUNTIME_USER}" \
       NPA_ISAAC_VENV=/opt/npa/sim/venv \
       NPA_LINUX_LIBC_DEV_VERSION="${OPENARM_LINUX_LIBC_DEV_VERSION}" \
       /opt/npa/docker/workbench/common/install_isaac_runtime_base.sh \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       "gnupg=${OPENARM_GNUPG_VERSION}" \
       "gnupg2=${OPENARM_GNUPG_VERSION}" \
       "libssl3=${OPENARM_OPENSSL_VERSION}" \
       "openssl=${OPENARM_OPENSSL_VERSION}" \
    && test "$(dpkg-query -W -f='${Version}' gnupg)" = "${OPENARM_GNUPG_VERSION}" \
    && test "$(dpkg-query -W -f='${Version}' gnupg2)" = "${OPENARM_GNUPG_VERSION}" \
    && test "$(dpkg-query -W -f='${Version}' libssl3)" = "${OPENARM_OPENSSL_VERSION}" \
    && test "$(dpkg-query -W -f='${Version}' openssl)" = "${OPENARM_OPENSSL_VERSION}" \
    && test "$(dpkg-query -W -f='${Version}' linux-libc-dev)" = "${OPENARM_LINUX_LIBC_DEV_VERSION}" \
    && rm -f /etc/ssh/ssh_host_* \
    && test -z "$(find /etc/ssh -maxdepth 1 -type f -name 'ssh_host_*' -print -quit)" \
    && rm -rf /var/lib/apt/lists/*

# The reviewed hash lock is additive to the exact shared Isaac prerequisite closure.
RUN /opt/npa/sim/venv/bin/python -m pip install --no-cache-dir --require-hashes \
      -r /opt/npa/openarm/mujoco-requirements.txt \
    && /opt/npa/sim/venv/bin/python -m pip install --no-cache-dir --require-hashes \
      -r /opt/npa/openarm/security-requirements.txt \
    && /opt/npa/sim/venv/bin/python -m pip install --no-cache-dir --no-deps wheel==0.47.0 \
    && git clone --filter=blob:none --no-checkout https://github.com/enactic/openarm_mujoco.git /opt/openarm/mujoco \
    && git -C /opt/openarm/mujoco checkout "${OPENARM_MUJOCO_COMMIT}" \
    && test "$(git -C /opt/openarm/mujoco rev-parse HEAD)" = "${OPENARM_MUJOCO_COMMIT}" \
    && /opt/npa/sim/venv/bin/python -m pip install --no-cache-dir --no-deps --no-build-isolation /opt/openarm/mujoco \
    && git clone --filter=blob:none --no-checkout https://github.com/enactic/openarm_isaac_lab.git /opt/openarm/isaac \
    && git -C /opt/openarm/isaac checkout "${OPENARM_ISAAC_COMMIT}" \
    && test "$(git -C /opt/openarm/isaac rev-parse HEAD)" = "${OPENARM_ISAAC_COMMIT}" \
    && printf '%s  %s\n' 66a512fea6d46110e34cd7d320573a276a0a5db6cd2508e4eaadabbe39eb5938 /opt/openarm/mujoco/LICENSE | sha256sum --check --strict \
    && printf '%s  %s\n' cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30 /opt/openarm/isaac/LICENSE.txt | sha256sum --check --strict \
    && /opt/npa/sim/venv/bin/python -m pip install --no-cache-dir --no-deps --no-build-isolation -e /opt/openarm/isaac/source/openarm \
    && /opt/npa/sim/venv/bin/python -m pip uninstall -y wheel \
    && /opt/npa/sim/venv/bin/python /opt/npa/openarm/prune_python_vendor_devel.py \
    && /opt/npa/sim/venv/bin/python -m pip check \
    && cp /opt/openarm/mujoco/LICENSE /usr/share/doc/npa-openarm/OPENARM_MUJOCO_LICENSE \
    && cp /opt/openarm/isaac/LICENSE.txt /usr/share/doc/npa-openarm/OPENARM_ISAAC_LICENSE \
    && rm -rf /opt/openarm/mujoco/.git /opt/openarm/isaac/.git \
    && find /opt/npa/src /opt/openarm /usr/share/doc/npa-openarm -type d -exec chmod a+rx {} + \
    && find /opt/npa/src /opt/openarm /usr/share/doc/npa-openarm -type f -exec chmod a+r {} + \
    && rm -rf /root/.cache /home/ubuntu/.cache

RUN /opt/npa/sim/venv/bin/python -c 'import mujoco, openarm_mujoco.v2 as o; m=mujoco.MjModel.from_xml_path(o.openarm_demo_xml()); assert m.nu == 17' \
    && /opt/npa/sim/venv/bin/python -c 'import importlib.metadata as m; assert m.version("GitPython") == "3.1.62"' \
    && /opt/npa/sim/venv/bin/python -c 'import importlib.metadata as m; assert m.version("fastapi") == "0.136.1"' \
    && /opt/npa/sim/venv/bin/python -c 'import importlib.metadata as m; assert m.version("starlette") == "1.6.0"' \
    && /opt/npa/sim/venv/bin/python -c 'import importlib.util; assert importlib.util.find_spec("isaacsim") is None; assert importlib.util.find_spec("isaaclab") is None' \
    && /opt/npa/sim/venv/bin/python /opt/npa/openarm/emit_license_inventory.py

RUN /opt/npa/sim/venv/bin/python -c "import site; from pathlib import Path; Path(site.getsitepackages()[0], 'npa-exact-source.pth').write_text('/opt/npa/src\n', encoding='utf-8')" \
    && env -u PYTHONPATH /opt/npa/sim/venv/bin/python -m npa workbench openarm system-info --output-format json >/dev/null

RUN install -d -m 0755 -o ubuntu -g ubuntu /workspace/isaaclab /tmp/isaac-sim-cache /tmp/isaac-sim-cache/logs \
    && install -d -m 0700 -o ubuntu -g ubuntu /tmp/xdg-runtime \
    && chown -R ubuntu:ubuntu /opt/openarm /opt/npa/openarm

WORKDIR /workspace/isaaclab
EXPOSE 8792
USER ubuntu
ENTRYPOINT ["/usr/local/bin/npa-workflow-entrypoint"]
CMD ["npa", "workbench", "openarm", "serve", "--host", "0.0.0.0"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 CMD ["/opt/npa/sim/venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8792/health', timeout=3).read()"]
