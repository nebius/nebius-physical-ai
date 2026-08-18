# LeIsaac v0.4.0 teleoperation service. Build directly from a digest-pinned CUDA
# base and install the shared runtime-fetch closure, so an imageio-ffmpeg
# executable from historical parent layers never enters any LeIsaac layer.
# Isaac Sim and Isaac Lab are fetched only after the shared EULA preflight.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@sha256:ad6d59a3bbf3e82c1c849c9ac09cfc2a3e0bbb8655042fd899be6681b3fe2a85

ARG ISAAC_LAB_VERSION=2.3.2.post1
ARG ISAAC_SIM_VERSION=5.1.0.0
ARG ISAAC_LAB_SRC_COMMIT=37ddf626871758333d6ed89cf64ad702aef127d0
ARG NPA_RUNTIME_USER=ubuntu
ARG LEISAAC_COMMIT=1651c321e9b0c1bb54233211fc7b3cd70d8373d5
ARG LEISAAC_VERSION=0.4.0
ARG NPA_SOURCE_COMMIT=unknown

LABEL npa.tool="leisaac" \
      npa.version="${LEISAAC_VERSION}" \
      npa.source_commit="${LEISAAC_COMMIT}" \
      npa.repository_commit="${NPA_SOURCE_COMMIT}" \
      npa.isaac_sim_version="5.1.0.0" \
      npa.isaac_lab_version="2.3.2.post1" \
      npa.isaac_provisioning="runtime-fetch" \
      npa.assets_provisioning="runtime-fetch"

USER root
SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive \
    ISAAC_LAB_VERSION=${ISAAC_LAB_VERSION} \
    ISAAC_SIM_VERSION=${ISAAC_SIM_VERSION} \
    NPA_ISAAC_LAB_SRC_COMMIT=${ISAAC_LAB_SRC_COMMIT} \
    NPA_ISAAC_CACHE_DIR=/opt/isaac-cache \
    NPA_ISAAC_BASE_PYTHON=/opt/npa/sim/venv/bin/python \
    NPA_ISAAC_WHEELS_FILE=/opt/npa/docker/workbench/common/isaac-nvidia-wheels.txt \
    ISAAC_LAB_PYTHON=/isaac-sim/python.sh \
    NPA_ISAAC_LAB_OUTPUT_DIR=/workspace/isaaclab/npa-runs \
    NPA_ISAAC_LAB_WORKDIR=/workspace/isaaclab \
    OMNI_USER_DIR=/tmp/isaac-sim-cache \
    OMNI_LOG_DIR=/tmp/isaac-sim-cache/logs \
    XDG_RUNTIME_DIR=/tmp/xdg-runtime \
    NVIDIA_DRIVER_CAPABILITIES=all \
    PYTHONUNBUFFERED=1

COPY docker/workbench/common /opt/npa/docker/workbench/common
COPY docker/workbench/isaac-lab/smoke_env.py /opt/npa/docker/workbench/isaac-lab/smoke_env.py
COPY docker/workbench/isaac-lab/smoke_functional.py /opt/npa/docker/workbench/isaac-lab/smoke_functional.py

# MoviePy is not used by LeIsaac's exercised runtime; recording uses Debian's
# dynamically packaged ffmpeg. Install the shared Isaac OSS closure through an
# explicit LeIsaac variant that omits both MoviePy and the separately pinned
# imageio-ffmpeg package, so its bundled executable never enters a committed
# image layer. Match the versioned lockfile entries, not only bare names.
RUN sed -E '/^(moviepy|imageio-ffmpeg)==/d' /opt/npa/docker/workbench/common/isaac-oss-deps.txt \
      > /opt/npa/docker/workbench/common/leisaac-oss-deps.txt \
    && chmod +x /opt/npa/docker/workbench/common/*.sh \
    && NPA_RUNTIME_USER="${NPA_RUNTIME_USER}" \
       NPA_ISAAC_VENV=/opt/npa/sim/venv \
       NPA_ISAAC_OSS_DEPS_FILE=/opt/npa/docker/workbench/common/leisaac-oss-deps.txt \
       /opt/npa/docker/workbench/common/install_isaac_runtime_base.sh \
    && find /opt/npa/sim/venv/lib/python3.11/site-packages -maxdepth 1 \
      \( -name moviepy -o -name 'moviepy-*.dist-info' \
         -o -name imageio_ffmpeg -o -name 'imageio_ffmpeg-*.dist-info' \) \
      -exec rm -rf -- {} + \
    && test -z "$(find /opt/npa/sim/venv/lib/python3.11/site-packages -maxdepth 1 \
      \( -name moviepy -o -name 'moviepy-*.dist-info' \
         -o -name imageio_ffmpeg -o -name 'imageio_ffmpeg-*.dist-info' \) -print -quit)" \
    && test -z "$(find /opt/npa/sim/venv -type f -path '*/imageio_ffmpeg/binaries/ffmpeg*' -print -quit)" \
    && find /opt/npa/docker/workbench/common -type d -exec chmod 0755 {} + \
    && find /opt/npa/docker/workbench/common -type f -exec chmod 0644 {} + \
    && chmod 0755 /opt/npa/docker/workbench/common/*.sh \
    && install -d -m 0755 -o "${NPA_RUNTIME_USER}" -g "${NPA_RUNTIME_USER}" \
      /workspace/isaaclab /workspace/isaaclab/npa-runs \
      /tmp/isaac-sim-cache /tmp/isaac-sim-cache/logs \
    && install -d -m 0700 -o "${NPA_RUNTIME_USER}" -g "${NPA_RUNTIME_USER}" /tmp/xdg-runtime \
    && chown -R "${NPA_RUNTIME_USER}:${NPA_RUNTIME_USER}" /opt/npa/docker/workbench/isaac-lab

COPY docker/workbench/leisaac/upstream-observability.patch /tmp/leisaac-upstream-observability.patch
COPY docker/workbench/leisaac/upstream-packaging.patch /tmp/leisaac-upstream-packaging.patch
COPY docker/workbench/leisaac/THIRD_PARTY_NOTICES.md /opt/npa/licenses/leisaac/THIRD_PARTY_NOTICES.md

# libXrandr is required by omni.kit.livestream.webrtc. All Python packages here
# are OSS; Isaac, task assets, and NVIDIA's browser client are deliberately absent.
# The software-keyboard path does not need the unlicensed Feetech SDK used by
# physical leader hardware, so that optional dependency is not redistributed.
# Upstream declares that SDK as mandatory even though v0.4.0 lazy-loads it only
# for leader hardware. The packaging-only sed below removes that metadata edge;
# real SO101Keyboard/task source stays byte-for-byte at the pinned commit.
# The parent also installs moviepy and imageio-ffmpeg. LeIsaac records through
# the distro ffmpeg binary, so the unused wrapper and bundled static binary are
# removed below and guarded from returning.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libxrandr2 \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --filter=blob:none --no-checkout https://github.com/LightwheelAI/leisaac.git /opt/leisaac \
    && git -C /opt/leisaac checkout "${LEISAAC_COMMIT}" \
    && test "$(git -C /opt/leisaac rev-parse HEAD)" = "${LEISAAC_COMMIT}" \
    && git -C /opt/leisaac apply --recount --check --unidiff-zero /tmp/leisaac-upstream-observability.patch \
    && git -C /opt/leisaac apply --recount --unidiff-zero /tmp/leisaac-upstream-observability.patch \
    && git -C /opt/leisaac apply --check /tmp/leisaac-upstream-packaging.patch \
    && git -C /opt/leisaac apply /tmp/leisaac-upstream-packaging.patch \
    && /opt/npa/sim/venv/bin/python -m py_compile \
      /opt/leisaac/source/leisaac/leisaac/devices/keyboard/so101_keyboard.py \
      /opt/leisaac/scripts/environments/teleoperation/teleop_se3_agent.py \
    && /opt/npa/sim/venv/bin/python -c 'import ast; p="/opt/leisaac/source/leisaac/leisaac/devices/keyboard/so101_keyboard.py"; c=next(n for n in ast.parse(open(p, encoding="utf-8").read()).body if isinstance(n, ast.ClassDef) and n.name == "SO101Keyboard"); f={n.name:n for n in c.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}; m={k:ast.unparse(v) for k,v in f.items()}; assert "self._remote_pulses.clear()" in m["reset"]; assert "_remote_pulses" not in m["_add_device_control_description"]; assert ast.unparse(f["_on_keyboard_event"].body[0].test) == "self._browser_teleop" and isinstance(f["_on_keyboard_event"].body[0].body[0], ast.Return)' \
    && ! grep -q 'feetech-servo-sdk' /opt/leisaac/source/leisaac/pyproject.toml \
    && /opt/npa/sim/venv/bin/python -m pip install --no-cache-dir \
      deepdiff==8.6.1 \
      osqp==1.0.4 \
      packaging==25.0 \
      pandas==2.3.3 \
      pygame==2.6.1 \
      boto3==1.43.63 \
      pyarrow==25.0.0 \
      pyserial==3.5 \
      fastapi==0.136.3 \
      aiortc==1.15.0 \
      "uvicorn[standard]==0.46.0" \
    && /opt/npa/sim/venv/bin/python -m pip install --no-cache-dir --no-deps /opt/leisaac/source/leisaac \
    && /opt/npa/sim/venv/bin/python -m pip uninstall -y moviepy imageio-ffmpeg \
    && ! /opt/npa/sim/venv/bin/python -m pip show feetech-servo-sdk \
    && ! /opt/npa/sim/venv/bin/python -m pip show moviepy imageio-ffmpeg \
    && /opt/npa/sim/venv/bin/python -m pip check \
    && cp /opt/leisaac/LICENSE /opt/leisaac/APACHE-2.0.txt \
    && test -z "$(find /opt/npa/sim/venv -type f -path '*/imageio_ffmpeg/binaries/ffmpeg*' -print -quit)" \
    && rm -f /tmp/leisaac-upstream-observability.patch /tmp/leisaac-upstream-packaging.patch \
    && rm -rf /opt/leisaac/.git \
    && rm -rf /root/.cache /home/"${NPA_RUNTIME_USER}"/.cache \
    && test ! -e /root/.cache \
    && test ! -e /home/"${NPA_RUNTIME_USER}"/.cache \
    && test ! -e /opt/leisaac/assets/robots/so101_follower.usd \
    && test ! -e /opt/leisaac/assets/scenes/kitchen_with_orange \
    && install -d -m 0775 -o ubuntu -g isaac-sim /opt/leisaac-cache \
    && chown -R ubuntu:ubuntu /opt/leisaac

COPY --chown=ubuntu:ubuntu docker/workbench/leisaac/session_server.py /opt/npa/leisaac/session_server.py
COPY --chown=ubuntu:ubuntu docker/workbench/leisaac/smoke_functional.py /opt/npa/leisaac/smoke_functional.py
COPY --chown=ubuntu:ubuntu src/npa/agent_backend/leisaac_registry.py /opt/npa/leisaac/leisaac_registry.py
COPY --chown=ubuntu:ubuntu src/npa/agent_backend/leisaac_transport.py /opt/npa/leisaac/leisaac_transport.py
COPY --chown=ubuntu:ubuntu src/npa/agent_backend/leisaac_datachannel.py /opt/npa/leisaac/leisaac_datachannel.py
COPY --chown=ubuntu:ubuntu src/npa/agent_backend/leisaac_bundles.py /opt/npa/leisaac/leisaac_bundles.py
COPY --chown=ubuntu:ubuntu src/npa/workbench/leisaac/dataset.py /opt/npa/leisaac/leisaac_dataset.py

RUN /opt/npa/sim/venv/bin/python -m py_compile \
      /opt/npa/leisaac/session_server.py \
      /opt/npa/leisaac/smoke_functional.py \
      /opt/npa/leisaac/leisaac_registry.py \
      /opt/npa/leisaac/leisaac_transport.py \
      /opt/npa/leisaac/leisaac_datachannel.py \
      /opt/npa/leisaac/leisaac_bundles.py \
      /opt/npa/leisaac/leisaac_dataset.py \
    && test -z "$(find /opt/leisaac-cache/client -mindepth 1 -print -quit 2>/dev/null)" \
    && test -z "$(find /opt/leisaac-cache/assets/runtime -type f \
         \( -iname '*.usd' -o -iname '*.usda' -o -iname '*.usdc' \) \
         -print -quit 2>/dev/null)"

ENV NPA_LEISAAC_CACHE_DIR=/opt/leisaac-cache \
    LEISAAC_ASSETS_ROOT=/opt/leisaac-cache/assets/runtime \
    NVIDIA_DRIVER_CAPABILITIES=all \
    PATH=/usr/bin:/opt/npa/sim/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

EXPOSE 8080/tcp 49100/tcp 47998/udp
USER ubuntu
ENTRYPOINT ["/opt/npa/sim/venv/bin/python", "/opt/npa/leisaac/session_server.py"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD ["/opt/npa/sim/venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3).read()"]
