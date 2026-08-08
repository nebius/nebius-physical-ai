# LeIsaac v0.4.0 teleoperation service. The parent contains no Isaac/Kit bytes:
# Isaac Sim and Isaac Lab are fetched at runtime after operator EULA acceptance.
FROM ghcr.io/nebius/nebius-physical-ai/npa-isaac-lab:2.3.2.post1@sha256:c2e142c5312c903fd2a165028ca17e34335813d034ce036b936c6dc3a886bea8

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

COPY docker/workbench/leisaac/upstream-observability.patch /tmp/leisaac-upstream-observability.patch

# libXrandr is required by omni.kit.livestream.webrtc. All Python packages here
# are OSS; Isaac, task assets, and NVIDIA's browser client are deliberately absent.
# The software-keyboard path does not need the unlicensed Feetech SDK used by
# physical leader hardware, so that optional dependency is not redistributed.
# Upstream declares that SDK as mandatory even though v0.4.0 lazy-loads it only
# for leader hardware. The packaging-only sed below removes that metadata edge;
# real SO101Keyboard/task source stays byte-for-byte at the pinned commit.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libxrandr2 \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --filter=blob:none --no-checkout https://github.com/LightwheelAI/leisaac.git /opt/leisaac \
    && git -C /opt/leisaac checkout "${LEISAAC_COMMIT}" \
    && test "$(git -C /opt/leisaac rev-parse HEAD)" = "${LEISAAC_COMMIT}" \
    && git -C /opt/leisaac apply --check --unidiff-zero /tmp/leisaac-upstream-observability.patch \
    && git -C /opt/leisaac apply --unidiff-zero /tmp/leisaac-upstream-observability.patch \
    && sed -i '/"feetech-servo-sdk",/d' /opt/leisaac/source/leisaac/pyproject.toml \
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
    && ! /opt/npa/sim/venv/bin/python -m pip show feetech-servo-sdk \
    && /opt/npa/sim/venv/bin/python -m pip check \
    && cp /opt/leisaac/LICENSE /opt/leisaac/APACHE-2.0.txt \
    && rm -f /tmp/leisaac-upstream-observability.patch \
    && rm -rf /opt/leisaac/.git \
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
