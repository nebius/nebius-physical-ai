# Digest pinned manually 2026-05-12 using operator's NGC API key.
# Tag: nvcr.io/nvidia/isaac-lab:2.3.2
# Multi-arch manifest list digest.
FROM nvcr.io/nvidia/isaac-lab:2.3.2@sha256:388dbc806f48359a964cb9f807feb226da95d0a107f470fdcad9780ea10fe6f2

ARG ISAAC_LAB_VERSION=2.3.2.post1
ARG ISAAC_SIM_VERSION=5.1.0.0

LABEL npa.tool="isaac-lab" \
      npa.version="${ISAAC_LAB_VERSION}" \
      npa.isaac_sim_version="${ISAAC_SIM_VERSION}"

ENV ACCEPT_EULA=Y \
    ISAACSIM_ACCEPT_EULA=YES \
    OMNI_KIT_ACCEPT_EULA=YES \
    PRIVACY_CONSENT=Y \
    ISAAC_LAB_VERSION=${ISAAC_LAB_VERSION} \
    ISAAC_SIM_VERSION=${ISAAC_SIM_VERSION} \
    ISAAC_LAB_PYTHON=/isaac-sim/python.sh \
    NPA_ISAAC_LAB_OUTPUT_DIR=/workspace/isaaclab/npa-runs \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-lc"]

USER root

RUN mkdir -p /opt/npa/docker/workbench/isaac-lab /workspace/isaaclab/npa-runs

COPY docker/workbench/isaac-lab/smoke_env.py /opt/npa/docker/workbench/isaac-lab/smoke_env.py
COPY docker/workbench/isaac-lab/smoke_functional.py /opt/npa/docker/workbench/isaac-lab/smoke_functional.py

RUN "${ISAAC_LAB_PYTHON}" -m pip install --no-cache-dir --no-deps --ignore-installed \
      "isaaclab==${ISAAC_LAB_VERSION}" \
      --extra-index-url https://pypi.nvidia.com

RUN "${ISAAC_LAB_PYTHON}" - <<'PY' \
    && "${ISAAC_LAB_PYTHON}" -m py_compile \
      /opt/npa/docker/workbench/isaac-lab/smoke_env.py \
      /opt/npa/docker/workbench/isaac-lab/smoke_functional.py
import os
from importlib import metadata
from pathlib import Path

expected_lab = os.environ["ISAAC_LAB_VERSION"]
actual_lab = metadata.version("isaaclab")
if actual_lab != expected_lab:
    raise RuntimeError(f"expected isaaclab {expected_lab}, found {actual_lab}")
if not Path("/isaac-sim/python.sh").is_file():
    raise RuntimeError("official Isaac Sim Python interpreter not found at /isaac-sim/python.sh")
print(f"ISAAC_LAB_BUILD_VERSION_OK isaaclab={actual_lab} isaac_sim_python=/isaac-sim/python.sh")
PY

RUN id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu \
    && chown -R ubuntu:ubuntu /opt/npa/docker/workbench/isaac-lab /workspace/isaaclab/npa-runs

# --- Schedulable-on-Kubernetes prerequisites -----------------------------------
#
# Without these the image cannot host a SkyPilot task at all: provisioning dies with
#   KubernetesError: Failed to get ssh user for pod ...: container not found ("ray-node")
# because the container exits during SkyPilot's runtime bootstrap. Verified on
# npa-rtxpro-mk8s by bisecting a derived image one ingredient at a time; ALL FOUR are
# required, and the last two are the ones that are easy to miss:
#
#   1. a system python3 (+ rsync/curl/ssh client) - the NVIDIA base ships none, only
#      /isaac-sim/python.sh;
#   2. /isaac-sim readable by the runtime user - it is mode 750 isaac-sim:isaac-sim, so
#      the user joins that GROUP (a recursive chown/chmod would rewrite multi-GB layers);
#   3. passwordless sudo for the runtime user - SkyPilot's in-pod setup shells out to
#      sudo, and Debian's default %sudo rule prompts for a password (this alone kept a
#      non-root image failing while an otherwise identical root image worked);
#   4. the system interpreter FIRST on PATH - otherwise python3 resolves to Isaac's kit
#      interpreter, which cannot import its own site-packages outside python.sh.
#
# Training is unaffected: `npa workbench isaac-lab train` and npa.workflows.rl_sweep
# still run through ISAAC_LAB_PYTHON (/isaac-sim/python.sh). The image stays non-root.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         python3 python3-venv python3-pip rsync curl openssh-client sudo netcat-openbsd \
    && rm -rf /var/lib/apt/lists/* \
    && (getent group isaac-sim >/dev/null 2>&1 && usermod -aG isaac-sim ubuntu || true) \
    && printf 'ubuntu ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/99-npa-runtime-user \
    && chmod 0440 /etc/sudoers.d/99-npa-runtime-user \
    && python3 --version

# SkyPilot's bootstrap must find the SYSTEM interpreter, not Isaac's kit python (which
# the base image puts earlier on PATH and which cannot import its own site-packages
# outside /isaac-sim/python.sh).
ENV PATH=/usr/bin:$PATH

WORKDIR /workspace/isaaclab
USER ubuntu
ENTRYPOINT ["/bin/bash"]
