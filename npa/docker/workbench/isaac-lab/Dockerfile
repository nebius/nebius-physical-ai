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
# because SkyPilot's runtime bootstrap needs a system python3 (plus rsync) inside the
# container, and the NVIDIA base ships none — Isaac's interpreter is /isaac-sim/python.sh,
# and /isaac-sim itself is mode 750 isaac-sim:isaac-sim, i.e. unreadable by the `ubuntu`
# user this image runs as. Diagnosed on npa-rtxpro-mk8s with a raw pod probe:
#   SHELL_OK / whoami=ubuntu / "/bin/bash: line 1: python3: command not found"
#
# Two small layers fix it without touching Isaac's own toolchain:
#   1. a system python3 (+ rsync/curl/ssh client) for SkyPilot's runtime;
#   2. group membership instead of a recursive chown/chmod, so `ubuntu` gains the
#      r-x the isaac-sim group already has on /isaac-sim (a `chmod -R` here would
#      rewrite multi-GB layers).
# `npa workbench isaac-lab train` and npa.workflows.rl_sweep still run training through
# ISAAC_LAB_PYTHON (/isaac-sim/python.sh); this only makes the container schedulable.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         python3 python3-venv python3-pip rsync curl openssh-client sudo netcat-openbsd \
    && rm -rf /var/lib/apt/lists/* \
    && (getent group isaac-sim >/dev/null 2>&1 && usermod -aG isaac-sim ubuntu || true) \
    && python3 --version

WORKDIR /workspace/isaaclab
USER ubuntu
ENTRYPOINT ["/bin/bash"]
