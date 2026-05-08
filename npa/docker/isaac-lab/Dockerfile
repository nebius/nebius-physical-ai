FROM nvcr.io/nvidia/isaac-lab:2.3.2

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

RUN mkdir -p /opt/npa/docker/isaac-lab /workspace/isaaclab/npa-runs

COPY docker/isaac-lab/smoke_env.py /opt/npa/docker/isaac-lab/smoke_env.py
COPY docker/isaac-lab/smoke_functional.py /opt/npa/docker/isaac-lab/smoke_functional.py

RUN "${ISAAC_LAB_PYTHON}" -m pip install --no-cache-dir --no-deps --ignore-installed \
      "isaaclab==${ISAAC_LAB_VERSION}" \
      --extra-index-url https://pypi.nvidia.com

RUN "${ISAAC_LAB_PYTHON}" - <<'PY' \
    && "${ISAAC_LAB_PYTHON}" -m py_compile \
      /opt/npa/docker/isaac-lab/smoke_env.py \
      /opt/npa/docker/isaac-lab/smoke_functional.py
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

WORKDIR /workspace/isaaclab
ENTRYPOINT ["/bin/bash"]
