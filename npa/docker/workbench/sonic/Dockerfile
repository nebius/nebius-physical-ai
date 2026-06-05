# Self-contained SONIC image. It uses NVIDIA's Isaac Lab distribution directly,
# not the Workbench Isaac Lab image, then installs SONIC as an application layer.
FROM --platform=linux/amd64 nvcr.io/nvidia/isaac-lab:2.3.2@sha256:388dbc806f48359a964cb9f807feb226da95d0a107f470fdcad9780ea10fe6f2

ARG ISAAC_LAB_VERSION=2.3.2.post1
ARG SONIC_REPO_URL=https://github.com/NVlabs/GR00T-WholeBodyControl.git
ARG SONIC_REPO_REF=0a87181c9106d0e49293400714b157676e0ec664
ARG SONIC_VERSION=0.1.2
ARG NVIDIA_DRIVER_PACKAGE_VERSION=580.126.09-1ubuntu1
ARG BUILD_SONIC_DEPLOY=0

LABEL npa.tool="sonic" \
      npa.version="${SONIC_VERSION}" \
      npa.architecture="self-contained-isaac-lab"

ENV ACCEPT_EULA=Y \
    ISAACSIM_ACCEPT_EULA=YES \
    OMNI_KIT_ACCEPT_EULA=YES \
    PRIVACY_CONSENT=Y \
    ISAAC_LAB_VERSION=${ISAAC_LAB_VERSION} \
    ISAAC_LAB_PYTHON=/isaac-sim/python.sh \
    SONIC_HOME=/opt/sonic \
    SONIC_REPO_REF=${SONIC_REPO_REF} \
    NVIDIA_DRIVER_CAPABILITIES=all \
    VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
    __GLX_VENDOR_LIBRARY_NAME=nvidia \
    OMNI_USER_DIR=/tmp/isaac-sim-cache \
    OMNI_LOG_DIR=/tmp/isaac-sim-cache/logs \
    XDG_RUNTIME_DIR=/tmp/xdg-runtime \
    PYTHONUNBUFFERED=1 \
    PIP_EXTRA_INDEX_URL=https://pypi.nvidia.com

SHELL ["/bin/bash", "-lc"]
USER root

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      clang \
      cmake \
      curl \
      git \
      git-lfs \
      libgtest-dev \
      nlohmann-json3-dev \
      pkg-config \
      python3 \
      python3-pip \
      wget \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system

RUN wget -qO /tmp/cuda-keyring.deb \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && rm /tmp/cuda-keyring.deb \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      "libnvidia-cfg1-580=${NVIDIA_DRIVER_PACKAGE_VERSION}" \
      "libnvidia-common-580=${NVIDIA_DRIVER_PACKAGE_VERSION}" \
      "libnvidia-compute-580=${NVIDIA_DRIVER_PACKAGE_VERSION}" \
      "libnvidia-decode-580=${NVIDIA_DRIVER_PACKAGE_VERSION}" \
      "libnvidia-gl-580=${NVIDIA_DRIVER_PACKAGE_VERSION}" \
      "libnvidia-gpucomp-580=${NVIDIA_DRIVER_PACKAGE_VERSION}" \
      vulkan-tools \
    && apt-mark hold \
      libnvidia-cfg1-580 \
      libnvidia-common-580 \
      libnvidia-compute-580 \
      libnvidia-decode-580 \
      libnvidia-gl-580 \
      libnvidia-gpucomp-580 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/npa/docker/workbench/sonic /tmp/npa-sonic-output
COPY docker/workbench/sonic/requirements.txt /opt/npa/docker/workbench/sonic/requirements.txt

RUN "${ISAAC_LAB_PYTHON}" -m pip install --no-cache-dir --no-deps --ignore-installed \
      "isaaclab==${ISAAC_LAB_VERSION}" \
      --extra-index-url https://pypi.nvidia.com

RUN "${ISAAC_LAB_PYTHON}" -m pip install --no-cache-dir \
      -r /opt/npa/docker/workbench/sonic/requirements.txt

RUN python3 -m pip install --break-system-packages --no-cache-dir uv \
    && curl -fsSL https://just.systems/install.sh | bash -s -- --to /usr/local/bin

RUN git clone "${SONIC_REPO_URL}" "${SONIC_HOME}" \
    && cd "${SONIC_HOME}" \
    && git checkout "${SONIC_REPO_REF}" \
    && (git lfs pull --include="download_from_hf.py,gear_sonic/**,gear_sonic_deploy/**,decoupled_wbc/**" --exclude="motionbricks/out/**" || true) \
    && rm -rf "${SONIC_HOME}/.git" "${SONIC_HOME}/docs"

RUN "${ISAAC_LAB_PYTHON}" -m pip install --no-cache-dir \
      -e "${SONIC_HOME}/gear_sonic[training,teleop,sim]" \
    && "${ISAAC_LAB_PYTHON}" - <<'PY'
import pathlib
import site

for site_dir in site.getsitepackages():
    path = pathlib.Path(site_dir) / "sonic.py"
    path.write_text("from gear_sonic import *\n", encoding="utf-8")
PY

RUN "${ISAAC_LAB_PYTHON}" - <<'PY'
import importlib
from importlib import metadata

expected = "2.3.2.post1"
actual = metadata.version("isaaclab")
if actual != expected:
    raise RuntimeError(f"expected isaaclab {expected}, found {actual}")
for module in ("gear_sonic", "sonic", "isaaclab"):
    importlib.import_module(module)
print("NPA_SONIC_BUILD_IMPORTS_OK")
PY

# The C++ deploy stack is included for VM/container deployment paths, but
# TensorRT and ONNX Runtime discovery are environment-sensitive. Keep the default
# train/smoke image build focused; deploy images can opt into this compile step.
RUN if [ "${BUILD_SONIC_DEPLOY}" = "1" ]; then \
      cd "${SONIC_HOME}/gear_sonic_deploy" \
      && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      && cmake --build build -j"$(nproc)"; \
    else \
      echo "Skipping optional SONIC C++ deploy build (BUILD_SONIC_DEPLOY=0)"; \
    fi

COPY docker/workbench/sonic/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu \
    && usermod -aG isaac-sim ubuntu \
    && install -d -m 775 -o root -g isaac-sim \
      /isaac-sim/kit/logs \
      /isaac-sim/kit/cache \
      /isaac-sim/kit/data \
      /tmp/isaac-sim-cache \
      /tmp/isaac-sim-cache/logs \
      /tmp/npa-sonic-output \
    && install -d -m 700 -o ubuntu -g ubuntu /tmp/xdg-runtime \
    && chown ubuntu:ubuntu /opt/sonic \
    && chown -R ubuntu:ubuntu /tmp/npa-sonic-output \
    && chgrp -R isaac-sim \
      /isaac-sim/kit/logs \
      /isaac-sim/kit/cache \
      /isaac-sim/kit/data \
      /tmp/isaac-sim-cache \
    && chmod -R g+rwX \
      /isaac-sim/kit/logs \
      /isaac-sim/kit/cache \
      /isaac-sim/kit/data \
      /tmp/isaac-sim-cache

WORKDIR /opt/sonic
USER ubuntu
ENTRYPOINT ["/entrypoint.sh"]
CMD ["smoke"]
