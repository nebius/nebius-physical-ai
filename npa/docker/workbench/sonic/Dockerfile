# Self-contained SONIC image. The default L40S build starts from NVIDIA's
# Isaac Lab image. The k8s/Blackwell build overrides BASE_IMAGE to the CUDA 13
# sm_120 base and installs the Isaac Sim/Lab Python layer on top.
ARG BASE_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.2@sha256:388dbc806f48359a964cb9f807feb226da95d0a107f470fdcad9780ea10fe6f2
FROM --platform=linux/amd64 ${BASE_IMAGE}

ARG BASE_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.2@sha256:388dbc806f48359a964cb9f807feb226da95d0a107f470fdcad9780ea10fe6f2
ARG ISAAC_LAB_VERSION=2.3.2.post1
ARG ISAAC_SIM_VERSION=5.1.0.0
ARG SONIC_REPO_URL=https://github.com/NVlabs/GR00T-WholeBodyControl.git
ARG SONIC_REPO_REF=0a87181c9106d0e49293400714b157676e0ec664
ARG SONIC_VERSION=0.1.2
ARG NVIDIA_DRIVER_PACKAGE_VERSION=580.126.09-1ubuntu1
ARG ISAAC_LAB_PYTHON=/isaac-sim/python.sh
ARG INSTALL_NVIDIA_DRIVER_USERSPACE=1
ARG INSTALL_ISAACSIM_EXTRA=0
ARG REQUIRE_TORCH_SM120=0
ARG NPA_DRIVER_PROVISIONING=baked
ARG NPA_CUDA_ARCHITECTURES=sm80,sm90
ARG NPA_ISAAC_LAB_INSTALL_MODE=base
ARG NPA_RUNTIME_USER=ubuntu
ARG BUILD_SONIC_DEPLOY=0

LABEL npa.tool="sonic" \
      npa.version="${SONIC_VERSION}" \
      npa.base_image="${BASE_IMAGE}" \
      npa.driver_provisioning="${NPA_DRIVER_PROVISIONING}" \
      npa.runtime_user="${NPA_RUNTIME_USER}" \
      npa.cuda_architectures="${NPA_CUDA_ARCHITECTURES}" \
      npa.isaac_lab_install="${NPA_ISAAC_LAB_INSTALL_MODE}" \
      npa.architecture="self-contained-isaac-lab"

ENV ACCEPT_EULA=Y \
    ISAACSIM_ACCEPT_EULA=YES \
    OMNI_KIT_ACCEPT_EULA=YES \
    PRIVACY_CONSENT=Y \
    ISAAC_LAB_VERSION=${ISAAC_LAB_VERSION} \
    SONIC_HOME=/opt/sonic \
    SONIC_REPO_REF=${SONIC_REPO_REF} \
    NPA_CUDA_ARCHITECTURES=${NPA_CUDA_ARCHITECTURES} \
    NVIDIA_DRIVER_CAPABILITIES=all \
    VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
    __GLX_VENDOR_LIBRARY_NAME=nvidia \
    OMNI_USER_DIR=/tmp/isaac-sim-cache \
    OMNI_LOG_DIR=/tmp/isaac-sim-cache/logs \
    XDG_RUNTIME_DIR=/tmp/xdg-runtime \
    PYTHONUNBUFFERED=1 \
    PIP_EXTRA_INDEX_URL=https://pypi.nvidia.com

ENV ISAAC_LAB_PYTHON=${ISAAC_LAB_PYTHON} \
    REQUIRE_TORCH_SM120=${REQUIRE_TORCH_SM120}

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
      libegl1 \
      libgtest-dev \
      libgl1 \
      libglvnd0 \
      libglu1-mesa \
      nlohmann-json3-dev \
      pkg-config \
      python3 \
      python3-pip \
      wget \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system

RUN if [ "${INSTALL_NVIDIA_DRIVER_USERSPACE}" = "1" ]; then \
      wget -qO /tmp/cuda-keyring.deb \
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
        libnvidia-gpucomp-580; \
    else \
      DEBIAN_FRONTEND=noninteractive apt-get update \
      && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends vulkan-tools; \
    fi \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/npa/docker/workbench/sonic /tmp/npa-sonic-output
COPY docker/workbench/sonic/requirements.txt /opt/npa/docker/workbench/sonic/requirements.txt

RUN "${ISAAC_LAB_PYTHON}" - <<'PY' > /tmp/npa-torch-constraints.txt
from importlib import metadata

for dist in metadata.distributions():
    name = dist.metadata.get("Name", "")
    canonical = name.lower().replace("_", "-")
    if canonical in {"torch", "torchvision", "torchaudio", "triton"} or canonical.startswith("nvidia-"):
        print(f"{canonical}==={dist.version}")
PY

RUN if [ "${INSTALL_ISAACSIM_EXTRA}" = "1" ]; then \
      "${ISAAC_LAB_PYTHON}" -m pip install --no-cache-dir \
        -c /tmp/npa-torch-constraints.txt \
        "isaaclab[all]==${ISAAC_LAB_VERSION}" \
        --extra-index-url https://pypi.nvidia.com \
      && "${ISAAC_LAB_PYTHON}" -m pip install --no-cache-dir --no-deps --ignore-installed \
        "isaacsim==${ISAAC_SIM_VERSION}" \
        "isaacsim-kernel==${ISAAC_SIM_VERSION}" \
        "isaacsim-extscache-kit==${ISAAC_SIM_VERSION}" \
        "isaacsim-extscache-kit-sdk==${ISAAC_SIM_VERSION}" \
        "isaacsim-extscache-physics==${ISAAC_SIM_VERSION}" \
        "isaacsim-app==${ISAAC_SIM_VERSION}" \
        "isaacsim-asset==${ISAAC_SIM_VERSION}" \
        "isaacsim-benchmark==${ISAAC_SIM_VERSION}" \
        "isaacsim-code-editor==${ISAAC_SIM_VERSION}" \
        "isaacsim-core==${ISAAC_SIM_VERSION}" \
        "isaacsim-cortex==${ISAAC_SIM_VERSION}" \
        "isaacsim-example==${ISAAC_SIM_VERSION}" \
        "isaacsim-gui==${ISAAC_SIM_VERSION}" \
        "isaacsim-replicator==${ISAAC_SIM_VERSION}" \
        "isaacsim-rl==${ISAAC_SIM_VERSION}" \
        "isaacsim-robot==${ISAAC_SIM_VERSION}" \
        "isaacsim-robot-motion==${ISAAC_SIM_VERSION}" \
        "isaacsim-robot-setup==${ISAAC_SIM_VERSION}" \
        "isaacsim-ros1==${ISAAC_SIM_VERSION}" \
        "isaacsim-ros2==${ISAAC_SIM_VERSION}" \
        "isaacsim-sensor==${ISAAC_SIM_VERSION}" \
        "isaacsim-storage==${ISAAC_SIM_VERSION}" \
        "isaacsim-template==${ISAAC_SIM_VERSION}" \
        "isaacsim-test==${ISAAC_SIM_VERSION}" \
        "isaacsim-utils==${ISAAC_SIM_VERSION}" \
        --extra-index-url https://pypi.nvidia.com; \
    else \
      "${ISAAC_LAB_PYTHON}" -m pip install --no-cache-dir --no-deps --ignore-installed \
        "isaaclab==${ISAAC_LAB_VERSION}" \
        --extra-index-url https://pypi.nvidia.com; \
    fi

RUN "${ISAAC_LAB_PYTHON}" -m pip install --no-cache-dir \
      -r /opt/npa/docker/workbench/sonic/requirements.txt

RUN python3 -m pip install --break-system-packages --no-cache-dir uv \
    && curl -fsSL https://just.systems/install.sh | bash -s -- --to /usr/local/bin

RUN git clone "${SONIC_REPO_URL}" "${SONIC_HOME}" \
    && cd "${SONIC_HOME}" \
    && git checkout "${SONIC_REPO_REF}" \
    && (git lfs pull --include="download_from_hf.py,gear_sonic/**,gear_sonic_deploy/**,decoupled_wbc/**" --exclude="motionbricks/out/**" || true) \
    && rm -rf "${SONIC_HOME}/.git" "${SONIC_HOME}/docs"

RUN "${ISAAC_LAB_PYTHON}" - <<'PY' > /tmp/npa-torch-constraints.txt
from importlib import metadata

for dist in metadata.distributions():
    name = dist.metadata.get("Name", "")
    canonical = name.lower().replace("_", "-")
    if canonical in {"torch", "torchvision", "torchaudio", "triton"} or canonical.startswith("nvidia-"):
        print(f"{canonical}==={dist.version}")
PY

RUN "${ISAAC_LAB_PYTHON}" -m pip install --no-cache-dir \
      -c /tmp/npa-torch-constraints.txt \
      -e "${SONIC_HOME}/gear_sonic[training,teleop,sim]" \
    && "${ISAAC_LAB_PYTHON}" - <<'PY'
import pathlib
import site

for site_dir in site.getsitepackages():
    path = pathlib.Path(site_dir) / "sonic.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("from gear_sonic import *\n", encoding="utf-8")
PY

RUN "${ISAAC_LAB_PYTHON}" - <<'PY'
import importlib
import os
from importlib import metadata

expected = "2.3.2.post1"
actual = metadata.version("isaaclab")
if actual != expected:
    raise RuntimeError(f"expected isaaclab {expected}, found {actual}")
for module in ("gear_sonic", "sonic", "isaaclab", "isaaclab.app"):
    importlib.import_module(module)
if os.environ.get("REQUIRE_TORCH_SM120") == "1":
    import torch

    arch_flags = getattr(torch._C, "_cuda_getArchFlags", lambda: "")()
    arches = set(arch_flags.split()) or set(torch.cuda.get_arch_list())
    if "sm_120" not in arches:
        raise RuntimeError(f"expected PyTorch with sm_120 kernels, found {sorted(arches)}")
    print(f"NPA_SONIC_TORCH_SM120_OK torch={torch.__version__} cuda={torch.version.cuda}")
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
    && (getent group isaac-sim >/dev/null 2>&1 || groupadd -r isaac-sim) \
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
USER ${NPA_RUNTIME_USER}
ENTRYPOINT ["/entrypoint.sh"]
CMD ["smoke"]
