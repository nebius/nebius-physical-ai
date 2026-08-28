# npa-groot: NVIDIA Isaac-GR00T inference and fine-tuning, with NVIDIA Isaac
# fetched at FIRST RUN.
#
# This image contains no NVIDIA Isaac Sim or Isaac Lab code. It used to bake
# `isaaclab[isaacsim,all]` into /opt/isaac-lab/venv (then delete extscache to claw back
# some size), which put Omniverse Kit in the layers and made the image
# non-redistributable. Isaac is now downloaded from https://pypi.nvidia.com on first use
# of /isaac-sim/python.sh, under the OPERATOR's own EULA acceptance; see
# npa/docker/workbench/common/isaac_bootstrap.sh.
#
# GR00T inference and fine-tuning need no Isaac or EULA acceptance: they run in
# GROOT_VENV (`python -m npa.smoke.test_groot_functional` covers inference). Only
# the Isaac Lab simulation paths pay the first-run download.
#
# Digest pinned 2026-05-12. Tag: nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
# To update: re-resolve via registry API and update digest below.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@sha256:ad6d59a3bbf3e82c1c849c9ac09cfc2a3e0bbb8655042fd899be6681b3fe2a85

ARG GROOT_RUNTIME_VERSION=0.1.0
ARG GROOT_REPO_REF=3df8b3825d67f755e69141446f4315f281b9b7e6
ARG ISAAC_LAB_VERSION=2.3.2.post1
ARG ISAAC_SIM_VERSION=5.1.0.0
ARG ISAAC_LAB_SRC_COMMIT=37ddf626871758333d6ed89cf64ad702aef127d0
ARG GROOT_UBUNTU_SNAPSHOT=20260827T000000Z
ARG GROOT_LINUX_LIBC_DEV_VERSION=5.15.0-190.200
ARG COSMOS_REASON_MODEL=nvidia/Cosmos-Reason2-2B
ARG COSMOS_REASON_REVISION=9ce19a195e423419c349abfc86fd07178b230561

LABEL npa.tool="groot" \
      npa.version="${GROOT_RUNTIME_VERSION}" \
      npa.groot_ref="${GROOT_REPO_REF}" \
      npa.isaac_lab_version="${ISAAC_LAB_VERSION}" \
      npa.isaac_provisioning="runtime-fetch" \
      npa.architecture="no-baked-omniverse-kit" \
      npa.cosmos_reason_revision="${COSMOS_REASON_REVISION}" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

# NOTE the absence of ACCEPT_EULA / ISAACSIM_ACCEPT_EULA / OMNI_KIT_ACCEPT_EULA /
# PRIVACY_CONSENT. All four used to be baked here. Acceptance is now the operator's to
# give at run time, which is the whole legal mechanism, and
# npa/tests/docker/test_packaging_contract.py fails the build if any of them reappears.
ENV DEBIAN_FRONTEND=noninteractive \
    ISAAC_SIM_VERSION=${ISAAC_SIM_VERSION} \
    NPA_ISAAC_LAB_SRC_COMMIT=${ISAAC_LAB_SRC_COMMIT} \
    NPA_ISAAC_CACHE_DIR=/opt/isaac-cache \
    NPA_ISAAC_BASE_PYTHON=/opt/isaac-lab/venv/bin/python \
    NPA_IMAGE_PYTHON=/opt/isaac-lab/venv/bin/python \
    NPA_ISAAC_WHEELS_FILE=/opt/npa/docker/workbench/common/isaac-nvidia-wheels.txt \
    ISAAC_LAB_PYTHON=/isaac-sim/python.sh \
    GROOT_RUNTIME_VERSION=${GROOT_RUNTIME_VERSION} \
    GROOT_REPO_REF=${GROOT_REPO_REF} \
    ISAAC_LAB_VERSION=${ISAAC_LAB_VERSION} \
    COSMOS_REASON_MODEL=${COSMOS_REASON_MODEL} \
    COSMOS_REASON_REVISION=${COSMOS_REASON_REVISION} \
    GROOT_HOME=/opt/groot \
    GROOT_DATA_MOUNT=/opt/groot-data \
    GROOT_REPO=/opt/groot/Isaac-GR00T \
    GROOT_VENV=/opt/groot/Isaac-GR00T/.venv \
    GROOT_MODEL_DIR=/opt/groot-data/models \
    GROOT_OUTPUT_DIR=/opt/groot-data/outputs \
    HF_HOME=/opt/groot-data/hf_cache \
    HUGGINGFACE_HUB_CACHE=/opt/groot-data/hf_cache \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      ffmpeg \
      git \
      git-lfs \
      gnupg \
      libegl1 \
      libffi-dev \
      libgl1 \
      libglib2.0-0 \
      libglvnd0 \
      libglu1-mesa \
      libsm6 \
      libssl-dev \
      libx11-6 \
      libxext6 \
      libxrender1 \
      ninja-build \
      pkg-config \
      python3.10 \
      python3.10-dev \
      python3.10-venv \
      python3-pip \
      unzip \
      wget \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
      python3.11 \
      python3.11-dev \
      python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash -u 1000 ubuntu \
    && install -d -m 0755 -o ubuntu -g ubuntu \
      /opt/groot \
      /opt/groot-data/models \
      /opt/groot-data/hf_cache \
      /opt/groot-data/outputs \
      /opt/groot-data/checkpoints \
      /opt/groot-data/data_cache \
      /opt/groot-data/checkpoint_cache \
      /opt/groot-data/base_model_cache \
      /opt/groot-data/eval_data_cache \
      /opt/groot-data/config_cache \
      /opt/isaac-lab \
      /opt/npa/src/npa \
      /opt/npa/docker/workbench/groot

USER ubuntu
WORKDIR /opt/groot

ENV PATH=/home/ubuntu/.local/bin:/opt/groot/Isaac-GR00T/.venv/bin:$PATH \
    PYTHONPATH=/opt/npa/src \
    NPA_SKIP_EAGER_IMPORTS=1 \
    NPA_LIGHT_WORKBENCH_TOOL=groot

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && git lfs install \
    && git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git "${GROOT_REPO}" \
    && git -C "${GROOT_REPO}" checkout "${GROOT_REPO_REF}" \
    && test "$(git -C "${GROOT_REPO}" rev-parse HEAD)" = "${GROOT_REPO_REF}" \
    && git -C "${GROOT_REPO}" submodule update --init --recursive

RUN python3.10 - <<'PY'
from pathlib import Path
import os

repo = Path(os.environ["GROOT_REPO"])
cosmos_model = os.environ["COSMOS_REASON_MODEL"]
cosmos_revision = os.environ["COSMOS_REASON_REVISION"]


def replace_once(rel: str, old: str, new: str) -> None:
    path = repo / rel
    text = path.read_text()
    if old in text:
        path.write_text(text.replace(old, new, 1))
        return
    if new in text:
        return
    raise RuntimeError("Could not apply GR00T runtime pin patch to " + rel)


replace_once(
    "gr00t/configs/model/gr00t_n1d7.py",
    "    model_revision: str | None = None\n",
    f"    model_revision: str | None = \"{cosmos_revision}\"\n",
)
replace_once(
    "gr00t/experiment/launch_finetune.py",
    f"    config.model.model_name = \"{cosmos_model}\"\n",
    f"    config.model.model_name = \"{cosmos_model}\"\n"
    f"    config.model.model_revision = \"{cosmos_revision}\"\n",
)
replace_once(
    "gr00t/configs/finetune_config.py",
    "    tune_diffusion_model: bool = True\n"
    "    \"\"\"If True, fine-tune the diffusion-based action decoder (if present in the model).\"\"\"\n\n",
    "    tune_diffusion_model: bool = True\n"
    "    \"\"\"If True, fine-tune the diffusion-based action decoder (if present in the model).\"\"\"\n\n"
    "    use_relative_action: bool = True\n"
    "    \"\"\"Convert action groups declared RELATIVE when training and inference.\"\"\"\n\n",
)
replace_once(
    "gr00t/experiment/launch_finetune.py",
    "    config.model.use_relative_action = True\n",
    "    config.model.use_relative_action = ft_config.use_relative_action\n",
)
replace_once(
    "gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
    "        super().__init__(config)\n"
    "        self.config = config\n\n"
    "        backbone_cls = get_backbone_cls(config)\n",
    "        super().__init__(config)\n"
    "        self.config = config\n"
    f"        if (\n"
    f"            getattr(config, \"model_name\", \"\") == \"{cosmos_model}\"\n"
    f"            and not getattr(config, \"model_revision\", None)\n"
    f"        ):\n"
    f"            config.model_revision = \"{cosmos_revision}\"\n"
    "        if getattr(config, \"model_revision\", None) and \"revision\" not in transformers_loading_kwargs:\n"
    "            transformers_loading_kwargs = {\n"
    "                **transformers_loading_kwargs,\n"
    "                \"revision\": config.model_revision,\n"
    "            }\n\n"
    "        backbone_cls = get_backbone_cls(config)\n",
)
replace_once(
    "gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py",
    f"        model_name: str = \"{cosmos_model}\",\n"
    "        model_type: str = \"qwen\",\n",
    f"        model_name: str = \"{cosmos_model}\",\n"
    f"        model_revision: str | None = \"{cosmos_revision}\",\n"
    "        model_type: str = \"qwen\",\n",
)
replace_once(
    "gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py",
    "        self.model_name = model_name\n"
    "        self.model_type = model_type\n\n",
    "        self.model_name = model_name\n"
    "        self.model_revision = model_revision\n"
    "        self.model_type = model_type\n"
    "        if model_revision and \"revision\" not in transformers_loading_kwargs:\n"
    "            transformers_loading_kwargs = {\n"
    "                **transformers_loading_kwargs,\n"
    "                \"revision\": model_revision,\n"
    "            }\n\n",
)
replace_once(
    "gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py",
    f"        processor_kwargs.setdefault(\"model_name\", \"{cosmos_model}\")\n",
    f"        processor_kwargs.setdefault(\"model_name\", \"{cosmos_model}\")\n"
    f"        processor_kwargs.setdefault(\"model_revision\", \"{cosmos_revision}\")\n",
)
print("GROOT_RUNTIME_PIN_PATCH_OK " + cosmos_revision)
PY

RUN cd "${GROOT_REPO}" \
    && uv sync --python 3.10 \
    && uv pip install --python "${GROOT_VENV}/bin/python" \
      boto3==1.42.91 \
      fastapi==0.136.1 \
      "uvicorn[standard]==0.38.0" \
      typer==0.24.1 \
      httpx==0.28.1 \
      paramiko==5.0.0 \
      pyyaml==6.0.3 \
      rich==15.0.0 \
      jinja2==3.1.6 \
      "mcap>=1.3,<2" \
    && uv pip install --python "${GROOT_VENV}/bin/python" --no-deps --upgrade \
      "GitPython>=3.1.50" \
      "PyJWT>=2.13.0" \
      "aiohttp>=3.13.3" \
      "aiomysql>=0.3.0" \
      "azure-core>=1.38.0" \
      "cryptography>=48.0.1" \
      "diffusers>=0.38.0" \
      "jaraco.context>=6.1.0" \
      "msgpack>=1.2.1" \
      "onnx>=1.21.0" \
      "pillow>=12.2.0" \
      "python-multipart>=0.0.30" \
      "starlette>=1.3.1" \
      "tornado>=6.5.6" \
      "transformers>=5.3.0" \
      "urllib3>=2.7.0" \
      "wheel>=0.46.2" \
    && (uv pip uninstall --python "${GROOT_VENV}/bin/python" -y wandb lxml || true)

# The Isaac side of the image: python3.11 + PyTorch + the OSS dependency closure Isaac
# needs, the runtime-fetch bootstrap, and the /isaac-sim/python.sh shim. What used to be
# here instead was `pip install "isaaclab[isaacsim,all]"` - i.e. Omniverse Kit, in the
# layers. The venv path is unchanged so the e2e shell harnesses keep working.
USER root
COPY docker/workbench/common /opt/npa/docker/workbench/common
RUN chmod +x /opt/npa/docker/workbench/common/*.sh \
    && NPA_RUNTIME_USER=ubuntu \
       NPA_ISAAC_VENV=/opt/isaac-lab/venv \
       NPA_ISAAC_SKIP_TORCH=0 \
       NPA_UBUNTU_SNAPSHOT=${GROOT_UBUNTU_SNAPSHOT} \
       NPA_LINUX_LIBC_DEV_VERSION=${GROOT_LINUX_LIBC_DEV_VERSION} \
       REQUIRE_TORCH_SM120=1 \
       NPA_INSTALL_SKYPILOT_PREREQS=1 \
       /opt/npa/docker/workbench/common/install_isaac_runtime_base.sh \
    `# Assert the exact packages promised by the image bootstrap contract.` \
    && dpkg-query -W openssh-server rsync sudo \
    && grep -Fx 'ubuntu ALL=(ALL) NOPASSWD:ALL' /etc/sudoers.d/99-npa-runtime-user \
    `# Generate unique SSH host keys only when SkyPilot starts the pod service.` \
    && sed -i '/^  start)$/a\    ssh-keygen -A' /etc/init.d/ssh \
    && grep -q 'ssh-keygen -A' /etc/init.d/ssh \
    && rm -f /etc/ssh/ssh_host_* \
    && chown -R ubuntu:ubuntu /opt/isaac-lab
# common/ deliberately stays root-owned 0755: the bootstrap downloads and executes code at
# run time, so the runtime user must not be able to rewrite it.
USER ubuntu

RUN /opt/isaac-lab/venv/bin/python -m pip install --no-deps --upgrade \
      "GitPython>=3.1.50" \
      "PyJWT>=2.13.0" \
      "aiohttp>=3.13.3" \
      "aiomysql>=0.3.0" \
      "azure-core>=1.38.0" \
      "cryptography>=48.0.1" \
      "diffusers>=0.38.0" \
      "jaraco.context>=6.1.0" \
      "msgpack>=1.2.1" \
      "onnx>=1.21.0" \
      "pillow>=12.2.0" \
      "python-multipart>=0.0.30" \
      "starlette>=1.3.1" \
      "tornado>=6.5.6" \
      "transformers>=5.3.0" \
      "urllib3>=2.7.0" \
      "wheel>=0.46.2" \
    && (/opt/isaac-lab/venv/bin/python -m pip uninstall -y wandb lxml || true)

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu workflows /opt/npa/workflows
COPY --chown=ubuntu:ubuntu docker/workbench/groot/smoke_env.py /opt/npa/docker/workbench/groot/smoke_env.py
COPY --chown=ubuntu:ubuntu docker/workbench/groot/smoke_functional.py /opt/npa/docker/workbench/groot/smoke_functional.py
COPY --chown=ubuntu:ubuntu docker/workbench/groot/entrypoint.sh /opt/npa/docker/workbench/groot/entrypoint.sh

RUN chmod 0755 /opt/npa/docker/workbench/groot/entrypoint.sh

RUN uv pip install --python "${GROOT_VENV}/bin/python" --no-deps -e /opt/npa \
    && "${GROOT_VENV}/bin/python" -c 'from npa.cli.entry import main; main()' \
      workbench groot finetune --help >/dev/null \
    && "${GROOT_VENV}/bin/python" - <<'PY'
import os
from io import BytesIO
from importlib import metadata
from pathlib import Path

from mcap.reader import make_reader
from mcap.writer import Writer
from npa.cli.groot import DEFAULT_EMBODIMENT_TAG, DEFAULT_MODEL, _build_server_py

expected = os.environ["GROOT_RUNTIME_VERSION"]
actual = metadata.version("gr00t")
if actual != expected:
    raise RuntimeError(f"expected gr00t {expected}, found {actual}")
buffer = BytesIO()
writer = Writer(buffer)
writer.start(profile="", library="npa-groot-build-smoke")
writer.finish()
if make_reader(BytesIO(buffer.getvalue())).get_summary() is None:
    raise RuntimeError("GR00T image could not write and read a native MCAP recording")
Path("/opt/groot/server.py").write_text(_build_server_py(DEFAULT_MODEL, DEFAULT_EMBODIMENT_TAG))
print(f"GROOT_BUILD_IMPORT_OK {actual}")
PY

# Assert the ABSENCE of a baked Isaac install - the inverse of the old
# ISAAC_LAB_BUILD_IMPORT_OK check, which is exactly the assertion a runtime-fetch image
# must not be able to make. The bootstrap verifies the versions after it installs.
RUN /opt/isaac-lab/venv/bin/python - <<'PY' \
    && "${GROOT_VENV}/bin/python" -m py_compile \
      /opt/groot/server.py \
      /opt/npa/docker/workbench/groot/smoke_env.py \
      /opt/npa/docker/workbench/groot/smoke_functional.py
import importlib.util
from importlib import metadata

leaked = sorted(
    dist.metadata["Name"]
    for dist in metadata.distributions()
    if (dist.metadata["Name"] or "").lower().replace("_", "-").startswith(("isaacsim", "isaaclab"))
)
if leaked:
    raise SystemExit(f"FATAL: Isaac packages baked into the image: {leaked}")
for module in ("isaacsim", "isaaclab"):
    if importlib.util.find_spec(module) is not None:
        raise SystemExit(f"FATAL: {module} is importable at build time; it must be runtime-fetched")
print("NPA_GROOT_NO_BAKED_ISAAC_OK")
PY

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends --only-upgrade \
      gpgv openssl libssl3 \
      "linux-libc-dev=${GROOT_LINUX_LIBC_DEV_VERSION}" \
    && apt-get purge -y --auto-remove \
      dirmngr \
      gnupg \
      gnupg-utils \
      gnupg2 \
      gpg \
      gpg-agent \
      software-properties-common \
    && rm -rf /opt/nvidia/nsight-compute \
    `# The isaacsim extscache / pip_prebundle cleanup that used to be here existed only` \
    `# to claw back size from a BAKED Isaac install. There is no baked Isaac any more,` \
    `# so it is gone rather than kept as a no-op.` \
    && rm -rf /home/ubuntu/.cache /root/.cache \
    && rm -rf /var/lib/apt/lists/*

USER ubuntu

WORKDIR /opt/groot
EXPOSE 8080
ENTRYPOINT ["/opt/npa/docker/workbench/groot/entrypoint.sh"]
