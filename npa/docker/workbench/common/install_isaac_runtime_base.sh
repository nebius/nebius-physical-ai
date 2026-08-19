#!/usr/bin/env bash
#
# install_isaac_runtime_base.sh - build-time layer for images that fetch Isaac at run time.
#
# Shared by npa-isaac-lab and npa-sonic (baked variant). Docker has no `include`, and two
# hand-maintained copies of a ~90-line layer would drift, so the logic lives here and each
# Dockerfile does COPY + RUN. This also avoids inventing an extra npa-isaac-base image
# that a build-your-own customer would have to build first.
#
# WHAT THIS BAKES (all freely redistributable, all from PyPI / Docker Hub / Ubuntu):
#   * python3.11 (Isaac requires exactly 3.11: both isaacsim and isaaclab declare
#     `Requires-Python: ==3.11.*`, and Ubuntu 22.04 ships 3.10, hence deadsnakes)
#   * a python3.11 venv with PyTorch and the COMPLETE OSS dependency closure that Isaac
#     Sim / Isaac Lab need at run time
#   * the runtime-fetch bootstrap, its hash-pinned wheel manifest, and the
#     /isaac-sim/python.sh shim
#   * the SkyPilot-on-Kubernetes prerequisites
#
# WHAT THIS DELIBERATELY DOES NOT BAKE:
#   * any NVIDIA Isaac Sim / Isaac Lab wheel (proprietary; fetched on first run under the
#     operator's own EULA acceptance - see isaac_bootstrap.sh)
#   * any EULA acceptance variable (that refusal is the legal mechanism)
#   * NVIDIA driver userspace libraries. A graphics-capable container runtime may inject
#     them. The Antioch/OpenPI stack also supports checksum-verified, driver-matched
#     runtime delivery into an operator-owned volume for managed compute-only nodes.
#     Either path keeps those third-party bytes out of the public image.
#
# Baking the OSS closure is what lets the runtime fetch use --no-deps --require-hashes:
# pip needs no dependency resolution at run time, so every byte it downloads is pinned to
# a sha256 this repo has reviewed.
#
set -euo pipefail

TORCH_VERSION="${TORCH_VERSION:-2.9.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.9.0}"
UBUNTU_SNAPSHOT="${NPA_UBUNTU_SNAPSHOT:-20260801T053000Z}"
PYTHON_VERSION="${NPA_ISAAC_PYTHON_VERSION:-3.11.15-1+jammy1}"
DEADSNAKES_POOL="${NPA_DEADSNAKES_POOL:-https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu/pool/main/p/python3.11}"
# cu128 wheels carry sm_120 kernels, which RTX PRO 6000 Blackwell (compute capability
# 12.0) requires. Verified in-pod: torch._C._cuda_getArchFlags() reports
# "sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120".
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
# Deliberately NOT named /opt/npa/isaac/venv. The redistribution claim is audited by
# listing every path in the built image that matches a loose `isaac|omni` grep, and an
# isaac-named venv directory buries that list under 60,000 unrelated site-packages files.
# Keeping the venv out of the namespace makes the audit output ~20 lines a human can read.
ISAAC_VENV="${NPA_ISAAC_VENV:-/opt/npa/sim/venv}"
RUNTIME_USER="${NPA_RUNTIME_USER:-ubuntu}"
CACHE_DIR="${NPA_ISAAC_CACHE_DIR:-/opt/isaac-cache}"
COMMON_DIR="${NPA_ISAAC_COMMON_DIR:-/opt/npa/docker/workbench/common}"
OSS_DEPS_FILE="${NPA_ISAAC_OSS_DEPS_FILE:-${COMMON_DIR}/isaac-oss-deps.txt}"
INSTALL_SKYPILOT_PREREQS="${NPA_INSTALL_SKYPILOT_PREREQS:-1}"
# Set to 1 when the base image already provides a python3.11 venv with PyTorch (npa-base
# does, at /opt/npa/venv, with cu130 for Blackwell). Reinstalling torch over it would
# waste ~3 GB of layer and risk contradicting the base's own CUDA pairing.
SKIP_TORCH="${NPA_ISAAC_SKIP_TORCH:-0}"
export DEBIAN_FRONTEND=noninteractive

log() { printf '\n=== install_isaac_runtime_base: %s ===\n' "$*"; }

log "system packages"
rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*.list \
  /etc/apt/sources.list.d/*.sources
printf '%s\n' \
  'Types: deb' \
  "URIs: https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/" \
  'Suites: jammy jammy-updates jammy-backports jammy-security' \
  'Components: main restricted universe multiverse' \
  'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg' \
  > /etc/apt/sources.list.d/ubuntu.sources
apt-get update
# CUDA's Jammy base retains an old linux-libc-dev build. These are development
# headers rather than the host kernel, but the fixed immutable-snapshot build is
# available, so upgrade it explicitly instead of carrying avoidable critical CVEs.
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  git \
  git-lfs \
  libegl1 \
  libgl1 \
  libglu1-mesa \
  libglvnd0 \
  libx11-6 \
  libxext6 \
  libxrender1 \
  linux-libc-dev=5.15.0-186.196 \
  libxt6 `# MaterialX render libs dlopen libXt.so.6; without it Kit logs three
          # "Could not load the dynamic library ... libMaterialXRender*.so" errors` \
  vulkan-tools

# Isaac needs python3.11 exactly; Ubuntu 22.04 ships 3.10. Do not add the
# moving deadsnakes apt repository. Fetch the reviewed version's immutable
# package paths and verify every byte against the SHA-256 recorded in the PPA
# Packages index before apt resolves their Ubuntu dependencies from the fixed
# snapshot above.
install -d -m 0755 /tmp/npa-python311
while IFS='|' read -r filename digest; do
  curl --fail --location --retry 3 \
    "${DEADSNAKES_POOL}/${filename}" \
    --output "/tmp/npa-python311/${filename}"
  printf '%s  %s\n' "$digest" "/tmp/npa-python311/${filename}" | sha256sum --check --strict
done <<'PYTHON311_DEBS'
libpython3.11_3.11.15-1+jammy1_amd64.deb|1ef8897f4f56b7e90a2c4bc07b68a7074b77567e4b112ded3331365eb3c10fc2
libpython3.11-dev_3.11.15-1+jammy1_amd64.deb|1adc394918add62fb6e497382046d67b66d4d73cc887cb8be597d9e623db98ad
libpython3.11-minimal_3.11.15-1+jammy1_amd64.deb|2242dc4450d5ef4bb51aa162229dcae9f921f13c44322aefc1a132631afe9493
libpython3.11-stdlib_3.11.15-1+jammy1_amd64.deb|4d9264d06f37fef6515da083efea8f4aed3225b6fcdfaf3ae69fdd22fbaa19fc
python3.11_3.11.15-1+jammy1_amd64.deb|83432e1464c31af89c0e7df5ca9e4655db1eeb3f0cf2427efb3c4c41d64b9e2e
python3.11-dev_3.11.15-1+jammy1_amd64.deb|f50550d76be43a305fa894ab001b76de635738ffbbccc381c43bd205b27efccb
python3.11-minimal_3.11.15-1+jammy1_amd64.deb|7de0e5a79cb46d2c017b3a882980d2ff9d943b3cd2a2c6fdccab6010f1fcd736
python3.11-venv_3.11.15-1+jammy1_amd64.deb|65edd1c51e458d118bee721d0815e0cbcb940ab351d851037a29f5f07a1beef8
PYTHON311_DEBS
apt-get install -y --no-install-recommends /tmp/npa-python311/*.deb
test "$(dpkg-query -W -f='${Version}' python3.11)" = "$PYTHON_VERSION"
rm -rf /tmp/npa-python311

if [ "$INSTALL_SKYPILOT_PREREQS" = "1" ]; then
  # SkyPilot's in-pod Kubernetes bootstrap needs a SYSTEM python3 plus rsync, an SSH
  # client/server, and passwordless sudo; without all of them provisioning fails with
  # `container not found ("ray-node")`. Guarded by
  # npa/tests/guardrails/test_workbench_image_k8s_prereqs.py.
  # Pre-create empty sentinels so openssh-server's post-install keygen skips host
  # identity generation.  Delete them in this same image layer; real keys are
  # generated only in the running container by workflow_runtime_entrypoint.sh.
  install -d -m 0755 /etc/ssh
  touch \
    /etc/ssh/ssh_host_rsa_key \
    /etc/ssh/ssh_host_ecdsa_key \
    /etc/ssh/ssh_host_ed25519_key
  apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip rsync openssh-client openssh-server sudo netcat-openbsd
  rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub
  printf 'ubuntu ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/99-npa-runtime-user
  chmod 0440 /etc/sudoers.d/99-npa-runtime-user
  install -d -m 0755 /run/sshd
fi

rm -rf /var/lib/apt/lists/*

if [ -x "$ISAAC_VENV/bin/python" ]; then
  log "reusing the base image's python venv at ${ISAAC_VENV}"
else
  log "python3.11 venv at ${ISAAC_VENV}"
  python3.11 -m venv "$ISAAC_VENV"
fi
"$ISAAC_VENV/bin/python" -m pip install --no-cache-dir --no-deps \
  "pip==26.2.1" \
  "setuptools==84.0.0" \
  "wheel==0.47.0" \
  "packaging==26.3"
"$ISAAC_VENV/bin/python" -m pip check

# Isaac requires exactly python3.11; fail here rather than at first run.
"$ISAAC_VENV/bin/python" - <<'PY'
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        f"Isaac requires python 3.11 (isaacsim/isaaclab declare Requires-Python: ==3.11.*), "
        f"but {sys.executable} is {sys.version_info.major}.{sys.version_info.minor}"
    )
print(f"python {sys.version.split()[0]} at {sys.executable}")
PY

if [ "$SKIP_TORCH" = "1" ]; then
  log "skipping PyTorch install (NPA_ISAAC_SKIP_TORCH=1; the base image provides it)"
  "$ISAAC_VENV/bin/python" -c 'import torch; print(f"base torch {torch.__version__}")'
else
  log "PyTorch ${TORCH_VERSION} from ${TORCH_INDEX_URL}"
  "$ISAAC_VENV/bin/python" -m pip install --no-cache-dir --no-deps \
    --index-url "$TORCH_INDEX_URL" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}"
fi

log "OSS dependency closure for Isaac Sim / Isaac Lab"
"$ISAAC_VENV/bin/python" -m pip install --no-cache-dir \
  --no-deps \
  -r "${COMMON_DIR}/isaac-oss-deps.txt"
# wheel is a build tool, not part of the runtime. Removing it resolves the
# otherwise impossible wheel>=24 / Isaac-Lab<24 packaging constraint without
# weakening Isaac Lab's declared runtime contract.
"$ISAAC_VENV/bin/python" -m pip uninstall --yes wheel
"$ISAAC_VENV/bin/python" -m pip check

log "runtime-fetch bootstrap, wheel manifest, and the /isaac-sim/python.sh shim"
install -d -m 0755 /opt/npa/bin
install -m 0755 "${COMMON_DIR}/isaac_bootstrap.sh" /opt/npa/bin/isaac-bootstrap
install -m 0755 "${COMMON_DIR}/isaac_python.sh" /opt/npa/bin/isaac-python
# /isaac-sim/python.sh is a compatibility path, not a vendor install: ~30 call sites in
# this repo (SkyPilot templates, the sim2real engine, byo_isaac_*, rl_sweep, retargeting)
# already invoke Isaac through it, and pods override ENTRYPOINT so it is also the only
# reliable bootstrap trigger. Mode 0755 (not NVIDIA's 0750 isaac-sim:isaac-sim) so the
# non-root runtime user can traverse it without a multi-GB recursive chown.
install -d -m 0755 /isaac-sim
install -m 0755 "${COMMON_DIR}/isaac_python.sh" /isaac-sim/python.sh
getent group isaac-sim >/dev/null 2>&1 || groupadd -r isaac-sim
id -u "$RUNTIME_USER" >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 "$RUNTIME_USER"
usermod -aG isaac-sim "$RUNTIME_USER"

log "Isaac cache mount point at ${CACHE_DIR}"
# Group-writable so a cold start works for the non-root runtime user, but the
# recommended production posture is a pre-warmed volume plus
# NPA_ISAAC_CACHE_READONLY=1, where the runtime user needs no write access at all.
install -d -m 0775 -o root -g isaac-sim "$CACHE_DIR"
install -d -m 0775 -o root -g isaac-sim "${CACHE_DIR}/v"

log "verifying the bootstrap contract (must REFUSE without operator EULA acceptance)"
bash -n /opt/npa/bin/isaac-bootstrap
bash -n /opt/npa/bin/isaac-python
/opt/npa/bin/isaac-bootstrap status >/dev/null
# This is the load-bearing legal mechanism, so the build proves the refusal instead of
# proving a baked install. Exit 78 is EX_CONFIG: the operator must act.
set +e
ACCEPT_EULA= env -u OMNI_KIT_ACCEPT_EULA -u ISAACSIM_ACCEPT_EULA \
  /opt/npa/bin/isaac-bootstrap ensure >/dev/null 2>/tmp/eula-refusal.txt
refusal_rc=$?
set -e
if [ "$refusal_rc" -ne 78 ]; then
  echo "FATAL: bootstrap must exit 78 without EULA acceptance, got ${refusal_rc}" >&2
  cat /tmp/eula-refusal.txt >&2
  exit 1
fi
grep -q 'ACCEPT_EULA=Y' /tmp/eula-refusal.txt
rm -f /tmp/eula-refusal.txt
echo "NPA_ISAAC_BOOTSTRAP_REFUSES_WITHOUT_EULA_OK"

log "verifying torch and that NO Isaac wheel is present in the image"
NPA_REQUIRE_TORCH_SM120="${REQUIRE_TORCH_SM120:-1}" "$ISAAC_VENV/bin/python" - <<'PY'
import importlib.util
import os
import sys
from importlib import metadata

import torch

# torch.cuda.get_arch_list() returns [] on a host with no NVIDIA driver, which is the
# normal case for a build machine (the dev VM has no GPU). So the build checks the thing
# it CAN check - that this is a CUDA wheel whose toolkit carries sm_120 kernels - and the
# actual per-device arch assertion happens where a device exists: `isaac-bootstrap verify`
# and the GPU golden evals. Asserting arch_list here would have meant a build that only
# passes on GPU builders, which is a worse contract, not a stronger one.
arch_list = torch.cuda.get_arch_list()
cuda_version = torch.version.cuda or ""
print(f"torch {torch.__version__} cuda={cuda_version or 'cpu'} arch_list={arch_list}")

if os.environ.get("NPA_REQUIRE_TORCH_SM120") == "1":
    if arch_list:
        if "sm_120" not in arch_list:
            raise SystemExit(
                f"expected sm_120 kernels for RTX PRO 6000 Blackwell, got {arch_list}"
            )
        print("NPA_TORCH_SM120_OK (arch list observed on a GPU-capable builder)")
    else:
        # CUDA >= 12.8 is where sm_120 (Blackwell, compute capability 12.0) appears.
        try:
            major, _, minor = cuda_version.partition(".")
            toolkit = (int(major), int(minor or 0))
        except ValueError as exc:
            raise SystemExit(
                f"cannot parse torch CUDA version {cuda_version!r}; expected a CUDA build"
            ) from exc
        if toolkit < (12, 8):
            raise SystemExit(
                f"torch is built against CUDA {cuda_version}, which has no sm_120 "
                f"kernels; RTX PRO 6000 Blackwell needs CUDA >= 12.8"
            )
        print(
            f"NPA_TORCH_CUDA_SUPPORTS_SM120_OK cuda={cuda_version} "
            f"(no driver on this builder, so the arch list is verified on GPU instead)"
        )

# The whole point of this image: no NVIDIA Isaac bytes ship in it.
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
print("NPA_NO_BAKED_ISAAC_OK", file=sys.stderr)
PY

rm -rf /root/.cache /home/"$RUNTIME_USER"/.cache 2>/dev/null || true
log "done"
