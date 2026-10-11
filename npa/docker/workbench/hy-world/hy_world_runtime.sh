#!/usr/bin/env bash
# Operator-runtime delivery for Tencent HY-World 2.0.  This image contains none
# of Tencent's source, checkpoints, caches, data, or generated assets.
set -euo pipefail

readonly EX_CONFIG=78
readonly EX_SOFTWARE=70
readonly SOURCE_REPOSITORY="https://github.com/Tencent-Hunyuan/HY-World-2.0"
readonly SOURCE_REF="df9988efb87bfc0f4947eb3889411cf957478b06"
readonly HY_WORLD_MODEL="tencent/HY-World-2.0"
readonly HY_WORLD_MODEL_REF="d78a16c91c7a56488894a1c8de4f5c7cc28aa8b0"
readonly WORLD_STEREO_MODEL="hanshanxue/WorldStereo"
readonly WORLD_STEREO_REF="ac2ad97ecb043fe80c2f19cd1898006becb9d66e"
readonly WAN_I2V_MODEL="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
readonly WAN_I2V_REF="b184e23a8a16b20f108f727c902e769e873ffc73"
readonly QWEN_IMAGE_MODEL="Qwen/Qwen-Image-Edit-2509"
readonly QWEN_IMAGE_REF="d3968ef930e841f4c73640fb8afa3b306a78167e"
readonly QWEN_VLM_MODEL="Qwen/Qwen3-VL-8B-Instruct"
readonly QWEN_VLM_REF="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
readonly ZIM_MODEL="naver-iv/zim-anything-vitl"
readonly ZIM_REF="667e2d7c233f6f1cacd12ccc64bdf6cc7b5aa16d"
readonly GROUNDING_DINO_MODEL="IDEA-Research/grounding-dino-tiny"
readonly GROUNDING_DINO_REF="a2bb814dd30d776dcf7e30523b00659f4f141c71"
readonly DINOV2_MODEL="facebook/dinov2-base"
readonly DINOV2_REF="f9e44c814b77203eaa57a6bdbbd535f21ede1415"
readonly SAM3_MODEL="facebook/sam3"
readonly SAM3_REF="3c879f39826c281e95690f02c7821c4de09afae7"
readonly MOGE_MODEL="Ruicheng/moge-2-vitl-normal"
readonly MOGE_REF="cb0e8bbd6b1e243589717c78e750b1ba4c093acf"
readonly UNI3C_MODEL="ewrfcas/Uni3C"
readonly UNI3C_REF="fca895f7fb454f4dfed43956cab09180d0946318"
# The selected upstream requirements leave PyTorch3D and FlashAttention mutable.
# Pin the two source-built runtime dependencies here; their resolved packages are
# also recorded in npa_resolved_inventory.txt for every generated scene.
readonly PYTORCH3D_REPOSITORY="https://github.com/facebookresearch/pytorch3d.git"
readonly PYTORCH3D_REF="88e182f989c80836f4bd744e0d9cb1852762ce01"
readonly FLASH_ATTN_VERSION="2.8.3"
readonly PYTORCH_WHEEL_INDEX="https://download.pytorch.org/whl/cu128"
readonly TORCH_VERSION="2.7.1+cu128"
readonly TORCHVISION_VERSION="0.22.1+cu128"
readonly TORCHMETRICS_VERSION="1.9.0"
readonly NATIVE_NUMPY_VERSION="1.26.4"
readonly NATIVE_SCIPY_VERSION="1.14.1"
readonly RERUN_SDK_VERSION="0.31.4"
readonly VGG16_WEIGHTS_URL="https://download.pytorch.org/models/vgg16-397923af.pth"
readonly VGG16_WEIGHTS_FILENAME="vgg16-397923af.pth"
readonly VGG16_WEIGHTS_SHA256_PREFIX="397923af"
readonly VGG16_WEIGHTS_SHA256="397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
readonly RUNTIME_CONTRACT_REVISION="hy-world-hub-ref-report-env-model-closure-v1"
readonly RUNTIME_BOOTSTRAP_SHA256="$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"

# Resolve this once, before configure_runtime_hf_cache repoints HF_HOME at this
# candidate's immutable model closure.  The source/venv cache belongs beside
# the shared model-cache location selected for the pod, never beneath a later
# nested HF_HOME value.
if [[ -z "${NPA_HY_WORLD_RUNTIME_CACHE:-}" ]]; then
  export NPA_HY_WORLD_RUNTIME_CACHE="${HF_HOME:-/workspace/model-cache/huggingface}/hy-world/runtime"
fi

cache_root() {
  printf '%s\n' "$NPA_HY_WORLD_RUNTIME_CACHE"
}

runtime_hf_home() {
  local closure
  closure="${SOURCE_REF}:${HY_WORLD_MODEL_REF}:${WORLD_STEREO_REF}:${WAN_I2V_REF}:${QWEN_IMAGE_REF}:${ZIM_REF}:${GROUNDING_DINO_REF}:${DINOV2_REF}:${SAM3_REF}:${MOGE_REF}:${UNI3C_REF}:${VGG16_WEIGHTS_SHA256}"
  printf '%s/hf-%s\n' "$(cache_root)" \
    "$(printf '%s' "$closure" | sha256sum | cut -c1-24)"
}

configure_runtime_hf_cache() {
  local home
  home="$(runtime_hf_home)"
  mkdir -p "$home"
  export HF_HOME="$home"
  export HF_HUB_CACHE="$home/hub"
  export HUGGINGFACE_HUB_CACHE="$home/hub"
}

configure_runtime_torch_cache() {
  local closure
  closure="${TORCH_VERSION}:${TORCHVISION_VERSION}:${VGG16_WEIGHTS_SHA256}"
  export TORCH_HOME="$(cache_root)/torch-$(printf '%s' "$closure" | sha256sum | cut -c1-24)"
  mkdir -p "$TORCH_HOME"
}

source_tree() {
  printf '%s/source-%s\n' "$(cache_root)" "$SOURCE_REF"
}

runtime_python() {
  printf '%s/.venv/bin/python\n' "$(source_tree)"
}

report_python() {
  printf '%s/.report-venv/bin/python\n' "$(source_tree)"
}

prefer_native_runtime_bin() {
  export PATH="$(source_tree)/.venv/bin:$PATH"
}

torch_probe_path() {
  local tree="$1"
  printf '%s/npa_torch_cuda_probe.json\n' "$tree"
}

validate_runtime_paths() {
  local root tree
  root="$(cache_root)"
  tree="$(source_tree)"
  [[ "$root" == /* && "$root" != / && "$tree" == "$root"/source-"$SOURCE_REF" ]] || {
    echo "npa-hy-world: runtime cache must be a non-root absolute directory" >&2
    exit "$EX_CONFIG"
  }
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "npa-hy-world: required command is missing: $1" >&2
    exit "$EX_SOFTWARE"
  }
}

require_value() {
  local name="$1" value="${!1:-}"
  if [[ -z "$value" ]]; then
    echo "npa-hy-world: $name is required for image-to-world generation" >&2
    exit "$EX_CONFIG"
  fi
}

terms() {
  cat <<'EOF'
Tencent HY-WORLD 2.0 Community License Agreement (2026-04-15) governs Tencent
HY-World source, inference/training code, algorithms, and trained weights.

The Agreement grants no right in the EU, UK, or South Korea. An entity with more
than 1,000,000 monthly active users at the release date needs a Tencent license.
It also forbids using HY-World works or Outputs to train/improve another AI model.
This bootstrap cannot determine an operator's territory, MAU, vendor grant, or
downstream use. It deliberately has no ACCEPT_TERMS-style self-certification.

Use is Tencent's documented acceptance mechanism. NPA neither collects a
duplicate acceptance nor asks the operator to self-attest territory or MAU; it
does not claim to create a Tencent grant. The operator remains responsible for
using the fetched components and Outputs under the Agreement. Complete exact
upstream access checks for every fetched component. This image does not
redistribute Tencent bytes; it fetches them only into the operator's mounted
runtime/model cache.

The released trajectory stage also loads Meta SAM 3 from Hugging Face. Its
repository requires the operator to agree to its own access terms and share
contact information. That is a separate upstream approval and is not replaced
by this bootstrap or a Tencent record.
EOF
}

health() {
  require_command git
  require_command uv
  require_command hf
  require_command ffmpeg
  require_command ffprobe
  require_command sha256sum
  echo "NPA_HY_WORLD_BOOTSTRAP_HEALTH_OK"
}

status() {
  local tree
  tree="$(source_tree)"
  printf 'source_repository=%s\nsource_ref=%s\nsource_ready=%s\n' \
    "$SOURCE_REPOSITORY" "$SOURCE_REF" "$([[ -f "$tree/.complete" ]] && echo true || echo false)"
  printf 'runtime_cache=%s\nhf_home=%s\n' "$(cache_root)" "${HF_HOME:-<unset>}"
}

bootstrap_integrity() {
  # A running container includes mutable operator inputs and runtime caches. The
  # complete-byte OCI scan is the only payload-absence gate; do not inspect the
  # mutable filesystem and mistake an authorized staged input for an image layer.
  local required
  for required in \
    /opt/npa/hy-world/asset_contract.py \
    /opt/npa/hy-world/validate_scene.py \
    /opt/npa/hy-world/hy_world_report.py \
    /opt/npa/hy-world/workflow_runner.py \
    /usr/local/bin/hy-world-runtime; do
    [[ -r "$required" ]] || {
      echo "npa-hy-world: bootstrap file is absent: $required" >&2
      exit "$EX_SOFTWARE"
    }
  done
  echo "NPA_HY_WORLD_BOOTSTRAP_INTEGRITY_OK"
}

source_is_complete() {
  local tree
  tree="$(source_tree)"
  [[ -f "$tree/.complete" ]] \
    && [[ "$(git -C "$tree" rev-parse HEAD)" == "$SOURCE_REF" ]] \
    && verify_runtime_contract "$tree" >/dev/null 2>&1
}

verify_runtime_contract() {
  local tree="$1" contract
  contract="$tree/npa_runtime_contract.json"
  python3 - "$contract" "$RUNTIME_CONTRACT_REVISION" "$TORCH_VERSION" \
    "$TORCHVISION_VERSION" "$PYTORCH_WHEEL_INDEX" "$NATIVE_NUMPY_VERSION" \
    "$NATIVE_SCIPY_VERSION" "$TORCHMETRICS_VERSION" "$RERUN_SDK_VERSION" \
    "$VGG16_WEIGHTS_SHA256" "$RUNTIME_BOOTSTRAP_SHA256" <<'PY'
import json
import sys
from pathlib import Path

(
    path,
    revision,
    torch,
    torchvision,
    wheel_index,
    native_numpy,
    native_scipy,
    torchmetrics,
    rerun_sdk,
    vgg16_sha256,
    bootstrap_sha256,
) = sys.argv[1:]
path = Path(path)
try:
    observed = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"runtime contract is unreadable: {exc}") from exc
expected = {
    "schema": "npa.hy_world.runtime_contract.v1",
    "revision": str(revision),
    "torch": str(torch),
    "torchvision": str(torchvision),
    "pytorch_wheel_index": str(wheel_index),
    "native_numpy": str(native_numpy),
    "native_scipy": str(native_scipy),
    "torchmetrics": str(torchmetrics),
    "report_rerun_sdk": str(rerun_sdk),
    "vgg16_sha256": str(vgg16_sha256),
    "bootstrap_sha256": str(bootstrap_sha256),
}
if observed != expected:
    raise SystemExit("runtime contract does not match the current bootstrap")
PY
}

write_runtime_contract() {
  local tree="$1" contract
  contract="$tree/npa_runtime_contract.json"
  "$(runtime_python)" - "$contract" "$RUNTIME_CONTRACT_REVISION" "$TORCH_VERSION" \
    "$TORCHVISION_VERSION" "$PYTORCH_WHEEL_INDEX" "$NATIVE_NUMPY_VERSION" \
    "$NATIVE_SCIPY_VERSION" "$TORCHMETRICS_VERSION" "$RERUN_SDK_VERSION" \
    "$VGG16_WEIGHTS_SHA256" "$RUNTIME_BOOTSTRAP_SHA256" <<'PY'
import json
import sys
from pathlib import Path

(
    path,
    revision,
    torch,
    torchvision,
    wheel_index,
    native_numpy,
    native_scipy,
    torchmetrics,
    rerun_sdk,
    vgg16_sha256,
    bootstrap_sha256,
) = sys.argv[1:]
path = Path(path)
payload = {
    "schema": "npa.hy_world.runtime_contract.v1",
    "revision": str(revision),
    "torch": str(torch),
    "torchvision": str(torchvision),
    "pytorch_wheel_index": str(wheel_index),
    "native_numpy": str(native_numpy),
    "native_scipy": str(native_scipy),
    "torchmetrics": str(torchmetrics),
    "report_rerun_sdk": str(rerun_sdk),
    "vgg16_sha256": str(vgg16_sha256),
    "bootstrap_sha256": str(bootstrap_sha256),
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

install_blackwell_torch_wheels() {
  local python="$1"
  uv pip install --python "$python" --index-url "$PYTORCH_WHEEL_INDEX" \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"
}

install_upstream_requirements() {
  local tree="$1" python="$2" requirements
  grep -qx 'torch==2.7.1' "$tree/requirements.txt" \
    && grep -qx 'torchvision==0.22.1' "$tree/requirements.txt" \
    && grep -qx "numpy==${NATIVE_NUMPY_VERSION}" "$tree/requirements.txt" \
    && grep -qx "scipy==${NATIVE_SCIPY_VERSION}" "$tree/requirements.txt" \
    && grep -qx 'torchmetrics' "$tree/requirements.txt" || {
      echo "npa-hy-world: upstream native requirements drifted from the reviewed runtime closure" >&2
      exit "$EX_SOFTWARE"
    }
  requirements="$tree/npa_requirements_without_torch.txt"
  sed -E '/^(torch|torchvision)==/d; /^torchmetrics$/d' "$tree/requirements.txt" > "$requirements"
  uv pip install --python "$python" -r "$requirements" \
    "torchmetrics==${TORCHMETRICS_VERSION}"
}

install_report_runtime() {
  local tree native report
  tree="$1"
  native="$2"
  report="$(report_python)"
  uv venv --python "$native" "$tree/.report-venv"
  uv pip install --python "$report" "rerun-sdk==${RERUN_SDK_VERSION}"
}

record_native_dependency_identity() {
  local tree python
  tree="$1"
  python="$tree/.venv/bin/python"
  "$python" - "$tree/npa_native_dependency_identity.json" \
    "$NATIVE_NUMPY_VERSION" "$NATIVE_SCIPY_VERSION" "$TORCHMETRICS_VERSION" <<'PY'
import json
import sys
from importlib import metadata
from pathlib import Path

path = Path(sys.argv[1])
expected_numpy, expected_scipy, expected_torchmetrics = sys.argv[2:]
observed = {
    "numpy": metadata.version("numpy"),
    "scipy": metadata.version("scipy"),
    "torchmetrics": metadata.version("torchmetrics"),
}
expected = {
    "numpy": expected_numpy,
    "scipy": expected_scipy,
    "torchmetrics": expected_torchmetrics,
}
if observed != expected:
    raise SystemExit(f"native dependency closure drifted: {observed}")
try:
    metadata.version("rerun-sdk")
except metadata.PackageNotFoundError:
    pass
else:
    raise SystemExit("rerun-sdk must remain isolated from the native runtime")
payload = {"schema": "npa.hy_world.native_dependencies.v1", "python": sys.version.split()[0], **observed}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

record_report_dependency_identity() {
  local tree python
  tree="$1"
  python="$tree/.report-venv/bin/python"
  "$python" - "$tree/npa_report_dependency_identity.json" "$RERUN_SDK_VERSION" <<'PY'
import json
import sys
from importlib import metadata
from pathlib import Path

import rerun

path, expected_rerun = Path(sys.argv[1]), sys.argv[2]
observed = {"rerun_sdk": metadata.version("rerun-sdk"), "numpy": metadata.version("numpy")}
if observed["rerun_sdk"] != expected_rerun or int(observed["numpy"].split(".", 1)[0]) < 2:
    raise SystemExit(f"report dependency closure drifted: {observed}")
payload = {
    "schema": "npa.hy_world.report_dependencies.v1",
    "python": sys.version.split()[0],
    "rerun_module": str(rerun.__version__),
    **observed,
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

fetch_torch_auxiliary_asset() {
  local tree asset
  tree="$1"
  configure_runtime_torch_cache
  asset="$tree/npa_torch_auxiliary_assets.json"
  "$(runtime_python)" - "$asset" "$TORCH_HOME" "$VGG16_WEIGHTS_URL" \
    "$VGG16_WEIGHTS_FILENAME" "$VGG16_WEIGHTS_SHA256_PREFIX" \
    "$VGG16_WEIGHTS_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

out, torch_home, url, filename, prefix, expected_digest = sys.argv[1:]
if Path(torch_home) != Path(torch.hub.get_dir()).parent:
    raise SystemExit("torch hub cache does not use the isolated TORCH_HOME")
cache = Path(torch.hub.get_dir()) / "checkpoints"
cache.mkdir(parents=True, exist_ok=True)
path = cache / filename
if not path.is_file():
    torch.hub.download_url_to_file(url, str(path), hash_prefix=prefix, progress=False)
with path.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
if not digest.startswith(prefix) or digest != expected_digest:
    raise SystemExit(f"torch auxiliary asset digest disagrees with the reviewed blob: {digest}")
payload = {
    "schema": "npa.hy_world.torch_auxiliary_assets.v1",
    "assets": [{"url": url, "filename": filename, "sha256": digest}],
}
target = Path(out)
temporary = target.with_suffix(target.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(target)
PY
}

verify_torch_auxiliary_asset() {
  local tree
  tree="$1"
  configure_runtime_torch_cache
  "$(runtime_python)" - "$tree/npa_torch_auxiliary_assets.json" "$TORCH_HOME" \
    "$VGG16_WEIGHTS_URL" "$VGG16_WEIGHTS_FILENAME" "$VGG16_WEIGHTS_SHA256_PREFIX" \
    "$VGG16_WEIGHTS_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

manifest, torch_home, url, filename, prefix, expected_digest = sys.argv[1:]
if Path(torch_home) != Path(torch.hub.get_dir()).parent:
    raise SystemExit("torch hub cache does not use the isolated TORCH_HOME")
path = Path(torch.hub.get_dir()) / "checkpoints" / filename
try:
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"torch auxiliary manifest is unreadable: {exc}") from exc
expected = {"url": url, "filename": filename}
assets = payload.get("assets")
if not isinstance(assets, list) or len(assets) != 1 or not isinstance(assets[0], dict):
    raise SystemExit("torch auxiliary manifest has an invalid asset list")
record = assets[0]
digest = str(record.get("sha256") or "")
if {key: record.get(key) for key in expected} != expected or len(digest) != 64:
    raise SystemExit("torch auxiliary manifest disagrees with the reviewed asset")
if not path.is_file():
    raise SystemExit("torch auxiliary asset is absent or changed")
with path.open("rb") as stream:
    actual_digest = hashlib.file_digest(stream, "sha256").hexdigest()
if actual_digest != digest or digest != expected_digest:
    raise SystemExit("torch auxiliary asset is absent or changed")
if not digest.startswith(prefix):
    raise SystemExit("torch auxiliary asset no longer matches the official URL prefix")
PY
}

verify_torch_blackwell_runtime() {
  local tree="$1" python probe
  python="$tree/.venv/bin/python"
  probe="$(torch_probe_path "$tree")"
  "$python" - "$probe" "$TORCH_VERSION" "$TORCHVISION_VERSION" <<'PY'
import json
import sys
from importlib import metadata
from pathlib import Path

import torch

probe, expected_torch, expected_torchvision = sys.argv[1:]
if torch.__version__ != str(expected_torch):
    raise SystemExit(f"expected torch {expected_torch}, got {torch.__version__}")
if metadata.version("torchvision") != str(expected_torchvision):
    raise SystemExit(f"expected torchvision {expected_torchvision}")
if torch.version.cuda != "12.8":
    raise SystemExit(f"expected torch CUDA 12.8, got {torch.version.cuda}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit("HY-World requires eight visible CUDA devices")
arch_list = sorted(set(torch.cuda.get_arch_list()))
arch_flags = str(torch._C._cuda_getArchFlags()).split()
compiled_arches = sorted(set(arch_list) | set(arch_flags))
if "sm_100" not in compiled_arches:
    raise SystemExit(f"torch wheel lacks B200 sm_100 support: {compiled_arches}")
capabilities = [list(torch.cuda.get_device_capability(index)) for index in range(8)]
if any(capability != [10, 0] for capability in capabilities):
    raise SystemExit(f"expected B200 capability [10, 0], got {capabilities}")
reductions = []
for index in range(8):
    value = torch.arange(1024, device=f"cuda:{index}", dtype=torch.float32).sum()
    torch.cuda.synchronize(index)
    if not torch.isfinite(value).item() or value.item() != 523776.0:
        raise SystemExit(f"CUDA reduction failed on device {index}")
    reductions.append(value.item())
payload = {
    "schema": "npa.hy_world.torch_cuda_probe.v1",
    "torch": torch.__version__,
    "torchvision": metadata.version("torchvision"),
    "cuda": torch.version.cuda,
    "get_arch_list": arch_list,
    "compiled_arches": compiled_arches,
    "device_capabilities": capabilities,
    "device_count": torch.cuda.device_count(),
    "reductions": reductions,
}
path = Path(probe)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
print("NPA_HY_WORLD_TORCH_B200_CUDA_PROBE_PASSED")
PY
}

install_runtime() {
  local tree="$1" python root pinned_git_requirements
  root="$(dirname "$tree")"
  export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$root/uv-python}"
  # Worldgen declares Python 3.11+, while the neutral CUDA base intentionally
  # carries no Python runtime payload. uv installs this generic interpreter in
  # the operator's mounted cache, never into an image layer.
  uv python install 3.11
  uv venv --python 3.11 "$tree/.venv"
  python="$tree/.venv/bin/python"
  install_blackwell_torch_wheels "$python"
  install_upstream_requirements "$tree" "$python"
  verify_torch_blackwell_runtime "$tree"
  # Upstream's requirements_git.txt intentionally leaves PyTorch3D on its
  # default branch. Replace only that mutable line with the reviewed immutable
  # commit while preserving its other exact requirements.
  pinned_git_requirements="$tree/npa_requirements_git_pinned.txt"
  sed '\|github.com/facebookresearch/pytorch3d.git|d' "$tree/requirements_git.txt" > "$pinned_git_requirements"
  uv pip install --python "$python" --no-build-isolation -r "$pinned_git_requirements" \
    "git+${PYTORCH3D_REPOSITORY}@${PYTORCH3D_REF}"
  # The released image-to-world path imports both native extensions. Compile
  # them in the authorized operator cache with the CUDA toolchain in the
  # neutral base image; no extension binary is baked or redistributed.
  require_command nvcc
  require_command g++
  uv pip install --python "$python" --no-build-isolation \
    -e "$tree/hyworld2/worldgen/third_party/gsplat_maskgaussian"
  RECAST_PATH="$tree/hyworld2/worldgen/third_party/recastnavigation" \
    uv pip install --python "$python" --no-build-isolation \
      "$tree/hyworld2/worldgen/third_party/navmesh"
  uv pip install --python "$python" --no-build-isolation "flash-attn==${FLASH_ATTN_VERSION}"
  "$python" - <<'PY'
import flash_attn
import gsplat
import pytorch3d
import recast
PY
  verify_torch_blackwell_runtime "$tree"
  uv pip freeze --python "$python" | LC_ALL=C sort > "$tree/npa_resolved_inventory.txt"
  record_native_dependency_identity "$tree"
  fetch_torch_auxiliary_asset "$tree"
  install_report_runtime "$tree" "$python"
  record_report_dependency_identity "$tree"
  uv pip freeze --python "$(report_python)" | LC_ALL=C sort \
    > "$tree/npa_report_resolved_inventory.txt"
}

verify_runtime_layout() {
  local tree="$1" python report stale
  python="$tree/.venv/bin/python"
  report="$tree/.report-venv/bin/python"
  [[ -x "$python" ]] || {
    echo "npa-hy-world: runtime virtualenv is absent: $python" >&2
    exit "$EX_SOFTWARE"
  }
  [[ -x "$report" ]] || {
    echo "npa-hy-world: report virtualenv is absent: $report" >&2
    exit "$EX_SOFTWARE"
  }
  "$python" - "$tree" <<'PY'
from pathlib import Path
import sys

tree = Path(sys.argv[1]).resolve()
expected = tree / ".venv"
if Path(sys.prefix).resolve() != expected:
    raise SystemExit(
        f"runtime virtualenv prefix {Path(sys.prefix).resolve()} does not match {expected}"
    )
PY
  "$report" - "$tree" <<'PY'
from pathlib import Path
import sys

tree = Path(sys.argv[1]).resolve()
expected = tree / ".report-venv"
if Path(sys.prefix).resolve() != expected:
    raise SystemExit(
        f"report virtualenv prefix {Path(sys.prefix).resolve()} does not match {expected}"
    )
PY
  stale="$(grep -RIl -- "$tree.incomplete" "$tree/.venv/bin" "$tree/.report-venv/bin" 2>/dev/null || true)"
  [[ -z "$stale" ]] || {
    echo "npa-hy-world: runtime entrypoint still references staging tree: $stale" >&2
    exit "$EX_SOFTWARE"
  }
  verify_runtime_contract "$tree"
  record_native_dependency_identity "$tree"
  record_report_dependency_identity "$tree"
  verify_torch_auxiliary_asset "$tree"
}

ensure() {
  health
  local root tree parent tmp lock
  validate_runtime_paths
  root="$(cache_root)"
  tree="$(source_tree)"
  parent="$(dirname "$tree")"
  mkdir -p "$parent"
  lock="$root/.bootstrap.lock"
  exec 9>"$lock"
  flock 9
  if source_is_complete; then
    # A cache created by an older bootstrap may have moved a virtualenv or an
    # editable install from .incomplete. Never accept it without rechecking.
    verify_runtime_layout "$tree"
    verify_torch_blackwell_runtime "$tree"
    return
  fi
  rm -rf -- "$tree.incomplete" "$tree"
  tmp="$tree.incomplete"
  mkdir -p "$tmp"
  git -C "$tmp" init -q
  git -C "$tmp" remote add origin "$SOURCE_REPOSITORY"
  git -C "$tmp" fetch --depth=1 origin "$SOURCE_REF"
  git -C "$tmp" checkout --detach -q FETCH_HEAD
  [[ "$(git -C "$tmp" rev-parse HEAD)" == "$SOURCE_REF" ]] || {
    echo "npa-hy-world: fetched source revision does not match pin" >&2
    exit "$EX_SOFTWARE"
  }
  git -C "$tmp" submodule update --init --recursive
  git -C "$tmp" submodule status --recursive > "$tmp/npa_submodules.txt"
  if grep -qE '^[+-]' "$tmp/npa_submodules.txt"; then
    echo "npa-hy-world: an upstream submodule is missing or not pinned" >&2
    exit "$EX_SOFTWARE"
  fi
  mv "$tmp" "$tree"
  # Python console scripts and editable installs retain absolute paths. Move
  # only the fetched source, then create the virtualenv at its final location.
  install_runtime "$tree"
  write_runtime_contract "$tree"
  verify_runtime_layout "$tree"
  : > "$tree/.complete"
}

hub_cache_directory() {
  local model="$1"
  require_value HF_HOME
  printf '%s/hub/models--%s\n' "$HF_HOME" "${model//\//--}"
}

register_hub_main_ref() {
  local model="$1" ref="$2" cache refs
  cache="$(hub_cache_directory "$model")"
  [[ -d "$cache/snapshots/$ref" ]] || {
    echo "npa-hy-world: pinned Hugging Face snapshot is absent: $model@$ref" >&2
    exit "$EX_SOFTWARE"
  }
  refs="$cache/refs"
  mkdir -p "$refs"
  if [[ -f "$refs/main" ]]; then
    python3 - "$refs/main" "$ref" <<'PY' || {
import os
import sys
from pathlib import Path

path, ref = Path(sys.argv[1]), sys.argv[2]
observed = path.read_bytes()
expected = ref.encode("ascii")
if observed == expected:
    raise SystemExit(0)
if observed != expected + b"\n":
    raise SystemExit("isolated cache main ref disagrees")
temporary = path.with_name(path.name + ".tmp")
temporary.write_bytes(expected)
os.replace(temporary, path)
PY
      echo "npa-hy-world: isolated cache main ref disagrees for $model" >&2
      exit "$EX_SOFTWARE"
    }
    return
  fi
  printf '%s' "$ref" > "$refs/main.tmp"
  mv "$refs/main.tmp" "$refs/main"
}

fetch_model_snapshot() {
  local model="$1" ref="$2"
  shift 2
  hf download "$model" --revision "$ref" "$@" >/dev/null
  # Upstream loaders use their model IDs at the default `main` revision. Bind
  # those lookups to the exact snapshot before enabling offline mode rather
  # than allowing a later remote main to change a run.
  register_hub_main_ref "$model" "$ref"
}

fetch_models() {
  local root model_manifest lock
  validate_runtime_paths
  root="$(cache_root)"
  mkdir -p "$root"
  configure_runtime_hf_cache
  # Model IDs in the upstream code resolve `refs/main`. Serialize population of
  # this immutable component closure so another worker cannot observe a partial
  # snapshot or rewrite an otherwise identical reference during a cold start.
  lock="$root/.model-fetch.lock"
  exec 8>"$lock"
  flock 8
  # Every model ref is immutable. `HF_HUB_OFFLINE=1` is set only after these
  # downloads so upstream's model-id calls cannot silently resolve a later main.
  fetch_model_snapshot "$HY_WORLD_MODEL" "$HY_WORLD_MODEL_REF"
  fetch_model_snapshot "$WORLD_STEREO_MODEL" "$WORLD_STEREO_REF"
  fetch_model_snapshot "$WAN_I2V_MODEL" "$WAN_I2V_REF"
  fetch_model_snapshot "$QWEN_IMAGE_MODEL" "$QWEN_IMAGE_REF"
  fetch_model_snapshot "$ZIM_MODEL" "$ZIM_REF" --include 'zim_vit_l_2092/*'
  fetch_model_snapshot "$GROUNDING_DINO_MODEL" "$GROUNDING_DINO_REF" \
    --include '*.json' --include '*.txt' --include 'model.safetensors'
  fetch_model_snapshot "$DINOV2_MODEL" "$DINOV2_REF"
  fetch_model_snapshot "$SAM3_MODEL" "$SAM3_REF"
  fetch_model_snapshot "$MOGE_MODEL" "$MOGE_REF"
  fetch_model_snapshot "$UNI3C_MODEL" "$UNI3C_REF" --include 'controlnet.pth'
  model_manifest="$root/npa_model_snapshot_revisions.json"
  "$(runtime_python)" - "$model_manifest" <<PY
import json
from pathlib import Path

payload = {
    "schema": "npa.hy_world.runtime_model_snapshots.v1",
    "models": [
        {"repository": "$HY_WORLD_MODEL", "revision": "$HY_WORLD_MODEL_REF"},
        {"repository": "$WORLD_STEREO_MODEL", "revision": "$WORLD_STEREO_REF"},
        {"repository": "$WAN_I2V_MODEL", "revision": "$WAN_I2V_REF"},
        {"repository": "$QWEN_IMAGE_MODEL", "revision": "$QWEN_IMAGE_REF"},
        {"repository": "$ZIM_MODEL", "revision": "$ZIM_REF"},
        {"repository": "$GROUNDING_DINO_MODEL", "revision": "$GROUNDING_DINO_REF"},
        {"repository": "$DINOV2_MODEL", "revision": "$DINOV2_REF"},
        {"repository": "$SAM3_MODEL", "revision": "$SAM3_REF"},
        {"repository": "$MOGE_MODEL", "revision": "$MOGE_REF"},
        {"repository": "$UNI3C_MODEL", "revision": "$UNI3C_REF"},
    ],
}
path = Path(__import__("sys").argv[1])
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
tmp.replace(path)
PY
}

decode_prompt() {
  require_value HY_WORLD_PROMPT_B64
  "$(runtime_python)" - <<'PY'
import base64
import os
import sys

try:
    sys.stdout.write(base64.b64decode(os.environ["HY_WORLD_PROMPT_B64"], validate=True).decode("utf-8"))
except (KeyError, ValueError, UnicodeDecodeError) as exc:
    raise SystemExit(f"invalid HY_WORLD_PROMPT_B64: {exc}")
PY
}

write_runtime_metadata() {
  local out="$1" image="$2" tree python
  tree="$(source_tree)"
  python="$(runtime_python)"
  "$python" - "$out" "$image" "$tree" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

out, image, tree = map(Path, sys.argv[1:])
if not str(image).startswith("sha256:") and "@sha256:" not in str(image):
    raise SystemExit("HY_WORLD_CONTAINER_IMAGE must be an immutable digest reference")
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

runtime_cache = tree.parent
model_manifest = runtime_cache / "npa_model_snapshot_revisions.json"
if not model_manifest.is_file():
    raise SystemExit("model snapshot revision manifest is absent")
inventory = tree / "npa_resolved_inventory.txt"
report_inventory = tree / "npa_report_resolved_inventory.txt"
native_dependencies = tree / "npa_native_dependency_identity.json"
report_dependencies = tree / "npa_report_dependency_identity.json"
torch_auxiliary_assets = tree / "npa_torch_auxiliary_assets.json"
submodules = tree / "npa_submodules.txt"
runtime_contract = tree / "npa_runtime_contract.json"
torch_probe = tree / "npa_torch_cuda_probe.json"
for required in (
    inventory,
    report_inventory,
    native_dependencies,
    report_dependencies,
    torch_auxiliary_assets,
    submodules,
    runtime_contract,
    torch_probe,
):
    if not required.is_file():
        raise SystemExit(f"required runtime provenance is absent: {required}")
payload = {
    "pipeline_mode": "image_to_world",
    "stages": ["pano", "traj", "render", "expand", "gs_data", "gs_train"],
    "source_repository": "https://github.com/Tencent-Hunyuan/HY-World-2.0",
    "source_ref": "df9988efb87bfc0f4947eb3889411cf957478b06",
    "hy_world_model": {"repository": "tencent/HY-World-2.0", "ref": "d78a16c91c7a56488894a1c8de4f5c7cc28aa8b0"},
    "worldstereo_model": {"repository": "hanshanxue/WorldStereo", "ref": "ac2ad97ecb043fe80c2f19cd1898006becb9d66e"},
    "wan_i2v_model": {"repository": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", "ref": "b184e23a8a16b20f108f727c902e769e873ffc73"},
    "qwen_image_model": {"repository": "Qwen/Qwen-Image-Edit-2509", "ref": "d3968ef930e841f4c73640fb8afa3b306a78167e"},
    "qwen_vlm": {"repository": "Qwen/Qwen3-VL-8B-Instruct", "ref": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"},
    "zim_model": {"repository": "naver-iv/zim-anything-vitl", "ref": "667e2d7c233f6f1cacd12ccc64bdf6cc7b5aa16d"},
    "grounding_dino_model": {"repository": "IDEA-Research/grounding-dino-tiny", "ref": "a2bb814dd30d776dcf7e30523b00659f4f141c71"},
    "dinov2_model": {"repository": "facebook/dinov2-base", "ref": "f9e44c814b77203eaa57a6bdbbd535f21ede1415"},
    "sam3_model": {"repository": "facebook/sam3", "ref": "3c879f39826c281e95690f02c7821c4de09afae7"},
    "moge_model": {"repository": "Ruicheng/moge-2-vitl-normal", "ref": "cb0e8bbd6b1e243589717c78e750b1ba4c093acf"},
    "uni3c_model": {"repository": "ewrfcas/Uni3C", "ref": "fca895f7fb454f4dfed43956cab09180d0946318"},
    "runtime_build_dependencies": {
        "pytorch3d": {"repository": "https://github.com/facebookresearch/pytorch3d.git", "ref": "88e182f989c80836f4bd744e0d9cb1852762ce01"},
        "flash_attn": {"package": "flash-attn", "version": "2.8.3"},
        "gsplat_maskgaussian": {"source_ref": "df9988efb87bfc0f4947eb3889411cf957478b06"},
        "navmesh": {"source_ref": "df9988efb87bfc0f4947eb3889411cf957478b06"},
    },
    "container_image": str(image),
    "resolved_python_packages": {"path": "npa_resolved_inventory.txt", "sha256": sha256(inventory)},
    "report_resolved_python_packages": {"path": "npa_report_resolved_inventory.txt", "sha256": sha256(report_inventory)},
    "native_runtime_dependencies": {"path": "npa_native_dependency_identity.json", "sha256": sha256(native_dependencies)},
    "report_runtime_dependencies": {"path": "npa_report_dependency_identity.json", "sha256": sha256(report_dependencies)},
    "torch_auxiliary_assets": {"path": "npa_torch_auxiliary_assets.json", "sha256": sha256(torch_auxiliary_assets)},
    "submodules": {"path": "npa_submodules.txt", "sha256": sha256(submodules)},
    "model_snapshot_revisions": {"path": "npa_model_snapshot_revisions.json", "sha256": sha256(model_manifest)},
    "runtime_contract": {"path": "npa_runtime_contract.json", "sha256": sha256(runtime_contract)},
    "torch_cuda_probe": {"path": "npa_torch_cuda_probe.json", "sha256": sha256(torch_probe)},
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
Path(out.parent / "npa_resolved_inventory.txt").write_bytes(inventory.read_bytes())
Path(out.parent / "npa_report_resolved_inventory.txt").write_bytes(report_inventory.read_bytes())
Path(out.parent / "npa_native_dependency_identity.json").write_bytes(native_dependencies.read_bytes())
Path(out.parent / "npa_report_dependency_identity.json").write_bytes(report_dependencies.read_bytes())
Path(out.parent / "npa_torch_auxiliary_assets.json").write_bytes(torch_auxiliary_assets.read_bytes())
Path(out.parent / "npa_submodules.txt").write_bytes(submodules.read_bytes())
Path(out.parent / "npa_model_snapshot_revisions.json").write_bytes(model_manifest.read_bytes())
Path(out.parent / "npa_runtime_contract.json").write_bytes(runtime_contract.read_bytes())
Path(out.parent / "npa_torch_cuda_probe.json").write_bytes(torch_probe.read_bytes())
PY
}

run_image_to_world() {
  require_value HY_WORLD_INPUT_IMAGE
  require_value HY_WORLD_SCENE_DIR
  require_value HY_WORLD_RESULT_DIR
  require_value NPA_HY_WORLD_LLM_ADDR
  require_value HY_WORLD_CONTAINER_IMAGE
  ensure
  fetch_models
  local tree python report worldgen scene result input out prompt
  tree="$(source_tree)"
  python="$(runtime_python)"
  report="$(report_python)"
  configure_runtime_torch_cache
  prefer_native_runtime_bin
  worldgen="$tree/hyworld2/worldgen"
  scene="$HY_WORLD_SCENE_DIR"
  result="$HY_WORLD_RESULT_DIR"
  input="$HY_WORLD_INPUT_IMAGE"
  out="${NPA_SMOKE_OUTPUT_DIR:-$result}"
  prompt="$(decode_prompt)"
  [[ -f "$input" ]] || { echo "npa-hy-world: input image is absent" >&2; exit "$EX_CONFIG"; }
  mkdir -p "$scene" "$result" "$out/reports"
  local upstream_home upstream_hub
  upstream_home="$(cache_root)/upstream-home"
  upstream_hub="$upstream_home/.cache/huggingface/hub"
  mkdir -p "$(dirname "$upstream_hub")"
  if [[ -e "$upstream_hub" && ! -L "$upstream_hub" ]]; then
    echo "npa-hy-world: upstream Hugging Face compatibility path is not an owned symlink" >&2
    exit "$EX_SOFTWARE"
  fi
  ln -sfn "$HF_HOME/hub" "$upstream_hub"
  export HOME="$upstream_home"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  "$python" "$tree/hyworld2/panogen/pipeline_with_qwen_image.py" \
    --image "$input" --prompt "$prompt" --seed "${HY_WORLD_SEED:-42}" --save "$scene/panorama.png"
  (
    cd "$worldgen"
    "$python" traj_generate.py --target_path "$scene" \
      --llm_addr "$NPA_HY_WORLD_LLM_ADDR" --llm_port "${NPA_HY_WORLD_LLM_PORT:-8000}" \
      --llm_name "${NPA_HY_WORLD_LLM_NAME:-$QWEN_VLM_MODEL}" \
      --apply_nav_traj --apply_up_route --apply_recon_iteration --force_vlm
    "$tree/.venv/bin/torchrun" --nproc_per_node=8 traj_render.py --target_path "$scene" \
      --llm_addr "$NPA_HY_WORLD_LLM_ADDR" --llm_port "${NPA_HY_WORLD_LLM_PORT:-8000}" \
      --llm_name "${NPA_HY_WORLD_LLM_NAME:-$QWEN_VLM_MODEL}"
    "$tree/.venv/bin/torchrun" --nproc_per_node=8 video_gen.py --target_path "$scene" --fsdp
    "$tree/.venv/bin/torchrun" --nproc_per_node=8 gen_gs_data.py --root_path "$scene" --save_normal --split_sky
    "$python" -m world_gs_trainer default --data_dir "$scene/gs_data" --result_dir "$result" \
      --max_steps 1500 --save_steps 1500 --eval_steps 1500 --ply_steps 1500 --save_ply --convert_to_spz \
      --render_traj_path interp --use_scale_regularization --antialiased --depth_loss --normal_loss \
      --sky_depth_from_pcd --use_mask_gaussian --mask_export_stochastic --no-mask-export-anchor-protection \
      --use_anchor_protection --export_mesh --strategy.refine-start-iter 150 \
      --strategy.refine-stop-iter 750 --strategy.refine-every 100 \
      --strategy.refine-scale2d-stop-iter 750 --strategy.reset-every 99990 \
      --strategy.grow-grad2d 0.0001 --strategy.prune-scale3d 0.1
  )
  write_runtime_metadata "$out/hy_world_runtime_metadata.json" "$HY_WORLD_CONTAINER_IMAGE"
  /opt/npa/hy-world/validate_scene.py --scene-dir "$scene" --result-dir "$result" --input-image "$input" \
    --runtime-metadata "$out/hy_world_runtime_metadata.json" --artifact "$out/hy_world_image_to_world.json"
  local render
  render="$(find "$result/videos" -type f -name '*.mp4' -print -quit)"
  [[ -n "$render" ]] || { echo "npa-hy-world: validator passed without a render" >&2; exit "$EX_SOFTWARE"; }
  "$report" /opt/npa/hy-world/hy_world_report.py --evidence "$out/hy_world_image_to_world.json" \
    --video "$render" --rrd "$out/reports/hy_world_scene.rrd" \
    --manifest "$out/reports/hy_world_scene_rrd_manifest.json" \
    --run-id "${NPA_BYOF_RUN_ID:?NPA_BYOF_RUN_ID is required for a run-scoped Rerun recording}"
}

main() {
  case "${1:-}" in
    health) health ;;
    status) status ;;
    terms) terms ;;
    bootstrap-integrity) bootstrap_integrity ;;
    ensure) ensure ;;
    fetch-models) ensure; fetch_models ;;
    run-image-to-world) run_image_to_world ;;
    *)
      echo "usage: hy-world-runtime {health|status|terms|bootstrap-integrity|ensure|fetch-models|run-image-to-world}" >&2
      exit 2
      ;;
  esac
}

if [[ "${NPA_HY_WORLD_RUNTIME_LIBRARY:-0}" != 1 ]]; then
  main "$@"
fi
