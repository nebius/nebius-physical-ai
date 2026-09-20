#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="$(cd "${NPA_ROOT}/.." && pwd)"
SOURCE_SHA="${SOURCE_SHA:-}"
IMAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-sha) SOURCE_SHA="${2:?}"; shift 2 ;;
    --image) IMAGE="${2:?}"; shift 2 ;;
    -h|--help)
      echo 'Usage: build.sh --source-sha FULL_COMMITTED_SHA --image REGISTRY/npa-robotwin:dev-FULL_COMMITTED_SHA'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'A full committed SOURCE_SHA is required' >&2; exit 2; }
[[ "$IMAGE" == */npa-robotwin:dev-"${SOURCE_SHA}" ]] || { echo 'Use npa-robotwin:dev-<full-source-sha>' >&2; exit 2; }
NPA_PYTHON="${NPA_ROOT}/.venv/bin/python"
[[ -x "$NPA_PYTHON" ]] || { echo 'npa/.venv/bin/python is required' >&2; exit 1; }
[[ "$(git -C "$REPO_ROOT" rev-parse "${SOURCE_SHA}^{commit}")" == "$SOURCE_SHA" ]]

# Validate only bytes exported from the named commit. Unrelated or overlapping
# worktree edits can never silently enter an exact-SHA development candidate.
context="$(mktemp -d "${TMPDIR:-/tmp}/npa-robotwin-context.XXXXXXXX")"
trap 'rm -rf -- "$context"' EXIT
git -C "$REPO_ROOT" archive "$SOURCE_SHA" \
  npa/docker/workbench/robotwin \
  | tar -x --same-permissions -C "$context"
context_root="$context/npa"
context_image_root="$context_root/docker/workbench/robotwin"

"$NPA_PYTHON" - \
  "$context_image_root/runtime-lock.json" \
  "$context_image_root/apt-packages.lock" \
  "$context_image_root/runtime-requirements.lock" \
  "$context_image_root/Dockerfile" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

runtime = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
apt_path = Path(sys.argv[2])
requirements_path = Path(sys.argv[3])
dockerfile = Path(sys.argv[4]).read_text(encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


bootstrap = runtime.get("bootstrap")
delivery = runtime.get("runtime_delivery")
if (
    runtime.get("status") != "bootstrap-complete-runtime-disabled"
    or not isinstance(bootstrap, dict)
    or bootstrap.get("status") != "complete"
    or not isinstance(delivery, dict)
    or not str(delivery.get("status", "")).startswith("disabled-")
):
    raise SystemExit("RoboTwin neutral build refused: bootstrap/runtime boundary is invalid")

apt = bootstrap.get("apt")
python_runtime = bootstrap.get("python_runtime")
if not isinstance(apt, dict) or not isinstance(python_runtime, dict):
    raise SystemExit("RoboTwin neutral build refused: bootstrap lock is invalid")
if apt.get("apt_lock_sha256") != digest(apt_path):
    raise SystemExit("RoboTwin neutral build refused: apt lock digest mismatch")
if python_runtime.get("requirements_lock_sha256") != digest(requirements_path):
    raise SystemExit("RoboTwin neutral build refused: requirements lock digest mismatch")

apt_lines = [
    line.split("\t")
    for line in apt_path.read_text(encoding="utf-8").splitlines()
    if line and not line.startswith("#")
]
binaries = [line for line in apt_lines if line[0] == "binary"]
sources = [line for line in apt_lines if line[0] == "source"]
if len(binaries) != apt.get("binary_package_count") or len(sources) != apt.get(
    "source_package_count"
):
    raise SystemExit("RoboTwin neutral build refused: apt closure count mismatch")
if any(
    len(line) not in {12, 13}
    or (len(line) == 13 and line[12] != "gitleaks:allow=public-ubuntu-copyright-sha256")
    or len(line[4]) != 64
    or not set(line[4]) <= set("0123456789abcdef")
    or len(line[10]) != 64
    or not set(line[10]) <= set("0123456789abcdef")
    for line in binaries
):
    raise SystemExit("RoboTwin neutral build refused: apt binary identity is invalid")

requirements = requirements_path.read_text(encoding="utf-8")
if "status=complete-empty" not in requirements or "artifact-count=0" not in requirements:
    raise SystemExit("RoboTwin neutral build refused: requirements lock is not empty/complete")
base = bootstrap.get("base")
manifest = base.get("manifest_digest") if isinstance(base, dict) else None
if not isinstance(manifest, str) or f"FROM ubuntu:22.04@{manifest}" not in dockerfile:
    raise SystemExit("RoboTwin neutral build refused: Dockerfile/base lock mismatch")


PY

# A local build supplies bytes for inspection; every publication gate runs before push.
docker buildx build --platform linux/amd64 --load --provenance=mode=max \
  --label "org.opencontainers.image.revision=$SOURCE_SHA" \
  --label "org.opencontainers.image.source=https://github.com/nebius/nebius-physical-ai" \
  --build-arg "NPA_SOURCE_SHA=$SOURCE_SHA" --tag "$IMAGE" \
  --file "$context_image_root/Dockerfile" "$context_root"
