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
"$NPA_PYTHON" - "${SCRIPT_DIR}/runtime-lock.json" "${SCRIPT_DIR}/apt-packages.lock" <<'PY'
import json
from pathlib import Path
import sys

runtime = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
apt_lock = Path(sys.argv[2]).read_text(encoding="utf-8")
if runtime.get("status") != "complete" or "INCOMPLETE" in apt_lock:
    raise SystemExit("RoboTwin Phase A build refused: immutable base/apt/runtime locks are incomplete")
PY

# Unreachable in Phase A. A later approved transaction must replace the locks
# and unresolved Dockerfile base before this standard additive build may run.
docker buildx build --platform linux/amd64 --load \
  --build-arg "SOURCE_SHA=$SOURCE_SHA" --tag "$IMAGE" \
  --file "${SCRIPT_DIR}/Dockerfile" "$NPA_ROOT"
