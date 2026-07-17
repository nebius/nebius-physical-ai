#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

REGISTRY=""
PUSH=0
VERSION=""
ALL_VERSIONS=0

usage() {
  cat <<'EOF'
Usage: build.sh [--registry REGISTRY] [--push] [--version VERSION | --all-versions]

Builds the LeRobot container image as npa-lerobot:<version>.
Default VERSION is the [tool.npa.supported-tools].lerobot pin (0.5.1).
Supported versions: 0.5.1 (default) and 0.6.0.
When --registry is provided, also tags REGISTRY/npa-lerobot:<version>.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --registry)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --registry requires a value" >&2
        exit 2
      fi
      REGISTRY="${2%/}"
      shift 2
      ;;
    --push)
      PUSH=1
      shift
      ;;
    --version)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --version requires a value" >&2
        exit 2
      fi
      VERSION="$2"
      shift 2
      ;;
    --all-versions)
      ALL_VERSIONS=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ "$PUSH" -eq 1 ] && [ -z "$REGISTRY" ]; then
  echo "ERROR: --push requires --registry" >&2
  exit 2
fi

if [ "$ALL_VERSIONS" -eq 1 ] && [ -n "$VERSION" ]; then
  echo "ERROR: --version and --all-versions are mutually exclusive" >&2
  exit 2
fi

default_version() {
  cd "$NPA_ROOT"
  python3 - <<'PY'
from pathlib import Path
import re

try:
    import tomllib
except ModuleNotFoundError:
    text = Path("pyproject.toml").read_text()
    section = text.split("[tool.npa.supported-tools]", 1)[1]
    match = re.search(r'^lerobot\s*=\s*"([^"]+)"', section, re.MULTILINE)
    if not match:
        raise SystemExit("Could not find [tool.npa.supported-tools].lerobot")
    print(match.group(1))
else:
    with Path("pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    print(data["tool"]["npa"]["supported-tools"]["lerobot"])
PY
}

supported_versions() {
  cd "$NPA_ROOT"
  python3 - <<'PY'
import json
from pathlib import Path

manifest = Path("src/npa/deploy/lerobot_version_manifest.json")
data = json.loads(manifest.read_text())
print("\n".join(data["supported_versions"]))
PY
}

build_one() {
  local version="$1"
  local local_image="npa-lerobot:${version}"
  local build_args=(
    -f "$SCRIPT_DIR/Dockerfile"
    --build-arg "LEROBOT_VERSION=${version}"
    -t "$local_image"
  )
  local registry_image=""

  if [ -n "$REGISTRY" ]; then
    registry_image="${REGISTRY}/npa-lerobot:${version}"
    build_args+=(-t "$registry_image")
  fi

  docker build "${build_args[@]}" "$NPA_ROOT"

  local size_bytes
  size_bytes="$(docker image inspect "$local_image" --format '{{.Size}}')"
  if command -v numfmt >/dev/null 2>&1; then
    echo "Built: $local_image ($(numfmt --to=iec-i --suffix=B "$size_bytes"))"
  else
    echo "Built: $local_image (${size_bytes} bytes)"
  fi
  if [ -n "$registry_image" ]; then
    echo "Tagged: $registry_image"
  fi
  if [ "$PUSH" -eq 1 ]; then
    docker push "$registry_image"
  fi
}

if [ "$ALL_VERSIONS" -eq 1 ]; then
  while IFS= read -r version; do
    [ -n "$version" ] || continue
    build_one "$version"
  done < <(supported_versions)
else
  if [ -z "$VERSION" ]; then
    VERSION="$(default_version)"
  fi
  case "$VERSION" in
    0.5.1|0.6.0) ;;
    *)
      echo "ERROR: unsupported LeRobot version: $VERSION (supported: 0.5.1, 0.6.0)" >&2
      exit 2
      ;;
  esac
  build_one "$VERSION"
fi
