#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

REGISTRY=""
PUSH=0

usage() {
  cat <<'EOF'
Usage: build.sh [--registry REGISTRY] [--push]

Builds the LanceDB Workbench image as npa-lancedb:<version>.
When --registry is provided, also tags REGISTRY/npa-lancedb:<version>.
Use --registry cr.eu-north1.nebius.cloud/<your-registry-id> --push to publish.
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

PYTHON_BIN="${NPA_PYTHON_BIN:-$NPA_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3)"
fi

VERSION="$(
  cd "$NPA_ROOT"
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import re

try:
    import tomllib
except ModuleNotFoundError:
    text = Path("pyproject.toml").read_text()
    section = text.split("[tool.npa.supported-tools]", 1)[1]
    match = re.search(r'^lancedb\s*=\s*"([^"]+)"', section, re.MULTILINE)
    if not match:
        raise SystemExit("Could not find [tool.npa.supported-tools].lancedb")
    print(match.group(1))
else:
    with Path("pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    print(data["tool"]["npa"]["supported-tools"]["lancedb"])
PY
)"

LOCAL_IMAGE="npa-lancedb:${VERSION}"
BUILD_ARGS=(
  --platform linux/amd64
  -f "$SCRIPT_DIR/Dockerfile"
  -t "$LOCAL_IMAGE"
)

if [ -n "$REGISTRY" ]; then
  REGISTRY_IMAGE="${REGISTRY}/npa-lancedb:${VERSION}"
  BUILD_ARGS+=(-t "$REGISTRY_IMAGE")
else
  REGISTRY_IMAGE=""
fi

docker build "${BUILD_ARGS[@]}" "$NPA_ROOT"

SIZE_BYTES="$(docker image inspect "$LOCAL_IMAGE" --format '{{.Size}}')"
if command -v numfmt >/dev/null 2>&1; then
  SIZE="$(numfmt --to=iec-i --suffix=B "$SIZE_BYTES")"
else
  SIZE="${SIZE_BYTES} bytes"
fi

echo "Built: $LOCAL_IMAGE"
if [ -n "$REGISTRY_IMAGE" ]; then
  echo "Tagged: $REGISTRY_IMAGE"
fi
echo "Image size: $SIZE"

if [ "$PUSH" -eq 1 ]; then
  docker push "$REGISTRY_IMAGE"
fi
