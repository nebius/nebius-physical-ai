#!/usr/bin/env bash
# Assemble an exact committed source tree and build locally. Publication is a
# separate operation after the parent's exact-byte/security/functional gates.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="$(cd "${NPA_ROOT}/.." && pwd)"
SOURCE_SHA="${SOURCE_SHA:-}"
IMAGE=""
OCI_OUTPUT=""
METADATA_FILE=""
BUILDER=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-sha) SOURCE_SHA="${2:?}"; shift 2 ;;
    --image) IMAGE="${2:?}"; shift 2 ;;
    --oci-output) OCI_OUTPUT="${2:?}"; shift 2 ;;
    --metadata-file) METADATA_FILE="${2:?}"; shift 2 ;;
    --builder) BUILDER="${2:?}"; shift 2 ;;
    -h|--help)
      echo 'Usage: build.sh --source-sha FULL_COMMITTED_SHA [--image REGISTRY/npa-ncore:dev-FULL_COMMITTED_SHA] [--oci-output PATH --metadata-file PATH --builder NAME]'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'A full committed SOURCE_SHA is required' >&2; exit 2; }
[[ "$(git -C "$REPO_ROOT" rev-parse "${SOURCE_SHA}^{commit}")" == "$SOURCE_SHA" ]]
IMAGE="${IMAGE:-local.invalid/npa-ncore:dev-${SOURCE_SHA}}"
[[ "$IMAGE" == */npa-ncore:dev-"${SOURCE_SHA}" ]] || { echo 'Use npa-ncore:dev-<full-source-sha>' >&2; exit 2; }
NPA_PYTHON="${NPA_ROOT}/.venv/bin/python"
[[ -x "$NPA_PYTHON" ]] || { echo 'npa/.venv/bin/python is required' >&2; exit 1; }
# Export only committed files; unrelated dirty paths never enter the build.
context="$(mktemp -d "${TMPDIR:-/tmp}/npa-ncore-context.XXXXXXXX")"
trap 'rm -rf -- "$context"' EXIT
# Preserve committed public file modes even when the private evidence runner
# uses umask 077; the surrounding temporary directory remains owner-only.
git -C "$REPO_ROOT" archive "$SOURCE_SHA" npa/src/npa npa/docker/workbench/ncore | tar -x --same-permissions -C "$context"
options=(--load --provenance=false)
if [[ -n "$OCI_OUTPUT" ]]; then
  [[ "$OCI_OUTPUT" != *,* ]] || { echo 'OCI output path cannot contain commas' >&2; exit 2; }
  [[ ! -e "$OCI_OUTPUT" ]] || { echo 'Refusing to overwrite an existing OCI artifact' >&2; exit 2; }
  options=(--output "type=oci,dest=$OCI_OUTPUT" --provenance=mode=max --sbom=true)
fi
[[ -z "$METADATA_FILE" ]] || options+=(--metadata-file "$METADATA_FILE")
[[ -z "$BUILDER" ]] || options+=(--builder "$BUILDER")
docker buildx build --platform linux/amd64 "${options[@]}" \
  --build-arg "SOURCE_SHA=$SOURCE_SHA" --tag "$IMAGE" \
  --file "$context/npa/docker/workbench/ncore/Dockerfile" "$context/npa"
echo "Built locally: $IMAGE (not published; artifact gates and real conversion remain required)"
