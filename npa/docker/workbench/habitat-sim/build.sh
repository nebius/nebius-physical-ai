#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || "$1" == -* ]]; then
  echo "usage: build.sh OWNER_ONLY_OUTPUT.oci.tar" >&2
  exit 2
fi
output="$1"
if [[ -e "$output" ]]; then
  echo "refusing to overwrite OCI output" >&2
  exit 2
fi

source_sha="$(git rev-parse --verify HEAD)"
if [[ ! "$source_sha" =~ ^[0-9a-f]{40}$ ]]; then
  echo "A full Git SHA is required" >&2
  exit 2
fi

exec docker buildx build \
  --build-arg "NPA_SOURCE_SHA=$source_sha" \
  --file npa/docker/workbench/habitat-sim/Dockerfile \
  --output "type=oci,dest=$output" \
  --provenance=mode=max \
  --sbom=true \
  npa
