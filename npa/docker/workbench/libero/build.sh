#!/usr/bin/env bash
set -euo pipefail

image=${1:?usage: build.sh ghcr.io/nebius/nebius-physical-ai/npa-libero:dev-<full-sha>}
source_sha=${NPA_SOURCE_SHA:-$(git rev-parse HEAD)}
[[ "$source_sha" =~ ^[0-9a-f]{40}$ ]]
[[ "$image" == "ghcr.io/nebius/nebius-physical-ai/npa-libero:dev-$source_sha" ]]
test -z "$(git status --short --untracked-files=no)"

metadata=${NPA_LIBERO_BUILD_METADATA:?NPA_LIBERO_BUILD_METADATA is required}
case "$metadata" in /*) ;; *) printf '%s\n' 'build metadata path must be absolute' >&2; exit 64 ;; esac
umask 077
mkdir -p "$(dirname "$metadata")"
BUILDX_METADATA_PROVENANCE=max docker buildx build \
  --load \
  --provenance=mode=max \
  --sbom=true \
  --metadata-file "$metadata" \
  --build-arg "NPA_SOURCE_SHA=$source_sha" \
  --label "org.opencontainers.image.revision=$source_sha" \
  --file npa/docker/workbench/libero/Dockerfile \
  --tag "$image" \
  npa

printf '%s\n' 'LIBERO neutral candidate built locally; publication and validation remain quarantined.'
