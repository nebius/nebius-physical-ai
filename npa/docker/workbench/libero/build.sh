#!/usr/bin/env bash
set -euo pipefail

image=${1:?usage: build.sh ghcr.io/nebius/nebius-physical-ai/npa-libero:dev-<full-sha>}
source_sha=${NPA_SOURCE_SHA:-$(git rev-parse HEAD)}
[[ "$source_sha" =~ ^[0-9a-f]{40}$ ]]
[[ "$image" == "ghcr.io/nebius/nebius-physical-ai/npa-libero:dev-$source_sha" ]]
image_inputs=(
  npa/docker/workbench/libero
)
git diff --quiet -- "${image_inputs[@]}"
git diff --cached --quiet -- "${image_inputs[@]}"
source_epoch=$(git show -s --format=%ct "$source_sha")
[[ "$source_epoch" =~ ^[1-9][0-9]*$ ]]

metadata=${NPA_LIBERO_BUILD_METADATA:?NPA_LIBERO_BUILD_METADATA is required}
manager_public_key=${NPA_LIBERO_MANAGER_ACCEPTANCE_PUBLIC_KEY_B64:?NPA_LIBERO_MANAGER_ACCEPTANCE_PUBLIC_KEY_B64 is required}
test "$(printf '%s' "$manager_public_key" | base64 -d | wc -c)" = 32
case "$metadata" in /*) ;; *) printf '%s\n' 'build metadata path must be absolute' >&2; exit 64 ;; esac
umask 077
mkdir -p "$(dirname "$metadata")"
BUILDX_METADATA_PROVENANCE=max docker buildx build \
  --load \
  --provenance=mode=max \
  --sbom=true \
  --metadata-file "$metadata" \
  --secret id=npa_libero_manager_acceptance_public_key_b64,env=NPA_LIBERO_MANAGER_ACCEPTANCE_PUBLIC_KEY_B64 \
  --build-arg "NPA_SOURCE_SHA=$source_sha" \
  --build-arg "SOURCE_DATE_EPOCH=$source_epoch" \
  --label "org.opencontainers.image.revision=$source_sha" \
  --file npa/docker/workbench/libero/Dockerfile \
  --tag "$image" \
  npa

printf '%s\n' 'LIBERO neutral candidate built locally; publication and validation remain quarantined.'
