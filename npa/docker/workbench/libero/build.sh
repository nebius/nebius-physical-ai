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
oci_archive=${NPA_LIBERO_BUILD_OCI_ARCHIVE:?NPA_LIBERO_BUILD_OCI_ARCHIVE is required}
customer_authorization_public_key=${NPA_LIBERO_CUSTOMER_AUTHORIZATION_PUBLIC_KEY_B64:?NPA_LIBERO_CUSTOMER_AUTHORIZATION_PUBLIC_KEY_B64 is required}
output_storage_authorization_public_key=${NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY_B64:?NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY_B64 is required}
test "$(printf '%s' "$customer_authorization_public_key" | base64 -d | wc -c)" = 32
test "$(printf '%s' "$output_storage_authorization_public_key" | base64 -d | wc -c)" = 32
test "$customer_authorization_public_key" != "$output_storage_authorization_public_key"
case "$metadata" in /*) ;; *) printf '%s\n' 'build metadata path must be absolute' >&2; exit 64 ;; esac
case "$oci_archive" in /*) ;; *) printf '%s\n' 'build OCI archive path must be absolute' >&2; exit 64 ;; esac
umask 077
mkdir -p "$(dirname "$metadata")"
mkdir -p "$(dirname "$oci_archive")"
command -v skopeo >/dev/null
BUILDX_METADATA_PROVENANCE=max docker buildx build \
  --output "type=oci,dest=$oci_archive,tar=true,rewrite-timestamp=true" \
  --provenance=mode=max \
  --sbom=true \
  --metadata-file "$metadata" \
  --secret id=npa_libero_customer_authorization_public_key_b64,env=NPA_LIBERO_CUSTOMER_AUTHORIZATION_PUBLIC_KEY_B64 \
  --secret id=npa_libero_output_storage_authorization_public_key_b64,env=NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY_B64 \
  --build-arg "NPA_SOURCE_SHA=$source_sha" \
  --build-arg "SOURCE_DATE_EPOCH=$source_epoch" \
  --label "org.opencontainers.image.revision=$source_sha" \
  --file npa/docker/workbench/libero/Dockerfile \
  --tag "$image" \
  npa
test -s "$metadata"
test -s "$oci_archive"
npa/.venv/bin/python npa/scripts/scan_image_libero_payload.py \
  --verify-build-oci "$oci_archive" \
  --build-metadata "$metadata"
skopeo copy --override-os linux --override-arch amd64 \
  "oci-archive:$oci_archive" "docker-daemon:$image" >/dev/null
expected_config_digest="$(npa/.venv/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["containerimage.config.digest"])' \
  "$metadata")"
test "$(docker image inspect --format '{{.Id}}' "$image")" = "$expected_config_digest"

printf '%s\n' 'LIBERO neutral candidate built locally; publication and validation remain quarantined.'
