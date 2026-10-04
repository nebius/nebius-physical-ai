#!/usr/bin/env bash
# Build a local/private, source-admission-only image. It never pushes.
set -euo pipefail
umask 077

image=${1:?usage: build-private.sh PRIVATE_IMAGE OCI_ARCHIVE METADATA_JSON SCAN_JSON}
oci_archive=${2:?usage: build-private.sh PRIVATE_IMAGE OCI_ARCHIVE METADATA_JSON SCAN_JSON}
metadata=${3:?usage: build-private.sh PRIVATE_IMAGE OCI_ARCHIVE METADATA_JSON SCAN_JSON}
scan=${4:?usage: build-private.sh PRIVATE_IMAGE OCI_ARCHIVE METADATA_JSON SCAN_JSON}

case "$image" in
  ghcr.io/nebius/nebius-physical-ai/*|docker.io/*|index.docker.io/*)
    printf '%s\n' 'refusing public image target for unvalidated admission image' >&2
    exit 64
    ;;
  *:*) ;;
  *)
    printf '%s\n' 'private image target must include an immutable build tag' >&2
    exit 64
    ;;
esac
for destination in "$oci_archive" "$metadata" "$scan"; do
  case "$destination" in /*) ;; *) printf '%s\n' 'output paths must be absolute' >&2; exit 64 ;; esac
  test ! -e "$destination" || { printf '%s\n' "refusing to overwrite $destination" >&2; exit 64; }
  mkdir -p "$(dirname "$destination")"
done

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
source_sha=$(git -C "$repo_root" rev-parse HEAD)
source_epoch=$(git -C "$repo_root" show -s --format=%ct "$source_sha")
[[ "$source_sha" =~ ^[0-9a-f]{40}$ ]]
[[ "$source_epoch" =~ ^[1-9][0-9]*$ ]]

inputs=(
  npa/docker/workbench/libero-plus/Dockerfile.admission
  npa/docker/workbench/libero-plus/THIRD_PARTY_NOTICES.md
  npa/docker/workbench/libero-plus/build-private.sh
  npa/docker/workbench/libero-plus/entrypoint.sh
  npa/docker/workbench/libero/debian-packages.lock
  npa/src/npa/__init__.py
  npa/src/npa/workflows/__init__.py
  npa/src/npa/workflows/libero_plus.py
)
git -C "$repo_root" diff --quiet -- "${inputs[@]}"
git -C "$repo_root" diff --cached --quiet -- "${inputs[@]}"

docker buildx build \
  --platform linux/amd64 \
  --provenance=mode=max \
  --sbom=true \
  --build-arg "NPA_SOURCE_SHA=$source_sha" \
  --build-arg "SOURCE_DATE_EPOCH=$source_epoch" \
  --label "org.opencontainers.image.revision=$source_sha" \
  --file "$repo_root/npa/docker/workbench/libero-plus/Dockerfile.admission" \
  --output "type=oci,dest=$oci_archive" \
  --tag "$image" \
  "$repo_root/npa"

test -s "$oci_archive"
"$repo_root/npa/.venv/bin/python" "$repo_root/npa/scripts/scan_image_omniverse_payload.py" \
  --tarball "$oci_archive" --json "$scan"
docker_archive="$oci_archive.docker.tar"
skopeo copy --override-os linux --override-arch amd64 \
  "oci-archive:$oci_archive" "docker-archive:$docker_archive:$image" >/dev/null
docker load --input "$docker_archive" >/dev/null
rm -- "$docker_archive"
docker image inspect "$image" >/dev/null

NPA_ADMISSION_IMAGE="$image" \
NPA_ADMISSION_OCI="$oci_archive" \
NPA_ADMISSION_SCAN="$scan" \
NPA_ADMISSION_SHA="$source_sha" \
"$repo_root/npa/.venv/bin/python" - "$metadata" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

image = os.environ["NPA_ADMISSION_IMAGE"]
inspection = json.loads(
    subprocess.check_output(["docker", "image", "inspect", image], text=True)
)[0]
payload = {
    "schema": "npa.libero-plus.neutral-admission-image.v1",
    "image": image,
    "image_id": inspection["Id"],
    "npa_source_revision": os.environ["NPA_ADMISSION_SHA"],
    "oci_archive_sha256": hashlib.sha256(Path(os.environ["NPA_ADMISSION_OCI"]).read_bytes()).hexdigest(),
    "payload": {"libero_plus_source": "absent", "asset": "absent", "weights": "absent"},
    "redistribution": "unvalidated-private-qualification",
    "scan": os.environ["NPA_ADMISSION_SCAN"],
    "status": "built-local-not-pushed",
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

printf '%s\n' 'LIBERO-Plus source-admission image built locally; it is not pushed or benchmark-qualified.'
