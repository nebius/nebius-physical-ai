#!/usr/bin/env bash
# Build a private derivative for licensed asset camera compatibility only.
set -euo pipefail
umask 077

image=${1:?usage: build-private.sh PRIVATE_IMAGE LOCAL_BASE_TAG EXPECTED_BASE_IMAGE_ID OCI_ARCHIVE METADATA_JSON SCAN_JSON}
base_image=${2:?usage: build-private.sh PRIVATE_IMAGE LOCAL_BASE_TAG EXPECTED_BASE_IMAGE_ID OCI_ARCHIVE METADATA_JSON SCAN_JSON}
expected_base_image_id=${3:?usage: build-private.sh PRIVATE_IMAGE LOCAL_BASE_TAG EXPECTED_BASE_IMAGE_ID OCI_ARCHIVE METADATA_JSON SCAN_JSON}
oci_archive=${4:?usage: build-private.sh PRIVATE_IMAGE LOCAL_BASE_TAG EXPECTED_BASE_IMAGE_ID OCI_ARCHIVE METADATA_JSON SCAN_JSON}
metadata=${5:?usage: build-private.sh PRIVATE_IMAGE LOCAL_BASE_TAG EXPECTED_BASE_IMAGE_ID OCI_ARCHIVE METADATA_JSON SCAN_JSON}
scan=${6:?usage: build-private.sh PRIVATE_IMAGE LOCAL_BASE_TAG EXPECTED_BASE_IMAGE_ID OCI_ARCHIVE METADATA_JSON SCAN_JSON}

case "$image" in
  ghcr.io/nebius/nebius-physical-ai/*|docker.io/*|index.docker.io/*)
    printf '%s\n' 'refusing public image target for private licensed-assets qualification' >&2
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
  npa/docker/workbench/libero-plus-assets/Dockerfile.private
  npa/docker/workbench/libero-plus-assets/THIRD_PARTY_NOTICES.md
  npa/docker/workbench/libero-plus-assets/build-private.sh
  npa/docker/workbench/libero-plus-assets/entrypoint.sh
  npa/docker/workbench/libero-plus-assets/native-executor-provenance.json
  npa/src/npa/workflows/libero_plus_assets.py
)
git -C "$repo_root" diff --quiet -- "${inputs[@]}"
git -C "$repo_root" diff --cached --quiet -- "${inputs[@]}"

base_metadata=$(docker image inspect "$base_image")
NPA_BASE_IMAGE_METADATA="$base_metadata" \
NPA_EXPECTED_BASE_IMAGE_ID="$expected_base_image_id" \
"$repo_root/npa/.venv/bin/python" - <<'PY'
import json
import os

image = json.loads(os.environ["NPA_BASE_IMAGE_METADATA"])[0]
labels = image.get("Config", {}).get("Labels", {}) or {}
if image.get("Id") != os.environ["NPA_EXPECTED_BASE_IMAGE_ID"]:
    raise SystemExit("local OpenWAM base image ID does not match the supplied immutable identity")
if labels.get("npa.tool") != "openwam":
    raise SystemExit("base image is not the inspected OpenWAM runtime")
if labels.get("org.nebius.npa.redistribution") != "unvalidated-operator-private":
    raise SystemExit("base image is not marked operator-private")
PY
docker run --rm --entrypoint /bin/bash "$base_image" -c '\
  set -euo pipefail; \
  test "$(sha256sum /opt/openwam-libero-source/LICENSE | awk "{print \$1}")" = e2885fd30a08381b799c4a33385522b23d637b4051b8f9a7f9f2519944b68ff6; \
  /opt/openwam-libero/bin/python -c "import libero, mujoco; assert mujoco.__version__ == \"3.3.2\""'

# The scanner examines every OCI layer. Create a task-owned flattened base so
# bytes removed for this camera-only profile cannot survive in a parent layer.
flat_suffix=${source_sha:0:12}
flat_container="npa-libero-plus-assets-flat-$flat_suffix"
flat_image="npa-libero-plus-assets-flat:$flat_suffix"
if docker container inspect "$flat_container" >/dev/null 2>&1 || \
  docker image inspect "$flat_image" >/dev/null 2>&1; then
  printf '%s\n' 'refusing to reuse an existing flattened private base' >&2
  exit 64
fi
cleanup_flat_base() {
  docker container rm -f "$flat_container" >/dev/null 2>&1 || true
  docker image rm "$flat_image" >/dev/null 2>&1 || true
}
trap cleanup_flat_base EXIT
docker create --name "$flat_container" --user root --entrypoint /bin/bash \
  "$base_image" -c '\
    set -euo pipefail; \
    rm -f \
      /opt/openwam-venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2 \
      /opt/openwam-libero/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2; \
    test ! -e /opt/openwam-venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2; \
    test ! -e /opt/openwam-libero/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2' \
  >/dev/null
docker start -a "$flat_container" >/dev/null
docker export "$flat_container" | docker import - "$flat_image" >/dev/null
docker rm "$flat_container" >/dev/null

docker buildx build \
  --platform linux/amd64 \
  --provenance=mode=max \
  --sbom=true \
  --build-arg "BASE_IMAGE=$flat_image" \
  --build-arg "NPA_SOURCE_SHA=$source_sha" \
  --build-arg "SOURCE_DATE_EPOCH=$source_epoch" \
  --label "org.opencontainers.image.revision=$source_sha" \
  --file "$repo_root/npa/docker/workbench/libero-plus-assets/Dockerfile.private" \
  --output "type=oci,dest=$oci_archive" \
  --tag "$image" \
  "$repo_root/npa"

test -s "$oci_archive"
"$repo_root/npa/.venv/bin/python" "$repo_root/npa/scripts/scan_image_omniverse_payload.py" \
  --tarball "$oci_archive" --json "$scan"
# The scanner has consumed the flattened base. Releasing only this exact
# task-owned intermediate before the daemon import avoids a second full archive
# write on a VDI with deliberately shared Docker storage.
cleanup_flat_base
skopeo copy --override-os linux --override-arch amd64 \
  "oci-archive:$oci_archive" "docker-daemon:$image" >/dev/null
docker image inspect "$image" >/dev/null
docker run --rm --entrypoint /bin/bash "$image" -c '\
  set -euo pipefail; \
  /opt/openwam-libero/bin/python /opt/npa/src/npa/workflows/libero_plus_assets.py --help >/dev/null; \
  test ! -e /workspace/.cache/npa-model/libero-plus-assets; \
  test ! -e /opt/npa/libero-plus-assets.zip; \
  test -s /usr/share/doc/npa-libero-plus-assets/THIRD_PARTY_NOTICES.md; \
  test -s /usr/share/doc/npa-libero-plus-assets/native-executor-provenance.json; \
  test -z "$(find /etc/ssh -maxdepth 1 -type f -name "ssh_host_*_key" -print -quit)"'
docker run --rm "$image" /bin/sh -c '\
  set -euo pipefail; \
  test "$(id -u)" -eq 1000; \
  test -w /tmp; test -w "$HOME"; \
  command -v sudo; sudo -n true; command -v sshd; command -v rsync; command -v service; \
  test -n "$(find /etc/ssh -maxdepth 1 -type f -name "ssh_host_*_key" -print -quit)"; \
  test "$(/bin/sh -c "printf %s forwarded" sentinel)" = forwarded'

NPA_ASSETS_IMAGE="$image" \
NPA_ASSETS_BASE_IMAGE="$base_image" \
NPA_ASSETS_BASE_IMAGE_ID="$expected_base_image_id" \
NPA_ASSETS_OCI="$oci_archive" \
NPA_ASSETS_SCAN="$scan" \
NPA_ASSETS_SHA="$source_sha" \
"$repo_root/npa/.venv/bin/python" - "$metadata" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

image = os.environ["NPA_ASSETS_IMAGE"]
inspection = json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True))[0]
payload = {
    "schema": "npa.libero-plus.licensed-assets.private-image.v1",
    "image": image,
    "image_id": inspection["Id"],
    "base_image": os.environ["NPA_ASSETS_BASE_IMAGE"],
    "base_image_id": os.environ["NPA_ASSETS_BASE_IMAGE_ID"],
    "npa_source_revision": os.environ["NPA_ASSETS_SHA"],
    "oci_archive_sha256": hashlib.sha256(Path(os.environ["NPA_ASSETS_OCI"]).read_bytes()).hexdigest(),
    "payload": {
        "libero_plus_source": "absent",
        "libero_plus_asset_archive": "runtime-fetch-only",
        "native_executor": "original-mit-libero",
        "weights": "absent",
    },
    "redistribution": "unvalidated-operator-private",
    "scan": os.environ["NPA_ASSETS_SCAN"],
    "status": "built-local-not-pushed",
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

printf '%s\n' 'Private licensed-assets camera image built locally; it is not public or a LIBERO-Plus benchmark image.'
