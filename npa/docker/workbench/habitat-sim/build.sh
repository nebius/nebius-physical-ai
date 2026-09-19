#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -ne 1 || "$1" == -* ]]; then
  echo "usage: build.sh OWNER_ONLY_OUTPUT.oci.tar" >&2
  exit 2
fi
caller_cwd="$(pwd -P)"
case "$1" in
  /*) output_candidate="$1" ;;
  *) output_candidate="$caller_cwd/$1" ;;
esac
output_parent_candidate="$(dirname -- "$output_candidate")"
output_name="$(basename -- "$output_candidate")"
if [[ ! -d "$output_parent_candidate" ]]; then
  echo "OCI output parent directory does not exist" >&2
  exit 2
fi
output_parent="$(cd "$output_parent_candidate" && pwd -P)"
if [[ "$output_name" == . || "$output_name" == .. || "$output_name" == *,* ]]; then
  echo "invalid OCI output name" >&2
  exit 2
fi
# Hold the directory inode, so replacing an ancestor cannot redirect output.
exec {output_directory_fd}< "$output_parent"
output_directory="/proc/$$/fd/$output_directory_fd"
if [[ "$(stat -Lc %u "$output_directory")" != "$(id -u)" ]] ||
   (( (8#$(stat -Lc %a "$output_directory") & 0022) != 0 )); then
  echo "OCI output parent must be owned and not group/world writable" >&2
  exit 2
fi
output="$output_directory/$output_name"
if [[ -e "$output" || -L "$output" ]]; then
  echo "refusing to overwrite OCI output" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
if [[ "$script_dir" != "$repo_root/npa/docker/workbench/habitat-sim" ]]; then
  echo "build script is not in the expected repository path" >&2
  exit 2
fi
if ! git -C "$repo_root" diff --quiet --ignore-submodules --; then
  echo "refusing a dirty repository" >&2
  exit 2
fi
if ! git -C "$repo_root" diff --cached --quiet --ignore-submodules --; then
  echo "refusing staged repository changes" >&2
  exit 2
fi
if [[ -n "$(git -C "$repo_root" ls-files --others --exclude-standard)" ]]; then
  echo "refusing untracked repository files" >&2
  exit 2
fi

source_sha="$(git -C "$repo_root" rev-parse --verify HEAD^{commit})"
if [[ ! "$source_sha" =~ ^[0-9a-f]{40}$ ]]; then
  echo "A full Git SHA is required" >&2
  exit 2
fi
readonly build_platform="linux/amd64"
if [[ "$build_platform" != "linux/amd64" ]]; then
  echo "refusing non-canonical Habitat build platform" >&2
  exit 2
fi

# Every repository-owned file copied by the Dockerfile is projected from the
# committed object database. Keep this list identical to verify_image.py.
readonly source_paths=(
  docker/workbench/habitat-sim/Dockerfile
  docker/workbench/habitat-sim/REDISTRIBUTION.md
  docker/workbench/habitat-sim/THIRD_PARTY_NOTICES.md
  docker/workbench/habitat-sim/apt-build.lock
  docker/workbench/habitat-sim/apt-runtime.lock
  docker/workbench/habitat-sim/build.sh
  docker/workbench/habitat-sim/entrypoint.sh
  docker/workbench/habitat-sim/licenses.json
  docker/workbench/habitat-sim/prepare_source.py
  docker/workbench/habitat-sim/requirements-build.lock
  docker/workbench/habitat-sim/requirements-runtime.lock
  docker/workbench/habitat-sim/runtime-payload.json
  docker/workbench/habitat-sim/source-manifest.json
  docker/workbench/habitat-sim/verify_apt_artifacts.sh
  docker/workbench/habitat-sim/verify_apt_source.py
  docker/workbench/habitat-sim/verify_image.py
  docker/workbench/packaging-contract.yaml
  scripts/image_byte_scan/core.py
  scripts/image_byte_scan/habitat_sim_verification.py
  src/npa/__init__.py
  src/npa/workflows/__init__.py
  src/npa/workflows/habitat_sim_smoke.py
)

projection="$(mktemp -d "${TMPDIR:-/tmp}/npa-habitat-source.XXXXXX")"
chmod 0700 "$projection"
output_temporary=
cleanup_warning() {
  local message="$1"
  printf '%s\n' "$message" >&2 || true
}
cleanup() {
  local original_status=$?
  if ! rm -rf -- "$projection"; then
    cleanup_warning "warning: source projection cleanup failed"
  fi
  if [[ -n "$output_temporary" ]]; then
    if ! rm -f -- "$output_temporary/candidate.oci.tar"; then
      cleanup_warning "warning: temporary OCI file cleanup failed"
    fi
    if ! rmdir -- "$output_temporary"; then
      cleanup_warning "warning: temporary OCI directory cleanup failed"
    fi
  fi
  return "$original_status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
manifest="$projection/npa-source-manifest.sha256"
: > "$manifest.unsorted"
for context_path in "${source_paths[@]}"; do
  repository_path="npa/$context_path"
  destination="$projection/inputs/$context_path"
  mkdir -p "$(dirname "$destination")"
  if ! git -C "$repo_root" cat-file blob "$source_sha:$repository_path" > "$destination"; then
    echo "missing committed source input: $repository_path" >&2
    exit 2
  fi
  if ! cmp -s "$repo_root/$repository_path" "$destination"; then
    echo "working source differs from committed object: $repository_path" >&2
    exit 2
  fi
  printf '%s  %s\n' "$(sha256sum "$destination" | cut -d' ' -f1)" \
    "inputs/$context_path" >> "$manifest.unsorted"
done
LC_ALL=C sort -k2,2 "$manifest.unsorted" > "$manifest"
rm -f "$manifest.unsorted"
manifest_sha256="$(sha256sum "$manifest" | cut -d' ' -f1)"
if [[ ! "$manifest_sha256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "source manifest digest is invalid" >&2
  exit 2
fi

cd "$repo_root"
if [[ -e "$output" || -L "$output" ]]; then
  echo "refusing to overwrite OCI output" >&2
  exit 2
fi
output_temporary="$(mktemp -d "$output_directory/.npa-habitat-oci.XXXXXX")"
: > "$output_temporary/candidate.oci.tar"
docker buildx build \
  --platform="$build_platform" \
  --build-arg "NPA_SOURCE_SHA=$source_sha" \
  --build-arg "NPA_SOURCE_MANIFEST_SHA256=$manifest_sha256" \
  --build-context "npa-source-provenance=$projection" \
  --file npa/docker/workbench/habitat-sim/Dockerfile \
  --output "type=oci,dest=$output_temporary/candidate.oci.tar" \
  --provenance=mode=max \
  --sbom=true \
  npa
if [[ ! -f "$output_temporary/candidate.oci.tar" || -L "$output_temporary/candidate.oci.tar" ||
      ! -s "$output_temporary/candidate.oci.tar" ||
      "$(stat -c '%u:%h' "$output_temporary/candidate.oci.tar")" != "$(id -u):1" ]]; then
  echo "invalid OCI build output" >&2
  exit 2
fi
chmod 0600 "$output_temporary/candidate.oci.tar"
# link(2) is atomic and refuses an existing destination, including symlinks.
ln -T -- "$output_temporary/candidate.oci.tar" "$output"
