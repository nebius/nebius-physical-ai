#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
revision="$(git -C "${repo_root}" rev-parse HEAD)"
if [[ ! "${revision}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "expected a full Git revision" >&2
  exit 1
fi
registry="${NPA_BYOF_ROBOMIMIC_REGISTRY:-local.invalid}"
image="${registry}/npa-robomimic:dev-${revision}"

context="$(mktemp -d "${TMPDIR:-/tmp}/npa-robomimic-context.XXXXXXXX")"
trap 'rm -rf -- "${context}"' EXIT
# Build only the exact committed image context named by the immutable tag.
git -C "${repo_root}" archive "${revision}:npa/docker/workbench/robomimic" \
  | tar -x --same-permissions -C "${context}"
docker build --pull=false --tag "${image}" "${context}"
echo "built local candidate ${image}; this helper does not push or publish"
