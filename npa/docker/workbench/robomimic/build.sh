#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
revision="$(git -C "${repo_root}" rev-parse HEAD)"
if [[ ! "${revision}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "expected a full Git revision" >&2
  exit 1
fi
registry="${NPA_PUBLIC_REGISTRY:-ghcr.io/nebius/nebius-physical-ai}"
image="${registry}/npa-robomimic:dev-${revision}"

docker build --pull=false --tag "${image}" "${repo_root}/npa/docker/workbench/robomimic"
echo "built local candidate ${image}; this helper does not push or publish"
