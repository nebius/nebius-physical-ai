#!/usr/bin/env bash
set -euo pipefail

image=${1:?usage: build.sh <private-registry>/npa-embodiedgen:dev-<full-source-sha>}
source_sha=${NPA_SOURCE_SHA:-$(git rev-parse HEAD)}
[[ "$source_sha" =~ ^[0-9a-f]{40}$ ]]
[[ "$image" == */npa-embodiedgen:dev-"$source_sha" ]]
test -z "$(git status --porcelain --untracked-files=all -- npa/docker/workbench/embodiedgen)"

docker buildx build --push \
  --build-arg "NPA_SOURCE_SHA=$source_sha" \
  --label "org.opencontainers.image.revision=$source_sha" \
  --file npa/docker/workbench/embodiedgen/Dockerfile \
  --tag "$image" npa
