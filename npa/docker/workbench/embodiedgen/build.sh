#!/usr/bin/env bash
set -euo pipefail

image=${1:?usage: build.sh <private-registry>/npa-embodiedgen:dev-<full-source-sha>}
source_sha=${NPA_SOURCE_SHA:-$(git rev-parse HEAD)}
private_registry=${NPA_REGISTRY:?NPA_REGISTRY must name the operator-controlled private registry namespace}
registry_host=${private_registry%%/*}
[[ "$source_sha" =~ ^[0-9a-f]{40}$ ]]
[[ "$image" == "${private_registry%/}/npa-embodiedgen:dev-${source_sha}" ]]
[[ "$registry_host" == "localhost" || "$registry_host" == *.* || "$registry_host" == *:* ]] || {
  echo "NPA_REGISTRY must start with a fully-qualified private registry host" >&2
  exit 2
}
test -z "$(git status --porcelain --untracked-files=all -- npa/docker/workbench/embodiedgen)"

PYTHONPATH="$(pwd)/npa/src${PYTHONPATH:+:${PYTHONPATH}}" \
  NPA_EMBODIEDGEN_PRIVATE_REGISTRY="${private_registry%/}" \
  npa/.venv/bin/python - <<'PY'
import os

from npa.deploy.images import is_public_registry

registry = os.environ["NPA_EMBODIEDGEN_PRIVATE_REGISTRY"]
if is_public_registry(registry):
    raise SystemExit(
        "EmbodiedGen is an operator-private runtime-fetch image; NPA_REGISTRY "
        "must not be an anonymous/public registry namespace"
    )
PY

docker buildx build --push \
  --build-arg "NPA_SOURCE_SHA=$source_sha" \
  --label "org.opencontainers.image.revision=$source_sha" \
  --file npa/docker/workbench/embodiedgen/Dockerfile \
  --tag "$image" npa
