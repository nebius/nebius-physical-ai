#!/usr/bin/env bash
# Build the neutral HY-World bootstrap from one committed source snapshot.
#
# A local build is useful for recipe review.  --push is deliberately narrower:
# it can deliver only to an operator-asserted private registry, after the same
# payload/secret/vulnerability/license/SBOM gates used by the secure image path.
# It cannot publish to the official public namespace or promote a release.
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="$(cd "${NPA_ROOT}/.." && pwd)"
NPA_PYTHON="${NPA_ROOT}/.venv/bin/python"
REGISTRY=""
TAG=""
PUSH=0
OPERATOR_PRIVATE=0
RECEIPT_PARENT=""
PRIVATE_RECEIPT=""
IMAGE_TAR=""
REMOTE_TAR=""
REMOTE_ARCHIVE_SHA256=""

usage() {
  cat <<'EOF'
Usage: build.sh [--tag dev-<full-source-sha>]
       build.sh --push --operator-private --registry HOST/NAMESPACE \
         --receipt-dir /absolute/private/directory [--tag dev-<full-source-sha>]

Without --push, build a local neutral bootstrap only.  --push requires an
operator-controlled private registry and emits private scan/SBOM/digest receipts.
It always refuses the official public registry and never promotes a release.
EOF
}

fail() {
  printf 'npa-hy-world build: %s\n' "$*" >&2
  exit 2
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command is missing: $1"
}

require_private_directory() {
  local directory="$1" mode
  [[ "$directory" == /* && -d "$directory" && ! -L "$directory" ]] \
    || fail "--receipt-dir must be an existing absolute directory"
  [[ "$(stat -c '%u' "$directory")" == "$(id -u)" ]] \
    || fail "--receipt-dir must be owned by the invoking operator"
  mode=$((8#$(stat -c '%a' "$directory")))
  (( (mode & 0077) == 0 )) \
    || fail "--receipt-dir must not be group/world accessible"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="${2:?}"; shift 2 ;;
    --tag) TAG="${2:?}"; shift 2 ;;
    --push) PUSH=1; shift ;;
    --operator-private) OPERATOR_PRIVATE=1; shift ;;
    --receipt-dir) RECEIPT_PARENT="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown option: $1" ;;
  esac
done

[[ -x "$NPA_PYTHON" ]] || fail "${NPA_PYTHON} is required"
SOURCE_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail "full source SHA required"
[[ -n "$TAG" ]] || TAG="dev-${SOURCE_COMMIT}"
[[ "$TAG" == "dev-${SOURCE_COMMIT}" ]] \
  || fail "tag must be exactly dev-${SOURCE_COMMIT}"

if (( PUSH )); then
  (( OPERATOR_PRIVATE )) \
    || fail "--push requires --operator-private; it is a registry-scope assertion, not terms acceptance"
  [[ -n "$REGISTRY" && "$REGISTRY" == */* && "$REGISTRY" != *[[:space:]]* ]] \
    || fail "--push requires --registry HOST/NAMESPACE"
  [[ "${REGISTRY%/}" != "ghcr.io/nebius/nebius-physical-ai" ]] \
    || fail "official public publication is only allowed through publish-public-images.yml"
  [[ -n "$RECEIPT_PARENT" ]] \
    || fail "--push requires --receipt-dir for private security receipts"
  require_private_directory "$RECEIPT_PARENT"
  require_command crane
  require_command sha256sum
  # Reuse NPA's central public-registry classification and fully-qualified
  # image-reference parser.  --operator-private records intent; it cannot turn
  # Docker Hub shorthand or a configured anonymous namespace into a private
  # destination.
  PYTHONPATH="${NPA_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "$NPA_PYTHON" -m npa.workbench.hy_world.private_delivery \
    private-registry "$REGISTRY" >/dev/null
fi
require_command docker

# Docker must never receive dirty or untracked checkout bytes. Archive only the
# exact source commit, including the workflow source that workflow_build stages
# for the normal npa build context. This works with RELAXED_DIRTY_TREE_MODE:
# unrelated worktree edits neither block nor enter the image.
BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/npa-hy-world-build.XXXXXXXX")"
cleanup() {
  local status=$?
  if [[ -n "$IMAGE_TAR" ]]; then
    rm -f -- "$IMAGE_TAR"
  fi
  if [[ -n "$REMOTE_TAR" ]]; then
    rm -f -- "$REMOTE_TAR"
  fi
  rm -rf -- "$BUILD_ROOT"
  # The image is intentionally retained for an operator to inspect or, after a
  # successful private delivery, remove by exact locally-owned dev tag.
  return "$status"
}
trap cleanup EXIT
git -C "$REPO_ROOT" archive "$SOURCE_COMMIT" -- .trivyignore npa workflows | tar -x -C "$BUILD_ROOT"
BUILD_NPA="$BUILD_ROOT/npa"
"$NPA_PYTHON" "$BUILD_NPA/src/npa/workflow_build.py" \
  --stage-catalog --package-root "$BUILD_NPA"

LOCAL_IMAGE="npa-hy-world:${TAG}"
REMOTE_IMAGE=""
if (( PUSH )); then
  REMOTE_IMAGE="${REGISTRY%/}/npa-hy-world:${TAG}"
  PRIVATE_RECEIPT="${RECEIPT_PARENT%/}/hy-world-${TAG}.json"
  [[ ! -e "$PRIVATE_RECEIPT" && ! -L "$PRIVATE_RECEIPT" ]] \
    || fail "refusing to overwrite existing private receipt: $PRIVATE_RECEIPT"
fi
ARGS=(
  --platform linux/amd64
  --file "$BUILD_NPA/docker/workbench/hy-world/Dockerfile"
  --tag "$LOCAL_IMAGE"
  --build-arg "NPA_SOURCE_COMMIT=$SOURCE_COMMIT"
  --label "org.opencontainers.image.revision=$SOURCE_COMMIT"
  --load
  --provenance=false
)
if [[ -n "$REMOTE_IMAGE" ]]; then
  ARGS+=(--tag "$REMOTE_IMAGE")
fi
env -u HF_TOKEN -u NGC_API_KEY -u NEBIUS_IAM_TOKEN -u NPA_HY_WORLD_LLM_ADDR \
  docker buildx build "${ARGS[@]}" "$BUILD_NPA"

if (( ! PUSH )); then
  printf 'Built local neutral bootstrap: %s\n' "$LOCAL_IMAGE"
  printf '%s\n' 'No image was pushed. --push requires the private delivery transaction.'
  exit 0
fi

DELIVERY_DIR="$(mktemp -d "${RECEIPT_PARENT%/}/hy-world-delivery.XXXXXXXX")"
chmod 0700 "$DELIVERY_DIR"
TRIVY_IMAGE="aquasec/trivy:0.70.0@sha256:be1190afcb28352bfddc4ddeb71470835d16462af68d310f9f4bca710961a41e"
TRIVY_CACHE="$DELIVERY_DIR/trivy-cache"
mkdir -p "$TRIVY_CACHE"

run_trivy() {
  docker run --rm \
    -e TMPDIR=/tmp/trivy \
    -v "$DELIVERY_DIR:/delivery" \
    "$@"
}

# Do not add scanner exclusions here. The first scan applies the reviewed
# vulnerability/license policy; the second rejects every secret severity and
# deliberately omits the ignorefile. Save once and scan that archive, then
# push that same archive, rather than relying on a mutable local image tag.
LOCAL_CONFIG_DIGEST="$(docker image inspect --format '{{.Id}}' "$LOCAL_IMAGE")"
[[ "$LOCAL_CONFIG_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] \
  || fail "local image has no immutable OCI config digest"
IMAGE_TAR="$DELIVERY_DIR/local-image.tar"
docker save --output "$IMAGE_TAR" "$LOCAL_IMAGE"
LOCAL_ARCHIVE_SHA256="$(sha256sum "$IMAGE_TAR" | awk '{print $1}')"
[[ "$LOCAL_ARCHIVE_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || fail "could not hash locally scanned image archive"
run_trivy -v "$BUILD_ROOT/.trivyignore:/.trivyignore:ro" "$TRIVY_IMAGE" image \
  --input /delivery/local-image.tar --cache-dir /delivery/trivy-cache --timeout 2562047h47m16s \
  --scanners vuln,secret,license --severity CRITICAL --exit-code 1 --ignore-unfixed \
  --ignorefile /.trivyignore --format json --output /delivery/trivy-policy-local.json
run_trivy "$TRIVY_IMAGE" image --input /delivery/local-image.tar --cache-dir /delivery/trivy-cache \
  --timeout 2562047h47m16s --scanners secret \
  --severity UNKNOWN,LOW,MEDIUM,HIGH,CRITICAL --exit-code 1 --format json \
  --output /delivery/trivy-secrets-local.json
run_trivy "$TRIVY_IMAGE" image --input /delivery/local-image.tar --cache-dir /delivery/trivy-cache \
  --timeout 2562047h47m16s --scanners vuln --format spdx-json \
  --output /delivery/sbom-local.spdx.json
"$NPA_PYTHON" "$BUILD_NPA/scripts/scan_image_hy_world_payload.py" \
  --docker-save "$IMAGE_TAR" --output "$DELIVERY_DIR/payload-local.json"

# An immutable dev-SHA tag must never overwrite another image.  ``crane`` has
# no compare-and-swap push for a Docker archive, so establish tag absence before
# copying and fail closed on any result other than a Registry v2 "unknown" tag.
# Operators must also keep the private dev-tag namespace immutable to close the
# registry-side race; that invariant is stated in the delivery receipt.
if crane manifest "$REMOTE_IMAGE" >"$DELIVERY_DIR/remote-before.json" \
  2>"$DELIVERY_DIR/remote-before.stderr"; then
  fail "refusing to overwrite existing private immutable development tag"
fi
if ! grep -Eqi 'MANIFEST_UNKNOWN|NAME_UNKNOWN|[[:space:]]404[[:space:]]|not found' \
  "$DELIVERY_DIR/remote-before.stderr"; then
  fail "could not establish that the private development tag is absent"
fi

# Only the locally scanned archive reaches the private registry. Resolve the
# remote content digest immediately; subsequent stages consume the digest,
# never this mutable tag.
crane push "$IMAGE_TAR" "$REMOTE_IMAGE"
REMOTE_DIGEST="$(crane digest "$REMOTE_IMAGE")"
[[ "$REMOTE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] \
  || fail "registry did not return an immutable sha256 digest"
IMMUTABLE_IMAGE="${REGISTRY%/}/npa-hy-world@${REMOTE_DIGEST}"
crane manifest "$IMMUTABLE_IMAGE" >"$DELIVERY_DIR/manifest-remote.json"
PYTHONPATH="${BUILD_NPA}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "$NPA_PYTHON" -m npa.workbench.hy_world.private_delivery bind-manifest \
  --local-config-digest "$LOCAL_CONFIG_DIGEST" --remote-digest "$REMOTE_DIGEST" \
  --manifest "$DELIVERY_DIR/manifest-remote.json" \
  --output "$DELIVERY_DIR/identity-binding.json"

# `crane` uses the operator's exact-host Docker credential configuration. Pull
# the immutable remote digest through that authenticated host tool, then give
# scanner containers only the resulting archive. Credentials never enter a
# container argv, mount, image layer, receipt, or scanner cache.
rm -f -- "$IMAGE_TAR"
IMAGE_TAR=""
REMOTE_TAR="$DELIVERY_DIR/remote-image.tar"
crane pull "$IMMUTABLE_IMAGE" "$REMOTE_TAR"
REMOTE_ARCHIVE_SHA256="$(sha256sum "$REMOTE_TAR" | awk '{print $1}')"
[[ "$REMOTE_ARCHIVE_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || fail "could not hash authenticated remote image archive"

# A post-push gate scans the exact remote-digest archive whose OCI config was
# just bound to the local archive. This is not a payload-only recheck: the
# complete security/SBOM result describes the image the workflow later pulls.
run_trivy -v "$BUILD_ROOT/.trivyignore:/.trivyignore:ro" "$TRIVY_IMAGE" image \
  --input /delivery/remote-image.tar --cache-dir /delivery/trivy-cache --timeout 2562047h47m16s \
  --scanners vuln,secret,license --severity CRITICAL --exit-code 1 --ignore-unfixed \
  --ignorefile /.trivyignore --format json --output /delivery/trivy-policy-remote.json
run_trivy "$TRIVY_IMAGE" image --input /delivery/remote-image.tar --cache-dir /delivery/trivy-cache \
  --timeout 2562047h47m16s --scanners secret \
  --severity UNKNOWN,LOW,MEDIUM,HIGH,CRITICAL --exit-code 1 --format json \
  --output /delivery/trivy-secrets-remote.json
run_trivy "$TRIVY_IMAGE" image --input /delivery/remote-image.tar --cache-dir /delivery/trivy-cache \
  --timeout 2562047h47m16s --scanners vuln --format spdx-json \
  --output /delivery/sbom-remote.spdx.json
"$NPA_PYTHON" "$BUILD_NPA/scripts/scan_image_hy_world_payload.py" \
  --docker-save "$REMOTE_TAR" --output "$DELIVERY_DIR/payload-remote.json"
rm -f -- "$REMOTE_TAR"
REMOTE_TAR=""

SOURCE_COMMIT="$SOURCE_COMMIT" LOCAL_IMAGE="$LOCAL_IMAGE" \
  REMOTE_IMAGE="$REMOTE_IMAGE" IMMUTABLE_IMAGE="$IMMUTABLE_IMAGE" \
  LOCAL_CONFIG_DIGEST="$LOCAL_CONFIG_DIGEST" LOCAL_ARCHIVE_SHA256="$LOCAL_ARCHIVE_SHA256" \
  REMOTE_ARCHIVE_SHA256="$REMOTE_ARCHIVE_SHA256" \
  DELIVERY_DIR="$DELIVERY_DIR" "$NPA_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

directory = Path(os.environ["DELIVERY_DIR"])
payload = {
    "schema": "npa.hy_world.private_delivery.v1",
    "source_commit": os.environ["SOURCE_COMMIT"],
    "local_image": os.environ["LOCAL_IMAGE"],
    "local_config_digest": os.environ["LOCAL_CONFIG_DIGEST"],
    "local_archive_sha256": os.environ["LOCAL_ARCHIVE_SHA256"],
    "remote_archive_sha256": os.environ["REMOTE_ARCHIVE_SHA256"],
    "remote_tag": os.environ["REMOTE_IMAGE"],
    "immutable_image": os.environ["IMMUTABLE_IMAGE"],
    "security_receipts": {
        "payload_local": "payload-local.json",
        "payload_remote": "payload-remote.json",
        "trivy_policy_local": "trivy-policy-local.json",
        "trivy_secrets_local": "trivy-secrets-local.json",
        "sbom_local": "sbom-local.spdx.json",
        "trivy_policy_remote": "trivy-policy-remote.json",
        "trivy_secrets_remote": "trivy-secrets-remote.json",
        "sbom_remote": "sbom-remote.spdx.json",
        "identity_binding": "identity-binding.json",
    },
    "dev_tag_overwrite": "refused-before-push; operator-private registry must enforce immutable dev tags",
    "publication": "operator-private-validation-only",
}
(directory / "private-delivery.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
# A stable, source-SHA receipt lets an agent read the immutable image reference
# as structured data without scraping build logs.  ``ln`` is atomic and refuses
# an existing name; both paths are below the verified private receipt parent.
ln "$DELIVERY_DIR/private-delivery.json" "$PRIVATE_RECEIPT"

printf 'Private validation digest: %s\n' "$IMMUTABLE_IMAGE"
printf 'Private receipts: %s\n' "$DELIVERY_DIR"
printf 'Structured receipt: %s\n' "$PRIVATE_RECEIPT"
printf '%s\n' 'This is not a public development image or release; retain receipts for GPU qualification.'
