#!/usr/bin/env bash
#
# install-sdk.sh - install the pinned @foxglove/embed TypeScript SDK assets.
#
# The SDK (https://docs.foxglove.dev/docs/embed/typescript-sdk) is MIT licensed,
# has no runtime dependencies, and publishes browser-ready ESM in the npm
# package's dist/ directory. NPA therefore serves those files verbatim rather
# than vendoring a bundled copy.
#
# Two consumers share this one recipe so they can never drift:
#   * npa/docker/workbench/foxglove-embed/Dockerfile  (container build)
#   * npa agent bootstrap                             (/opt/npa-agent/foxglove/sdk)
#
# The download is verified against the npm registry's `dist.integrity` digest
# (base64 sha512). A mismatch aborts without touching the destination.
#
# Usage:
#   install-sdk.sh --dest <dir> [--version <semver>] [--integrity sha512-<b64>]
#                  [--registry <url>]
#
# Defaults mirror npa.workbench.foxglove (FOXGLOVE_EMBED_SDK_VERSION /
# FOXGLOVE_EMBED_SDK_INTEGRITY); npa/tests/docker/test_foxglove_image.py fails if
# they drift apart.
set -euo pipefail

VERSION="0.58.0"
INTEGRITY="sha512-hNxqEQWPk2Wm0KmDlNs3Y0TTEl9Wm+4CuppBZcLzK8j8m2EcwbbCVWg43oCsf5HJgwXt7KYorIdoMO7CICQ7Vg=="
REGISTRY="https://registry.npmjs.org"
DEST=""

usage() {
  sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dest) DEST="${2:?--dest requires a directory}"; shift 2 ;;
    --version) VERSION="${2:?--version requires a value}"; shift 2 ;;
    --integrity) INTEGRITY="${2:?--integrity requires a value}"; shift 2 ;;
    --registry) REGISTRY="${2:?--registry requires a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "install-sdk.sh: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$DEST" ]; then
  echo "install-sdk.sh: --dest is required" >&2
  exit 2
fi

for binary in curl tar openssl; do
  command -v "$binary" >/dev/null 2>&1 || {
    echo "install-sdk.sh: missing required binary: $binary" >&2
    exit 3
  }
done

TARBALL_URL="${REGISTRY%/}/@foxglove/embed/-/embed-${VERSION}.tgz"
WORKDIR="$(mktemp -d)"
cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

echo "install-sdk.sh: fetching @foxglove/embed@${VERSION} from ${TARBALL_URL}"
curl -fsSL --retry 3 --retry-delay 2 --max-time 120 -o "$WORKDIR/embed.tgz" "$TARBALL_URL"

ACTUAL="sha512-$(openssl dgst -sha512 -binary "$WORKDIR/embed.tgz" | openssl base64 -A)"
if [ "$ACTUAL" != "$INTEGRITY" ]; then
  echo "install-sdk.sh: integrity mismatch for @foxglove/embed@${VERSION}" >&2
  echo "  expected: $INTEGRITY" >&2
  echo "  actual:   $ACTUAL" >&2
  exit 4
fi
echo "install-sdk.sh: integrity verified (${ACTUAL%%-*} digest matches the pinned value)"

# The npm package layout is package/dist/*.js — strip both leading components so
# the destination holds index.js next to its relative imports.
mkdir -p "$WORKDIR/stage"
tar -xzf "$WORKDIR/embed.tgz" -C "$WORKDIR/stage" --strip-components=2 package/dist

for required in index.js FoxgloveViewer.js types.js layout.generated.js; do
  [ -s "$WORKDIR/stage/$required" ] || {
    echo "install-sdk.sh: extracted SDK is missing $required" >&2
    exit 5
  }
done

# Source maps reference sources that are not shipped to the browser; drop them so
# devtools does not 404 on every load.
rm -f "$WORKDIR/stage"/*.map

# Record provenance next to the assets (surfaced by /api/foxglove/config).
cat > "$WORKDIR/stage/npa-sdk-manifest.json" <<JSON
{
  "package": "@foxglove/embed",
  "version": "${VERSION}",
  "integrity": "${INTEGRITY}",
  "source": "${TARBALL_URL}",
  "license": "MIT"
}
JSON

chmod -R a+rX "$WORKDIR/stage"

# Atomic-ish swap: never leave a half-populated destination behind.
mkdir -p "$(dirname "$DEST")"
rm -rf "$DEST.incoming"
mv "$WORKDIR/stage" "$DEST.incoming"
rm -rf "$DEST"
mv "$DEST.incoming" "$DEST"

echo "install-sdk.sh: installed @foxglove/embed@${VERSION} into ${DEST}"
