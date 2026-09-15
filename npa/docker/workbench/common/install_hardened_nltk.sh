#!/usr/bin/env bash
# Install NLTK from the upstream commit that fixes GHSA-8mgp-746c-j5xp.

set -euo pipefail

readonly NLTK_SOURCE_COMMIT="cbc98458b43de5f792f0382583c16df39e5c5117"
readonly NLTK_SOURCE_SHA256="3c4a9e92b53e34074ae5b7bbf21109868b5ef620b40c68b8542a2db37f7d9d0c"
readonly NLTK_SOURCE_VERSION="3.10.3"
readonly NLTK_HARDENED_VERSION="3.10.3.post1+npa.cbc98458"
readonly NLTK_SOURCE_URL="https://codeload.github.com/nltk/nltk/tar.gz/${NLTK_SOURCE_COMMIT}"

if [ "$#" -ne 1 ]; then
  echo "usage: $0 SYSTEM|PYTHON" >&2
  exit 64
fi

target="$1"
if [ "$target" = SYSTEM ]; then
  python_path="$(command -v python)"
  uv_target=(--system)
elif [ -x "$target" ]; then
  python_path="$target"
  uv_target=(--python "$target")
else
  echo "target Python is not executable: $target" >&2
  exit 66
fi

work="$(mktemp -d)"
cleanup() {
  rm -rf -- "$work"
}
trap cleanup EXIT

archive="$work/nltk.tar.gz"
source_dir="$work/source"
curl --fail --location "$NLTK_SOURCE_URL" --output "$archive"
printf '%s  %s\n' "$NLTK_SOURCE_SHA256" "$archive" | sha256sum --check --strict
mkdir "$source_dir"
tar --extract --gzip --file "$archive" --strip-components=1 --directory "$source_dir"
test "$(cat "$source_dir/nltk/VERSION")" = "$NLTK_SOURCE_VERSION"
printf '%s\n' "$NLTK_HARDENED_VERSION" > "$source_dir/nltk/VERSION"

# The overlays install an exact setuptools version first. Avoid an unpinned
# isolated build environment, and install only this verified source.
UV_CACHE_DIR="${UV_CACHE_DIR:-$work/uv-cache}" \
  uv pip install "${uv_target[@]}" --no-deps --no-build-isolation "$source_dir"
test "$("$python_path" -c 'import importlib.metadata; print(importlib.metadata.version("nltk"))')" = \
  "$NLTK_HARDENED_VERSION"
