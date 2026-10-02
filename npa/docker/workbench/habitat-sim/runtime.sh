#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ "$(id -u)" == 1000 ]] || { echo 'Habitat runtime requires workload user' >&2; exit 78; }
lock_root=/usr/share/doc/npa-habitat-sim
runtime_root="$(mktemp -d "${TMPDIR:-/tmp}/npa-habitat-runtime.XXXXXX")"
cleanup() { local status=$?; trap - EXIT; rm -rf -- "$runtime_root"; exit "$status"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# The public bootstrap carries no simulator, scientific wheels or GPU libraries.
# Official Ubuntu binaries and the already-reviewed upstream source/wheel locks
# are obtained only in this ephemeral operator workload.
sudo apt-get update
bash "$lock_root/verify_apt_artifacts.sh" direct-records "$lock_root/apt-build.lock" "$runtime_root/apt-packages"
mkdir "$runtime_root/debs"
while IFS='|' read -r package version _digest; do
  (cd "$runtime_root/debs" && apt-get download "$package=$version")
done < "$runtime_root/apt-packages"
bash "$lock_root/verify_apt_artifacts.sh" verify-direct "$runtime_root/apt-packages" "$runtime_root/debs"
sudo apt-get install -y --no-install-recommends "$runtime_root"/debs/*.deb
/usr/bin/python3 "$lock_root/prepare_source.py" --manifest "$lock_root/source-manifest.json" \
  --output "$runtime_root/source" --inventory-output "$runtime_root/source-projection.json"
/usr/bin/python3 -m venv "$runtime_root/venv"
sed 's/^# runtime-fetch: //' "$lock_root/requirements-runtime.lock" > "$runtime_root/runtime.lock"
"$runtime_root/venv/bin/pip" install --disable-pip-version-check --no-cache-dir \
  --only-binary=:all: --no-deps --require-hashes \
  -r "$lock_root/requirements-build.lock" -r "$runtime_root/runtime.lock"
(
  cd "$runtime_root/source"
  HABITAT_BUILD_GUI_VIEWERS=OFF HABITAT_WITH_BULLET=ON HABITAT_WITH_CUDA=OFF \
  HABITAT_WITH_AUDIO=OFF HABITAT_BUILD_TEST=OFF HABITAT_BUILD_BASIS_COMPRESSOR=OFF \
  SKBUILD_CMAKE_BUILD_TYPE=Release \
  CMAKE_ARGS="-DFETCHCONTENT_SOURCE_DIR_IMATH=$runtime_root/source/src/deps/imath -DFETCHCONTENT_FULLY_DISCONNECTED=ON" \
    "$runtime_root/venv/bin/pip" install . --no-build-isolation --no-deps
)
"$runtime_root/venv/bin/pip" check
export NPA_HABITAT_SOURCE_ROOT="$runtime_root/source"
"$runtime_root/venv/bin/python" "$@"
