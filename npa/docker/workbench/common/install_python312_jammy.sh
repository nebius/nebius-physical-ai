#!/usr/bin/env bash
# Install the hash-verified Python 3.12 runtime on Ubuntu Jammy.
set -euo pipefail

. /etc/os-release
if [ "$(id -u)" != 0 ] || [ "${ID}:${VERSION_ID}" != ubuntu:22.04 ]; then
  echo 'Python 3.12 bootstrap requires root on Ubuntu 22.04' >&2
  exit 1
fi

python_debs="$(mktemp -d)"
trap 'rm -rf "$python_debs"' EXIT
while IFS='|' read -r filename digest; do
  curl --fail --location --retry 3 \
    "https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu/pool/main/p/python3.12/${filename}" \
    --output "${python_debs}/${filename}"
  printf '%s  %s\n' "$digest" "${python_debs}/${filename}" | sha256sum --check --strict
done <<'PYTHON312_DEBS'
libpython3.12_3.12.15-1+jammy1_amd64.deb|f89750316bb92d36c5a15c970893044732806217f80d58c97b018082f2259823
libpython3.12-dev_3.12.15-1+jammy1_amd64.deb|990b8bfac930065343c4d8f87098888ba27b28c22470111e15e3987fff382e94
libpython3.12-stdlib_3.12.15-1+jammy1_amd64.deb|691f31a5a861c96c5b8ebc81787da86688a4f5671a3adbc059e5e0691d930dcd
python3.12_3.12.15-1+jammy1_amd64.deb|0994bc05716e910c314f89212672ac835bfacefbf6e3d56d613789dd876fd3af
python3.12-dev_3.12.15-1+jammy1_amd64.deb|361f4f0a9bc8babe977716735204b9f6943309bd85eecfc31d1f0c0c95a74689
python3.12-venv_3.12.15-1+jammy1_amd64.deb|a15bfc45040d8a9d863d3b79175f4d8f381191746fd49df198968c779301ed3e
PYTHON312_DEBS

apt-get update
apt-get install -y --no-install-recommends "${python_debs}"/*.deb
test "$(dpkg-query -W -f='${Version}' python3.12)" = 3.12.15-1+jammy1
python3.12 -c 'import sys; assert sys.version_info[:2] == (3, 12)'
