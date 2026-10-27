#!/usr/bin/env bash
# Install only from a dated Ubuntu snapshot and verified Python 3.12 packages.
# This runs inside the build stage; it does not grant host package or credential access.
set -euo pipefail

readonly UBUNTU_SNAPSHOT="https://snapshot.ubuntu.com/ubuntu/20260820T000000Z"
readonly DEADSNakes_BASE="https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu"
readonly PYTHON312_POOL="${DEADSNakes_BASE}/pool/main/p/python3.12"
readonly CMAKE_WINDOWS_TEMPLATE="/usr/share/cmake-3.22/Templates/Windows/Windows_TemporaryKey.pfx"

test -r /usr/share/keyrings/ubuntu-archive-keyring.gpg
rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*.list
cat >/etc/apt/sources.list.d/npa-snapshot.list <<EOF
deb [signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] ${UBUNTU_SNAPSHOT} jammy main universe
deb [signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] ${UBUNTU_SNAPSHOT} jammy-updates main universe
deb [signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] ${UBUNTU_SNAPSHOT} jammy-security main universe
EOF

apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates=20260601~22.04.1 \
  curl=7.81.0-1ubuntu1.26 \
  build-essential=12.9ubuntu3 \
  cmake=3.22.1-1ubuntu1.22.04.2 \
  ffmpeg=7:4.4.2-0ubuntu0.22.04.1 \
  fuse=2.9.9-5ubuntu3 \
  git=1:2.34.1-1ubuntu1.17 \
  libegl1=1.4.0-1 \
  libegl1-mesa-dev=23.2.1-1ubuntu3.1~22.04.4 \
  libffi-dev=3.4.2-4 \
  libgl1=1.4.0-1 \
  libgl1-mesa-dev=23.2.1-1ubuntu3.1~22.04.4 \
  libglib2.0-0=2.72.4-0ubuntu2.9 \
  libglvnd0=1.4.0-1 \
  libsm6=2:1.2.3-1build2 \
  libssl-dev=3.0.2-0ubuntu1.26 \
  libx11-dev=2:1.7.5-1ubuntu0.3 \
  libxext-dev=2:1.3.4-1build1 \
  libxext6=2:1.3.4-1build1 \
  libxrender-dev=1:0.9.10-1build4 \
  ninja-build=1.10.1-1 \
  pkg-config=0.29.2-1ubuntu3 \
  python3=3.10.6-1~22.04.1 \
  openssh-server=1:8.9p1-3ubuntu0.16 \
  rsync=3.2.7-0ubuntu0.22.04.7 \
  sudo=1.9.9-1ubuntu2.6 \
  netcat-openbsd=1.218-4ubuntu1 \
  wget=1.21.2-2ubuntu1.4 \
  openssl=3.0.2-0ubuntu1.26 \
  libssl3=3.0.2-0ubuntu1.26

test "$(dpkg-query -W -f='${Version}' fuse)" = "2.9.9-5ubuntu3"
test "$(dpkg-query -W -f='${Version}' wget)" = "1.21.2-2ubuntu1.4"
test -x /bin/fusermount
test -x /usr/bin/wget

# cmake-data ships a Windows-only signing template. Remove this known inert
# PKCS#12 path in the same layer that installs cmake so it is never emitted in
# an ancestor layer.
test -f "${CMAKE_WINDOWS_TEMPLATE}"
rm "${CMAKE_WINDOWS_TEMPLATE}"
test ! -e "${CMAKE_WINDOWS_TEMPLATE}"
test ! -L "${CMAKE_WINDOWS_TEMPLATE}"

install -d -m 0700 /tmp/npa-python312
while IFS=' ' read -r package sha256; do
  curl --fail --location --silent --show-error \
    "${PYTHON312_POOL}/${package}" \
    --output "/tmp/npa-python312/${package}"
  printf '%s  %s\n' "${sha256}" "/tmp/npa-python312/${package}" | sha256sum --check --status
done <<'PYTHON312_PACKAGES'
libpython3.12_3.12.15-1+jammy1_amd64.deb f89750316bb92d36c5a15c970893044732806217f80d58c97b018082f2259823
libpython3.12-dev_3.12.15-1+jammy1_amd64.deb 990b8bfac930065343c4d8f87098888ba27b28c22470111e15e3987fff382e94
libpython3.12-stdlib_3.12.15-1+jammy1_amd64.deb 691f31a5a861c96c5b8ebc81787da86688a4f5671a3adbc059e5e0691d930dcd
python3.12_3.12.15-1+jammy1_amd64.deb 0994bc05716e910c314f89212672ac835bfacefbf6e3d56d613789dd876fd3af
python3.12-dev_3.12.15-1+jammy1_amd64.deb 361f4f0a9bc8babe977716735204b9f6943309bd85eecfc31d1f0c0c95a74689
python3.12-venv_3.12.15-1+jammy1_amd64.deb a15bfc45040d8a9d863d3b79175f4d8f381191746fd49df198968c779301ed3e
PYTHON312_PACKAGES
apt-get install -y --no-install-recommends /tmp/npa-python312/*.deb
test "$(python3.12 --version)" = "Python 3.12.15"
rm -f /etc/ssh/ssh_host_*
test -z "$(find /etc/ssh -maxdepth 1 -type f -name 'ssh_host_*' -print -quit)"
# The image-byte gate treats these as path-only findings in every layer. After
# every system package is installed, fail before committing this layer if any
# PKCS#12 path remains anywhere on the root filesystem.
test -z "$(find / -xdev \( -iname '*.p12' -o -iname '*.pfx' \) -print -quit)"
rm -rf /tmp/npa-python312 /var/lib/apt/lists/*
