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
libpython3.12_3.12.16-1+jammy1_amd64.deb 80c863d7d18530e9b58d4ec90519fd52f1f57b88ed2d407ada3d35af5a73d1e5
libpython3.12-dev_3.12.16-1+jammy1_amd64.deb cac6c77319ecff0cd4276cd6ef927f973ef019a37052bcd3165d01e97de9e422
libpython3.12-stdlib_3.12.16-1+jammy1_amd64.deb 60c27abdaec2057115f3b557d73f521715062bf29ca403a53d63ab35069ed219
python3.12_3.12.16-1+jammy1_amd64.deb 1824684d47a9711fad480f2330ae86749cdf6b93516868863db39550563e4915
python3.12-dev_3.12.16-1+jammy1_amd64.deb 9e0bb14ece20676cfb0c8e58d77354892551c59228f8e42ec3ec0a2bbb9cf6f4
python3.12-venv_3.12.16-1+jammy1_amd64.deb 92c86592323e3bc01ed6dfebeca053832735ca74bddd525c00e6f19ff72c8337
PYTHON312_PACKAGES
apt-get install -y --no-install-recommends /tmp/npa-python312/*.deb
test "$(dpkg-query -W -f='${Version}' python3.12)" = "3.12.16-1+jammy1"
rm -f /etc/ssh/ssh_host_*
test -z "$(find /etc/ssh -maxdepth 1 -type f -name 'ssh_host_*' -print -quit)"
# The image-byte gate treats these as path-only findings in every layer. After
# every system package is installed, fail before committing this layer if any
# PKCS#12 path remains anywhere on the root filesystem.
test -z "$(find / -xdev \( -iname '*.p12' -o -iname '*.pfx' \) -print -quit)"
rm -rf /tmp/npa-python312 /var/lib/apt/lists/*
