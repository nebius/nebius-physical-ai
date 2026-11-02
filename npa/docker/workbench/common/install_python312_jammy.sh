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
libpython3.12_3.12.16-1+jammy1_amd64.deb|80c863d7d18530e9b58d4ec90519fd52f1f57b88ed2d407ada3d35af5a73d1e5
libpython3.12-dev_3.12.16-1+jammy1_amd64.deb|cac6c77319ecff0cd4276cd6ef927f973ef019a37052bcd3165d01e97de9e422
libpython3.12-stdlib_3.12.16-1+jammy1_amd64.deb|60c27abdaec2057115f3b557d73f521715062bf29ca403a53d63ab35069ed219
python3.12_3.12.16-1+jammy1_amd64.deb|1824684d47a9711fad480f2330ae86749cdf6b93516868863db39550563e4915
python3.12-dev_3.12.16-1+jammy1_amd64.deb|9e0bb14ece20676cfb0c8e58d77354892551c59228f8e42ec3ec0a2bbb9cf6f4
python3.12-venv_3.12.16-1+jammy1_amd64.deb|92c86592323e3bc01ed6dfebeca053832735ca74bddd525c00e6f19ff72c8337
PYTHON312_DEBS

apt-get update
apt-get install -y --no-install-recommends "${python_debs}"/*.deb
test "$(dpkg-query -W -f='${Version}' python3.12)" = 3.12.16-1+jammy1
python3.12 -c 'import sys; assert sys.version_info[:2] == (3, 12)'
