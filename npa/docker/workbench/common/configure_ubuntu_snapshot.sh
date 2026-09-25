#!/usr/bin/env bash
# Replace every inherited APT source with one immutable Ubuntu snapshot.

set -euo pipefail

snapshot="${1:?usage: configure_ubuntu_snapshot.sh UBUNTU_SNAPSHOT}"
if [ "$(id -u)" -ne 0 ]; then
  echo "Ubuntu snapshot configuration must run as root" >&2
  exit 1
fi

. /etc/os-release
case "${ID}:${VERSION_ID}" in
  ubuntu:22.04)
    suites="jammy jammy-updates jammy-backports jammy-security"
    ;;
  ubuntu:24.04)
    suites="noble noble-updates noble-backports noble-security"
    ;;
  *)
    echo "unsupported Ubuntu snapshot base: ${ID}:${VERSION_ID}" >&2
    exit 1
    ;;
esac

rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*.list \
  /etc/apt/sources.list.d/*.sources
printf '%s\n' \
  'Types: deb' \
  "URIs: https://snapshot.ubuntu.com/ubuntu/${snapshot}/" \
  "Suites: ${suites}" \
  'Components: main restricted universe multiverse' \
  'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg' \
  > /etc/apt/sources.list.d/ubuntu.sources
