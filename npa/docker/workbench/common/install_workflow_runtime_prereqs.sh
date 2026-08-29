#!/usr/bin/env bash
# Install the non-root SkyPilot Kubernetes bootstrap closure.

set -euo pipefail

snapshot="${1:?usage: install_workflow_runtime_prereqs.sh UBUNTU_SNAPSHOT}"
if [ "$(id -u)" -ne 0 ]; then
  echo "workflow runtime prerequisites must be installed as root" >&2
  exit 1
fi

# The Genesis-derived Sim2Real images currently inherit Ubuntu 22.04. Keep the
# mapping explicit and fail closed if their base changes: silently pointing an
# unknown release at a moving mirror would make the supposedly immutable image
# depend on build time.
. /etc/os-release
case "${ID}:${VERSION_ID}" in
  ubuntu:22.04)
    suites="jammy jammy-updates jammy-backports jammy-security"
    ;;
  ubuntu:24.04)
    suites="noble noble-updates noble-backports noble-security"
    ;;
  *)
    echo "unsupported workflow runtime base: ${ID}:${VERSION_ID}" >&2
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

apt-get update
# NVIDIA's Genesis-derived base contains development packages whose declared
# dependencies are absent from its final layer (observed live: libc6-dev
# without linux-libc-dev).  Apt refuses even unrelated installs while that
# graph is broken.  Repair it from the same immutable snapshot first; this is
# deliberately fail-closed and never falls back to a moving mirror.
apt-get --fix-broken install -y --no-install-recommends
# Genesis' Jammy base retains an older linux-libc-dev build. These are
# userspace development headers rather than the cluster's kernel, but the
# fixed build is available in this immutable snapshot, so do not publish the
# avoidable critical CVEs inherited from the parent filesystem.
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  ffmpeg \
  linux-libc-dev=5.15.0-190.200 \
  netcat-openbsd \
  openssh-client \
  openssh-server \
  procps \
  python3 \
  rsync \
  sudo \
  wget
rm -rf /var/lib/apt/lists/*
rm -f /etc/ssh/ssh_host_*
test -z "$(find /etc/ssh -maxdepth 1 -type f -name 'ssh_host_*' -print -quit)"

id -u ubuntu >/dev/null 2>&1
printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' \
  > /etc/sudoers.d/99-npa-workflow-runtime
chmod 0440 /etc/sudoers.d/99-npa-workflow-runtime
visudo -cf /etc/sudoers.d/99-npa-workflow-runtime
install -d -m 0755 /run/sshd
test -x /usr/bin/python3
su ubuntu -s /bin/bash -c \
  'test "$(id -u)" -ne 0 && test -x /usr/bin/python3 && sudo -n true && rsync --version >/dev/null'
