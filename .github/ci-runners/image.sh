#!/usr/bin/env bash
# Prepare a credential-free Ubuntu image for disposable GitHub Actions workers.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get upgrade -y
apt-get install -y --no-install-recommends \
  build-essential ca-certificates curl docker.io file gh git gnupg jq \
  libasound2t64 libatk-bridge2.0-0 libatk1.0-0 libcups2 libgbm1 libgtk-3-0 \
  libnss3 libxss1 libxtst6 openssh-client pkg-config python-is-python3 \
  python3 python3-dev python3-pip python3-venv rsync shellcheck sudo \
  tmux unzip xauth xvfb xz-utils zip zstd

useradd --create-home --shell /bin/bash runner
usermod -aG docker runner
printf '%s\n' 'runner ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/runner
chmod 0440 /etc/sudoers.d/runner
install -d -o runner -g runner /opt/actions-runner /opt/hostedtoolcache

archive=$(mktemp)
trap 'rm -f "$archive"' EXIT
curl --fail --location --proto '=https' --tlsv1.2 \
  https://github.com/actions/runner/releases/download/v2.337.0/actions-runner-linux-x64-2.337.0.tar.gz \
  --output "$archive"
printf '%s  %s\n' \
  70920811a4f8ad4328818682bca5c6469c1c942fab52448868071d0063816613 \
  "$archive" | sha256sum --check
tar -xzf "$archive" -C /opt/actions-runner
/opt/actions-runner/bin/installdependencies.sh
chown -R runner:runner /opt/actions-runner /opt/hostedtoolcache
curl --fail --location --proto '=https' --tlsv1.2 \
  https://github.com/cli/cli/releases/download/v2.101.0/gh_2.101.0_linux_amd64.tar.gz \
  --output "$archive"
printf '%s  %s\n' \
  9bca2d1c16825f109907a23307628a2f0698fbf99662b73a5cf0b020293072b8 \
  "$archive" | sha256sum --check
tar -xzf "$archive" -C /usr/local --strip-components=1 \
  gh_2.101.0_linux_amd64/bin/gh
systemctl enable docker
printf '%s\n' 'npa-ci-runner-image-v1' > /etc/npa-ci-runner-image
