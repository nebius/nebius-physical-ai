#!/usr/bin/env bash
# Install the checksum-verified Trivy binary alongside the pinned Python scanners.
set -euo pipefail
umask 077

tool_directory="${1:?Pass a private tools directory}"
mkdir -p "$tool_directory"
tool_directory="$(cd "$tool_directory" && pwd)"
curl --fail --silent --show-error --location \
  https://github.com/aquasecurity/trivy/releases/download/v0.74.0/trivy_0.74.0_Linux-64bit.tar.gz \
  --output "$tool_directory/trivy.tar.gz"
(
  cd "$tool_directory"
  printf '%s\n' '2ae6fe3ee734b7fdf11335663e18c75ea12dccc76062f09f164a3b0f8be4371a  trivy.tar.gz' | sha256sum --check
  tar -xzf trivy.tar.gz trivy
  chmod 0700 trivy
  ./trivy --version
)
