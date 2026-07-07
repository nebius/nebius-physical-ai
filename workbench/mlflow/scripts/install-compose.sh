#!/usr/bin/env bash
set -euo pipefail
if docker compose version >/dev/null 2>&1; then
  docker compose version
  exit 0
fi
version="${DOCKER_COMPOSE_VERSION:-v2.40.3}"
arch="$(uname -m)"
case "$arch" in
  x86_64|amd64) arch=x86_64 ;;
  aarch64|arm64) arch=aarch64 ;;
  *) echo "unsupported arch: $arch" >&2; exit 1 ;;
esac
mkdir -p "$HOME/.docker/cli-plugins"
url="https://github.com/docker/compose/releases/download/${version}/docker-compose-linux-${arch}"
curl -fsSL "$url" -o "$HOME/.docker/cli-plugins/docker-compose"
chmod 0755 "$HOME/.docker/cli-plugins/docker-compose"
docker compose version
