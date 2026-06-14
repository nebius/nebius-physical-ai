#!/usr/bin/env bash
# Install CLI tools for sim2real operator demo (Mac or Linux). Idempotent.
set -euo pipefail

echo "=== Sim2Real prerequisites ==="

export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin:${HOME}/.nebius/bin:${PATH}"

_need() {
  command -v "$1" >/dev/null 2>&1
}

_install_nebius_cli() {
  if _need nebius; then
    echo "  nebius: $(command -v nebius)"
    return 0
  fi
  echo "  installing Nebius CLI..."
  curl -fsSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
  export PATH="${HOME}/.nebius/bin:${PATH}"
}

case "$(uname -s)" in
  Darwin)
    if ! _need git; then
      echo "  install Xcode CLT: xcode-select --install"
      xcode-select --install 2>/dev/null || true
    fi
    if ! _need brew; then
      echo "ERROR: Homebrew required on Mac — https://brew.sh" >&2
      exit 1
    fi
    for pkg in python@3.12 kubectl awscli git; do
      if brew list "$pkg" >/dev/null 2>&1; then
        echo "  brew $pkg: ok"
      else
        echo "  brew install $pkg"
        brew install "$pkg"
      fi
    done
    _install_nebius_cli
    ;;
  Linux)
    if ! _need git; then
      sudo apt-get update -qq && sudo apt-get install -y git
    fi
    if ! _need python3; then
      sudo apt-get update -qq && sudo apt-get install -y python3 python3-venv python3-pip
    fi
    if ! _need kubectl; then
      curl -fsSL "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
        -o /tmp/kubectl && chmod +x /tmp/kubectl && sudo mv /tmp/kubectl /usr/local/bin/kubectl
    fi
    if ! _need aws; then
      sudo apt-get update -qq && sudo apt-get install -y awscli || pip3 install --user awscli
    fi
    _install_nebius_cli
    ;;
  *)
    echo "ERROR: unsupported OS — use Mac or Linux" >&2
    exit 1
    ;;
esac

for cmd in git python3 kubectl aws nebius; do
  if _need "$cmd"; then
    echo "  ok: $cmd"
  else
    echo "  MISSING: $cmd" >&2
    exit 1
  fi
done

echo "=== Prerequisites OK ==="
