#!/usr/bin/env bash
# Daily session alias — delegates to ~/npa-sim2real-demo/run.sh demo
set -euo pipefail

export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin:/opt/homebrew/opt/python@3.12/libexec/bin:${HOME}/.nebius/bin:${PATH}"

DEMO="${NPA_SIM2REAL_DEMO:-$HOME/npa-sim2real-demo}"
REPO="$DEMO/nebius-physical-ai"
OPS="$REPO/ops/private/sim2real-rtxpro"
export NPA_SIM2REAL_DEMO="$DEMO"

if [ ! -d "$REPO/.git" ]; then
  echo "ERROR: run first-time-setup.sh first (see QUICKSTART.md)" >&2
  exit 1
fi

mkdir -p "$DEMO"
RUN_SRC="${OPS}/operator-run.sh"
[ -f "${RUN_SRC}" ] || RUN_SRC="${OPS}/mac-run.sh"
cp "${RUN_SRC}" "$DEMO/run.sh" && chmod +x "$DEMO/run.sh"

INSTALL_LIB="${OPS}/lib/private-install.sh"
if [ ! -f "${INSTALL_LIB}" ]; then
  INSTALL_LIB="$(cd "$(dirname "$0")" && pwd)/lib/private-install.sh"
fi
# shellcheck disable=SC1091
source "${INSTALL_LIB}"
operator_install_private_config

if [ "${SIM2REAL_PASTE_SKIP_DEMO:-0}" = "1" ]; then
  cd "$DEMO" && ./run.sh help >/dev/null 2>&1 || true
  echo "SIM2REAL_PASTE_SKIP_DEMO=1 — stop before ./run.sh demo"
  exit 0
fi

cd "$DEMO"
exec ./run.sh demo
