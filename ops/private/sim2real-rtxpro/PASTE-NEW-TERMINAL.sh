#!/usr/bin/env bash
# Paste block for a new Mac terminal — see OPERATOR-GUIDE.md
bash <<'EOF'
set -euo pipefail
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin:${HOME}/.nebius/bin:${PATH}"
export KUBECONFIG="${KUBECONFIG:-$HOME/.npa/clusters/npa-rtxpro-mk8s/kubeconfig.resolved}"
export KUBECONTEXT="${KUBECONTEXT:-npa-rtxpro-mk8s}"
[[ -f "$HOME/.npa/sim2real-operator.env" ]] && source "$HOME/.npa/sim2real-operator.env"
DEMO="$HOME/npa-sim2real-demo"
REPO="$DEMO/nebius-physical-ai"
if [[ ! -d "$REPO/.git" ]]; then
  git clone --branch feat/sim2real-mandatory-stages https://github.com/nebius/nebius-physical-ai.git "$REPO"
fi
(cd "$REPO" && git fetch origin feat/sim2real-mandatory-stages && git checkout feat/sim2real-mandatory-stages && git pull --ff-only origin feat/sim2real-mandatory-stages || git reset --hard origin/feat/sim2real-mandatory-stages)
cp "$REPO/ops/private/sim2real-rtxpro/mac-run.sh" "$DEMO/run.sh" && chmod +x "$DEMO/run.sh"
cd "$DEMO" && ./run.sh demo
EOF
