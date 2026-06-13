#!/usr/bin/env bash
# Shell bootstrap for Mac operators — PATH, kubeconfig, operator env file.
# Source after operator-env.sh; optional ROOT for config.yaml lookup.

operator_bootstrap_shell() {
  local root="${1:-}"

  if [[ -f "${HOME}/.npa/sim2real-operator.env" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/.npa/sim2real-operator.env"
  fi

  local ctx="${KUBECONTEXT:-}"
  if [[ -z "${ctx}" && -n "${root}" && -f "${root}/npa/.venv/bin/python" ]]; then
    ctx="$("${root}/npa/.venv/bin/python" - <<'PY' 2>/dev/null || true
import yaml
from pathlib import Path
path = Path.home() / ".npa" / "config.yaml"
if not path.exists():
    raise SystemExit(0)
cfg = yaml.safe_load(path.read_text()) or {}
storage = cfg.get("storage") or {}
ctx = str(storage.get("k8s_context", "") or "")
if not ctx:
    for proj in (cfg.get("projects") or {}).values():
        if isinstance(proj, dict) and proj.get("k8s_context"):
            ctx = str(proj["k8s_context"])
            break
if ctx:
    print(ctx)
PY
)"
  fi

  if [[ -n "${ctx}" ]]; then
    export KUBECONTEXT="${ctx}"
  fi

  if [[ -z "${KUBECONFIG:-}" && -n "${KUBECONTEXT:-}" ]]; then
    local kc="${HOME}/.npa/clusters/${KUBECONTEXT}/kubeconfig.resolved"
    if [[ ! -f "${kc}" ]]; then
      kc="${HOME}/.npa/clusters/${KUBECONTEXT}/kubeconfig"
    fi
    if [[ -f "${kc}" ]]; then
      export KUBECONFIG="${kc}"
    fi
  fi
}

# When sourced directly (no function call needed if ROOT in env).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  # shellcheck source=operator-env.sh
  source "${SCRIPT_DIR}/operator-env.sh"
  operator_bootstrap_shell "${NPA_SIM2REAL_REPO:-}"
fi
