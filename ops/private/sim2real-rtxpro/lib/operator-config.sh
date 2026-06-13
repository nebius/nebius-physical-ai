#!/usr/bin/env bash
# Resolve operator config from ~/.npa/config.yaml (no secrets, no hardcoded tenant IDs).

# Walk up from START until npa/pyproject.toml is found (works from lib/ or script dir).
npa_repo_root() {
  local start="${1:-.}"
  local dir
  dir="$(cd "${start}" && pwd)"
  while [ "${dir}" != "/" ]; do
    if [ -f "${dir}/npa/pyproject.toml" ]; then
      printf '%s\n' "${dir}"
      return 0
    fi
    dir="$(dirname "${dir}")"
  done
  echo "ERROR: repo root not found (expected npa/pyproject.toml above ${start})" >&2
  return 1
}

# Bash 3.2 (macOS default) lacks readarray — capture command lines into ARRAY_NAME.
npa_read_lines() {
  local _arr="${1:?array name required}"
  shift
  local _line
  eval "${_arr}=()"
  while IFS= read -r _line; do
    eval "${_arr}"'+=("${_line}")'
  done < <("$@")
}

operator_read_config() {
  local root="${1:?root required}"
  "${root}/npa/.venv/bin/python" - <<'PY'
import sys, yaml
from pathlib import Path

path = Path.home() / ".npa" / "config.yaml"
if not path.exists():
    print("MISSING", file=sys.stderr)
    sys.exit(1)
cfg = yaml.safe_load(path.read_text()) or {}
storage = cfg.get("storage") or {}
bucket = str(storage.get("bucket", "")).replace("s3://", "").split("/")[0]
endpoint = storage.get("endpoint_url", "https://storage.eu-north1.nebius.cloud")
registry = str(storage.get("registry", cfg.get("registry", ""))).rstrip("/")
k8s_context = str(storage.get("k8s_context", "") or "")
if not k8s_context:
    for proj in (cfg.get("projects") or {}).values():
        if isinstance(proj, dict) and proj.get("k8s_context"):
            k8s_context = str(proj["k8s_context"])
            break
print(bucket)
print(endpoint)
print(registry)
print(k8s_context)
PY
}

operator_kubeconfig_path() {
  local ctx="${1:?context required}"
  if [ -n "${KUBECONFIG:-}" ] && [ -f "${KUBECONFIG}" ]; then
    echo "${KUBECONFIG}"
    return
  fi
  local path="${HOME}/.npa/clusters/${ctx}/kubeconfig"
  echo "${path}"
}
