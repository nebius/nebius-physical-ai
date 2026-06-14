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

# Nebius mk8s kubeconfigs use exec auth via the Nebius CLI (path varies on Mac).
operator_find_nebius_cli() {
  local candidate
  if candidate="$(command -v nebius 2>/dev/null)"; then
    printf '%s\n' "${candidate}"
    return 0
  fi
  for candidate in \
    /opt/homebrew/bin/nebius \
    /usr/local/bin/nebius \
    "${HOME}/.nebius/bin/nebius"; do
    if [ -x "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

# Patch exec.command in kubeconfig when the bundled path is wrong for this machine.
operator_export_kubeconfig() {
  local ctx="${1:?context required}"
  local root="${2:-}"
  local src resolved nebius py

  if [ -n "${KUBECONFIG:-}" ] && [ -f "${KUBECONFIG}" ] && [ "${NPA_KUBECONFIG_PATCHED:-0}" != "1" ]; then
    export KUBECONFIG
    return 0
  fi

  src="$(operator_kubeconfig_path "${ctx}")"
  if [ ! -f "${src}" ]; then
    echo "ERROR: kubeconfig not found: ${src}" >&2
    return 1
  fi

  if ! grep -q 'command:.*nebius' "${src}" 2>/dev/null; then
    export KUBECONFIG="${src}"
    return 0
  fi

  if ! nebius="$(operator_find_nebius_cli)"; then
    cat >&2 <<'EOF'
ERROR: Nebius CLI not found — required for mk8s kubeconfig auth on Mac.

Install:
  curl -fsSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
  export PATH="${HOME}/.nebius/bin:${PATH}"

Then re-run. Ensure `nebius` is on PATH and profile `npa-mk8s` is configured:
  nebius mk8s cluster get-credentials --context npa-rtxpro-mk8s
EOF
    return 1
  fi

  resolved="${HOME}/.npa/clusters/${ctx}/kubeconfig.resolved"
  mkdir -p "${HOME}/.npa/clusters/${ctx}"

  if [ -n "${root}" ] && [ -x "${root}/npa/.venv/bin/python" ]; then
    py="${root}/npa/.venv/bin/python"
  elif command -v python3 >/dev/null; then
    py="python3"
  else
    echo "ERROR: python3 required to patch kubeconfig" >&2
    return 1
  fi

  "${py}" - "${src}" "${resolved}" "${nebius}" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, nebius = sys.argv[1:4]
cfg = yaml.safe_load(Path(src).read_text()) or {}
for entry in cfg.get("users") or []:
    user = entry.get("user") or {}
    exec_cfg = user.get("exec")
    if isinstance(exec_cfg, dict) and exec_cfg.get("command"):
        exec_cfg["command"] = nebius
Path(dst).write_text(yaml.safe_dump(cfg, default_flow_style=False), encoding="utf-8")
PY
  chmod 600 "${resolved}"
  export KUBECONFIG="${resolved}"
  export NPA_KUBECONFIG_PATCHED=1
}

# Export AWS_* for aws CLI S3 cleanup (reads ~/.npa/credentials.yaml).
operator_export_storage_env() {
  local root="${1:?root required}"
  eval "$("${root}/npa/.venv/bin/python" - <<'PY'
import os, shlex, sys, yaml
from pathlib import Path

path = Path.home() / ".npa" / "credentials.yaml"
if not path.exists():
    sys.exit(0)
creds = (yaml.safe_load(path.read_text()) or {}).get("storage") or {}
pairs = []
for key, env in (
    ("aws_access_key_id", "AWS_ACCESS_KEY_ID"),
    ("aws_secret_access_key", "AWS_SECRET_ACCESS_KEY"),
    ("endpoint_url", "AWS_ENDPOINT_URL"),
):
    val = str(creds.get(key) or "").strip()
    if val:
        pairs.append(f"export {env}={shlex.quote(val)}")
if pairs:
    print("\n".join(pairs))
PY
)"
}
