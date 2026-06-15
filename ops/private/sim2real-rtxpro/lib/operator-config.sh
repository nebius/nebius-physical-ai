#!/usr/bin/env bash
# Resolve operator config from ~/.npa/config.yaml (no secrets, no hardcoded tenant IDs).

# Resolve public nebius-physical-ai checkout (Layout A: sibling under demo root; Layout B: in-repo ops/).
npa_repo_root() {
  local start="${1:-.}"
  local dir
  if [ -n "${NPA_CHECKOUT:-}" ] && [ -f "${NPA_CHECKOUT}/npa/pyproject.toml" ]; then
    printf '%s\n' "${NPA_CHECKOUT}"
    return 0
  fi
  if [ -n "${NPA_SIM2REAL_REPO:-}" ] && [ -f "${NPA_SIM2REAL_REPO}/npa/pyproject.toml" ]; then
    printf '%s\n' "${NPA_SIM2REAL_REPO}"
    return 0
  fi
  dir="$(cd "${start}" && pwd)"
  while [ "${dir}" != "/" ]; do
    if [ -f "${dir}/npa/pyproject.toml" ]; then
      printf '%s\n' "${dir}"
      return 0
    fi
    if [ -f "${dir}/nebius-physical-ai/npa/pyproject.toml" ]; then
      printf '%s\n' "${dir}/nebius-physical-ai"
      return 0
    fi
    dir="$(dirname "${dir}")"
  done
  echo "ERROR: repo root not found (run ./setup.sh or set NPA_CHECKOUT)" >&2
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

# Canonical staged run id (accept timestamp-only, job name, or duplicated prefixes).
operator_normalize_staged_run_id() {
  local rid="${1:?run id required}"
  case "${rid}" in
    sim2real-staged-*) printf '%s\n' "${rid}"; return 0 ;;
  esac
  rid="${rid#sim2real-}"
  case "${rid}" in
    sim2real-staged-*) printf '%s\n' "${rid}"; return 0 ;;
    staged-*) printf '%s\n' "sim2real-${rid}"; return 0 ;;
    *) printf '%s\n' "sim2real-staged-${rid}" ;;
  esac
}

# Orchestrator Job name for a staged run id.
operator_orchestrator_job_name() {
  local run_id
  run_id="$(operator_normalize_staged_run_id "$1")"
  printf 'sim2real-%s\n' "${run_id}"
}

# Print kubectl wide lines for s2r-* sibling Jobs (label match + name slug fallback).
operator_list_sibling_jobs() {
  local ctx="${1:?context required}"
  local ns="${2:?namespace required}"
  local run_id="${3:?run id required}"
  local root="${4:?root required}"
  local py="${root}/npa/.venv/bin/python"
  if [ ! -x "${py}" ]; then
    py="python3"
  fi
  kubectl --context "${ctx}" get jobs -n "${ns}" -o json 2>/dev/null \
    | RUN_ID="${run_id}" "${py}" - <<'PY'
import json, os, sys

def safe_slug(value: str) -> str:
    chars = [c.lower() if c.isalnum() else "-" for c in str(value)]
    return "-".join(part for part in "".join(chars).split("-") if part)

run_id = os.environ["RUN_ID"]
run_label = (safe_slug(run_id)[:63] or "run").rstrip("-")
run_part = safe_slug(run_id)[:22] or "run"
raw = sys.stdin.read().strip()
if not raw:
    sys.exit(0)
data = json.loads(raw)
rows = []
for item in data.get("items", []):
    name = str((item.get("metadata") or {}).get("name") or "")
    if not name.startswith("s2r-"):
        continue
    labels = (item.get("metadata") or {}).get("labels") or {}
    label_run = str(labels.get("sim2real.local/run-id") or "")
    if label_run and label_run != run_label and run_part not in name:
        continue
    if not label_run and run_part not in name:
        continue
    status = item.get("status") or {}
    rows.append(
        {
            "name": name,
            "active": int(status.get("active") or 0),
            "succeeded": int(status.get("succeeded") or 0),
            "failed": int(status.get("failed") or 0),
        }
    )
if not rows:
    sys.exit(0)
for row in sorted(rows, key=lambda r: r["name"]):
    print(
        f"{row['name']:<56} "
        f"active={row['active']} "
        f"succeeded={row['succeeded']} "
        f"failed={row['failed']}"
    )
PY
}
