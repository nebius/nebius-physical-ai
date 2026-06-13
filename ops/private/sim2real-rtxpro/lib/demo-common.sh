#!/usr/bin/env bash
# Shared helpers for sim2real customer demo scripts (source, do not execute).
# shellcheck source=operator-config.sh
_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=operator-config.sh
source "${_LIB_DIR}/operator-config.sh"

demo_common_root() {
  npa_repo_root "${_LIB_DIR}"
}

demo_bootstrap_venv() {
  local root="$1"
  local py="${root}/npa/.venv/bin/python"
  if [ ! -f "${root}/npa/pyproject.toml" ]; then
    echo "ERROR: invalid repo root: ${root} (missing npa/pyproject.toml)" >&2
    echo "       Run from the nebius-physical-ai checkout, not ops/ alone." >&2
    return 1
  fi
  if [ -x "${py}" ]; then
    return 0
  fi
  echo "Creating npa/.venv (first run — may take a few minutes on Mac)..."
  if ! command -v python3 >/dev/null; then
    echo "python3 not found — install Python 3.10+ (brew install python@3.12)." >&2
    return 1
  fi
  python3 -m venv "${root}/npa/.venv"
  "${root}/npa/.venv/bin/python" -m pip install -U pip -q
  "${root}/npa/.venv/bin/python" -m pip install -e "${root}/npa" -q
  echo "Installed npa into ${root}/npa/.venv"
}

demo_read_storage_config() {
  local root="$1"
  "${root}/npa/.venv/bin/python" - <<'PY'
import sys, yaml
from pathlib import Path

cfg_path = Path.home() / ".npa" / "config.yaml"
if not cfg_path.exists():
    print("MISSING_CONFIG", file=sys.stderr)
    sys.exit(1)
cfg = yaml.safe_load(cfg_path.read_text()) or {}
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

demo_preflight() {
  local root="$1"
  local py="${root}/npa/.venv/bin/python"
  local missing=0
  local k8s_context="${KUBECONTEXT:-}"

  if [ ! -f "${HOME}/.npa/config.yaml" ]; then
    echo "ERROR: ~/.npa/config.yaml missing — run: npa configure" >&2
    missing=1
  fi
  if [ ! -f "${HOME}/.npa/credentials.yaml" ]; then
    echo "ERROR: ~/.npa/credentials.yaml missing — run: npa configure" >&2
    missing=1
  fi
  if [ -z "${k8s_context}" ] && [ -f "${HOME}/.npa/config.yaml" ]; then
    _ctx_cfg=()
    while IFS= read -r _line; do
      _ctx_cfg+=("${_line}")
    done < <(demo_read_storage_config "${root}" 2>/dev/null || true)
    k8s_context="${_ctx_cfg[3]:-}"
  fi
  if [ -z "${k8s_context}" ]; then
    echo "ERROR: k8s_context not set — add storage.k8s_context to ~/.npa/config.yaml" >&2
    missing=1
  fi
  local kubeconfig="${KUBECONFIG:-${HOME}/.npa/clusters/${k8s_context}/kubeconfig}"
  if [ ! -f "${kubeconfig}" ]; then
    echo "ERROR: kubeconfig not found: ${kubeconfig}" >&2
    echo "       Cluster runs on Nebius; laptop needs kubeconfig to submit/monitor." >&2
    missing=1
  fi
  if ! command -v kubectl >/dev/null; then
    echo "ERROR: kubectl not on PATH" >&2
    missing=1
  fi
  if ! "${py}" -c "import rerun" 2>/dev/null; then
    echo "ERROR: rerun-sdk not installed — re-run script to bootstrap venv" >&2
    missing=1
  fi
  return "${missing}"
}

demo_visualize_rrd() {
  local rrd="$1"
  local log_dir="$2"
  local run_id="$3"
  local root="$4"
  local rerun="${root}/npa/.venv/bin/rerun"
  local bind="${RERUN_BIND:-127.0.0.1}"
  local port="${RERUN_WEB_PORT:-}"
  local rerun_log="${log_dir}/rerun-${run_id}.log"

  if [ ! -f "${rrd}" ] || [ ! -s "${rrd}" ]; then
    echo "ERROR: Rerun recording missing: ${rrd}" >&2
    return 1
  fi

  if [ "${VISUALIZE:-1}" != "1" ]; then
    echo "VISUALIZE=0 — open manually: ${rerun} ${rrd} --web-viewer"
    return 0
  fi

  if [ "${OPEN_RERUN:-0}" = "1" ] && { [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; }; then
    echo "Opening native Rerun viewer..."
    exec "${rerun}" "${rrd}"
  fi

  if [ -z "${port}" ]; then
    port="$("${root}/npa/.venv/bin/python" -c \
      'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')"
  fi

  pkill -f "rerun.*${rrd}" 2>/dev/null || true
  nohup "${rerun}" "${rrd}" --web-viewer --web-viewer-port "${port}" --bind "${bind}" \
    > "${rerun_log}" 2>&1 &
  local pid=$!
  local url=""
  for _ in $(seq 1 25); do
    url="$(grep -oE 'Hosting a web-viewer at http[^[:space:]]+' "${rerun_log}" 2>/dev/null \
      | sed 's/Hosting a web-viewer at //' | head -1 || true)"
    [ -n "${url}" ] && break
    sleep 0.5
  done
  [ -z "${url}" ] && url="http://${bind}:${port}/"

  echo ""
  echo "=== Rerun (local viewer — Nebius run artifacts) ==="
  echo "  ${url}"
  echo "  pid=${pid}  log=${rerun_log}"
  echo ""
  echo "Walkthrough (~30 s): rollouts/camera → critique → signal/reward → heldout/scores"
  echo "Stop: kill ${pid}"
}
