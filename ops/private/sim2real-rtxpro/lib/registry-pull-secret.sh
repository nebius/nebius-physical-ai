#!/usr/bin/env bash
# Refresh npa-nebius-registry before K8s sim2real submits (no npa package imports).

registry_refresh_nebius_pull_secret() {
  local registry_server="${1:?registry host required}"
  local k8s_context="${2:-}"
  local namespace="${3:-default}"
  local secret_name="${4:-npa-nebius-registry}"

  registry_server="${registry_server#https://}"
  registry_server="${registry_server#http://}"
  registry_server="${registry_server%/}"

  case "${registry_server}" in
    cr.*.nebius.cloud) ;;
    *)
      echo "Skip registry refresh (not Nebius CR): ${registry_server}" >&2
      return 0
      ;;
  esac

  local nebius=""
  if nebius="$(command -v nebius 2>/dev/null)"; then
    :
  elif [ -x "${HOME}/.nebius/bin/nebius" ]; then
    nebius="${HOME}/.nebius/bin/nebius"
  elif [ -x /opt/homebrew/bin/nebius ]; then
    nebius=/opt/homebrew/bin/nebius
  else
    echo "WARN: nebius CLI not found — skip registry pull-secret refresh" >&2
    return 0
  fi

  if ! command -v kubectl >/dev/null 2>&1; then
    echo "WARN: kubectl not found — skip registry pull-secret refresh" >&2
    return 0
  fi

  local py=""
  if [ -n "${NPA_SIM2REAL_REPO:-}" ] && [ -x "${NPA_SIM2REAL_REPO}/npa/.venv/bin/python" ]; then
    py="${NPA_SIM2REAL_REPO}/npa/.venv/bin/python"
  elif command -v python3 >/dev/null; then
    py=python3
  else
    py=/usr/bin/python3
  fi

  "${py}" - "${registry_server}" "${secret_name}" "${namespace}" "${k8s_context}" "${nebius}" <<'PY'
import base64
import json
import os
import subprocess
import sys

server, secret_name, namespace, k8s_context, nebius = sys.argv[1:6]
proc = subprocess.run(
    [nebius, "iam", "get-access-token"],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    timeout=30,
    check=False,
)
token = proc.stdout.strip()
if proc.returncode != 0 or not token:
    detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
    raise SystemExit(f"nebius iam get-access-token failed: {detail}")

username = "iam"
auth = base64.b64encode(f"{username}:{token}".encode()).decode("ascii")
payload = {
    "apiVersion": "v1",
    "kind": "Secret",
    "metadata": {"name": secret_name, "namespace": namespace},
    "type": "kubernetes.io/dockerconfigjson",
    "data": {
        ".dockerconfigjson": base64.b64encode(
            json.dumps(
                {
                    "auths": {
                        server: {
                            "username": username,
                            "password": token,
                            "auth": auth,
                        }
                    }
                }
            ).encode()
        ).decode("ascii"),
    },
}
cmd = ["kubectl"]
if k8s_context:
    cmd.extend(["--context", k8s_context])
cmd.extend(["-n", namespace, "apply", "-f", "-"])
env = dict(os.environ)
apply = subprocess.run(
    cmd,
    input=json.dumps(payload),
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env=env,
    check=False,
)
if apply.returncode != 0:
    detail = apply.stderr.strip() or apply.stdout.strip() or f"exit {apply.returncode}"
    raise SystemExit(f"kubectl apply registry secret failed: {detail}")
print(f"Refreshed {secret_name} for {server}")
PY
}

registry_server_from_image() {
  local image="${1:-}"
  image="${image#docker:}"
  if [[ "${image}" != */* ]]; then
    return 1
  fi
  local host="${image%%/*}"
  if [[ "${host}" == *.* || "${host}" == *:* || "${host}" == localhost ]]; then
    printf '%s\n' "${host}"
    return 0
  fi
  return 1
}

registry_refresh_for_images() {
  local k8s_context="${1:-}"
  shift
  local image server
  for image in "$@"; do
    if server="$(registry_server_from_image "${image}")"; then
      registry_refresh_nebius_pull_secret "${server}" "${k8s_context}"
      return 0
    fi
  done
}
