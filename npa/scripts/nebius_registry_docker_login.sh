#!/usr/bin/env bash
# Log Docker into the Nebius container registry using the registry tenant profile.
#
# The workbench registry (e00cm0vc6t09m0z5gw) lives in tenant arch-sandbox and is
# reachable via skypilot-sa (profiles agent-sa / agent-service). Serverless jobs
# for workbench-poc use profile npa-mk8s (NEBIUS_SERVERLESS_PROFILE) instead.
#
# Usage:
#   ./npa/scripts/nebius_registry_docker_login.sh
#   NEBIUS_REGISTRY_PROFILE=agent-service ./npa/scripts/nebius_registry_docker_login.sh
#   REGISTRY_HOST=cr.eu-north1.nebius.cloud ./npa/scripts/nebius_registry_docker_login.sh
set -euo pipefail

unset NEBIUS_IAM_TOKEN NPA_IAM_TOKEN 2>/dev/null || true

REGISTRY_HOST="${REGISTRY_HOST:-cr.eu-north1.nebius.cloud}"
NEBIUS_CLI="${NEBIUS_CLI:-nebius}"
PROFILE="${NEBIUS_REGISTRY_PROFILE:-agent-sa}"

profile_args=()
if [[ -n "${PROFILE}" ]]; then
  profile_args=(--profile "${PROFILE}")
fi

token="$("${NEBIUS_CLI}" "${profile_args[@]}" iam get-access-token)"
if [[ -z "${token}" ]]; then
  echo "ERROR: empty token from ${NEBIUS_CLI} ${profile_args[*]} iam get-access-token" >&2
  exit 1
fi

printf '%s\n' "${token}" | docker login "${REGISTRY_HOST}" -u iam --password-stdin
echo "docker login ok host=${REGISTRY_HOST} profile=${PROFILE:-default}"
