#!/usr/bin/env bash
# Keep the SkyPilot worker idle, or launch a command after runtime admission.
set -euo pipefail

if [[ $# -eq 0 ]]; then
  exec sleep infinity
fi
if [[ "$1" != --runtime ]]; then
  exec "$@"
fi
shift
if [[ $# -eq 0 ]]; then
  echo "ERROR: --runtime requires a workload command" >&2
  exit 2
fi
case "${NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE-YES}" in
  YES) export NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE=YES ;;
  NO) echo "ERROR: NVIDIA runtime delivery was explicitly declined" >&2; exit 78 ;;
  *) echo "ERROR: NVIDIA runtime delivery setting must be YES or NO" >&2; exit 78 ;;
esac
root=/opt/npa-cosmos3-serving/runtime
mkdir -p "${root}"
exec 9>"${root}/.npa-bootstrap.lock"
flock 9
/opt/npa-cosmos3-serving/runtime_bootstrap.sh true
export PATH="${root}/venv/bin:${PATH}"
python /opt/npa-cosmos3-serving/prepare_guardrail_runtime.py
flock -u 9
exec 9>&-
exec "$@"
