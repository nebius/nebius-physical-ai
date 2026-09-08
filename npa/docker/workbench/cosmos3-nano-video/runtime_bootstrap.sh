#!/usr/bin/env bash
# Populate the operator's runtime cache before launching Nano or KubeRay.
set -euo pipefail

case "${NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE-YES}" in
  YES) export NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE=YES ;;
  NO) echo "ERROR: NVIDIA runtime delivery was explicitly declined" >&2; exit 78 ;;
  *) echo "ERROR: NVIDIA runtime delivery setting must be YES or NO" >&2; exit 78 ;;
esac

root=/opt/npa-cosmos3-serving/runtime
lock=/opt/npa/nano-requirements.lock
mkdir -p "${root}"
# KubeRay may start several commands against the same writable runtime volume.
exec 9>"${root}/.npa-bootstrap.lock"
flock 9
/opt/npa-cosmos3-serving/runtime_bootstrap.sh true
export PATH="${root}/venv/bin:${PATH}"
python /opt/npa-cosmos3-serving/prepare_guardrail_runtime.py
closure_sha="${NPA_COSMOS3_NANO_CLOSURE_SHA256:?missing pinned Nano closure checksum}"
echo "${closure_sha}  ${lock}" | sha256sum -c -
marker="${root}/nano-ready-${NPA_COSMOS3_CLOSURE_SHA256}-${closure_sha}"
if [[ ! -f "${marker}" || ! -x "${root}/venv/bin/ray" ]]; then
  python -m pip install --no-cache-dir --require-hashes --only-binary=:all: \
    --no-deps -r "${lock}"
  python -m pip check
  touch "${marker}"
fi
flock -u 9
exec 9>&-
exec "$@"
