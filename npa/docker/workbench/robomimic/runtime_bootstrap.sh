#!/usr/bin/env bash
set -euo pipefail

readonly runtime_root="${NPA_ROBOMIMIC_RUNTIME_ROOT:-/opt/npa-runtime/robomimic}"
readonly verifier="/opt/npa/robomimic/verify_image.py"

case "${1:-}" in
  verify)
    exec /usr/local/bin/python3 "${verifier}" runtime --runtime-root "${runtime_root}"
    ;;
  exec)
    shift
    /usr/local/bin/python3 "${verifier}" runtime --runtime-root "${runtime_root}" >/dev/null
    exec "${runtime_root}/payload/bin/python" "$@"
    ;;
  assert-refusal)
    empty_root="$(mktemp -d)"
    trap 'rm -rf -- "${empty_root}"' EXIT
    set +e
    /usr/local/bin/python3 "${verifier}" runtime --runtime-root "${empty_root}" \
      >"${empty_root}.stdout" 2>"${empty_root}.stderr"
    status=$?
    set -e
    if [[ "${status}" -ne 78 ]]; then
      echo "expected missing-runtime refusal status 78; got ${status}" >&2
      exit 1
    fi
    if find "${empty_root}" -mindepth 1 -print -quit | grep -q .; then
      echo "runtime verifier mutated the missing runtime root" >&2
      exit 1
    fi
    rm -f -- "${empty_root}.stdout" "${empty_root}.stderr"
    echo "NPA_ROBOMIMIC_RUNTIME_REFUSAL_OK"
    ;;
  *)
    echo "usage: robomimic-runtime {verify|exec|assert-refusal}" >&2
    exit 64
    ;;
esac
