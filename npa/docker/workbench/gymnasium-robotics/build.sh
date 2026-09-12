#!/usr/bin/env bash
# Neutral entrypoint and pre-network image-build gate.
set -euo pipefail

SCRIPT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
IMAGE_ROOT="${NPA_GYMNASIUM_IMAGE_ROOT:-/opt/npa/gymnasium-robotics}"
if [ -f "$SCRIPT_ROOT/source-lock.json" ]; then
  DEFAULT_LOCK_ROOT="$SCRIPT_ROOT"
else
  DEFAULT_LOCK_ROOT="$IMAGE_ROOT"
fi
LOCK_ROOT="${NPA_GYMNASIUM_LOCK_ROOT:-$DEFAULT_LOCK_ROOT}"
CACHE_ROOT="${NPA_GYMNASIUM_RUNTIME_CACHE:-/workspace/.cache/npa/gymnasium-robotics}"

# Filled only by a manager-authorized legal-closure transaction. Keeping the
# array empty ensures a status-only lock edit cannot unlock apt network access.
BOOTSTRAP_APT_PACKAGES=()

verify_bootstrap_locks() {
  incomplete=()
  for name in apt-runtime.lock.json corresponding-source.lock.json; do
    if ! grep -Eq '"status"[[:space:]]*:[[:space:]]*"complete"' "$LOCK_ROOT/$name"; then
      incomplete+=("$name")
    fi
  done
  if [ "${#BOOTSTRAP_APT_PACKAGES[@]}" -eq 0 ]; then
    incomplete+=("trusted-bootstrap-package-vector")
  fi
  if [ "${#incomplete[@]}" -gt 0 ]; then
    printf 'Neutral bootstrap packaging refusal before network access: %s\n' \
      "${incomplete[*]}" >&2
    return 65
  fi
}

install_bootstrap() {
  verify_bootstrap_locks
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    "${BOOTSTRAP_APT_PACKAGES[@]}"
  rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*.deb
}

prepare_runtime() {
  exec python3 -I -B "$IMAGE_ROOT/runtime-bootstrap.py" \
    --manifest "$IMAGE_ROOT/source-lock.json" \
    --requirements "$IMAGE_ROOT/requirements.lock" \
    --cache-root "$CACHE_ROOT" \
    prepare "$@"
}

run_smoke() {
  exec python3 -I -B "$IMAGE_ROOT/runtime-bootstrap.py" \
    --manifest "$IMAGE_ROOT/source-lock.json" \
    --requirements "$IMAGE_ROOT/requirements.lock" \
    --cache-root "$CACHE_ROOT" \
    exec -- "$IMAGE_ROOT/capability_smoke.py" "$@"
}

case "${1:-}" in
  verify-bootstrap-locks)
    verify_bootstrap_locks
    ;;
  install-bootstrap)
    install_bootstrap
    ;;
  prepare-runtime)
    shift
    prepare_runtime "$@"
    ;;
  run-smoke)
    shift
    run_smoke "$@"
    ;;
  "")
    exec /bin/sh
    ;;
  *)
    exec "$@"
    ;;
esac
