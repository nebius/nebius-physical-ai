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

# Exact direct package vector resolved from the signed immutable Ubuntu
# snapshot. The final installed graph is checked below as a second boundary.
BOOTSTRAP_APT_PACKAGES=(
  "ca-certificates=20260601~24.04.1"
  "openssh-server=1:9.6p1-3ubuntu13.19"
  "python3-boto3=1.34.46+dfsg-1ubuntu1"
  "python3.12=3.12.3-1ubuntu0.16"
  "python3.12-venv=3.12.3-1ubuntu0.16"
  "rsync=3.2.7-1ubuntu1.5"
  "sudo=1.9.15p5-3ubuntu5.24.04.2"
)
EXPECTED_APT_LOCK_SHA256="6e1df9be2187010e9d4ee12dc2a4d95e4f0aa799ff321c70d86ec2d8772b855e"
EXPECTED_CORRESPONDING_SOURCE_LOCK_SHA256="7a097851d8c9eae45bb663d7d8d989f507afc0fcdc12e721d7431dd27aa9a3be"
EXPECTED_FINAL_PACKAGE_MANIFEST_SHA256="e5e25fc8ac5570ebf90aa6a73d71243d611fba80d90e49ef2c9485bdc647a505"
SNAPSHOT_URL="https://snapshot.ubuntu.com/ubuntu/20260905T000000Z"

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
  if [ "$(sha256sum "$LOCK_ROOT/apt-runtime.lock.json" | cut -d' ' -f1)" != "$EXPECTED_APT_LOCK_SHA256" ]; then
    incomplete+=("reviewed-apt-lock-bytes")
  fi
  if [ "$(sha256sum "$LOCK_ROOT/corresponding-source.lock.json" | cut -d' ' -f1)" != "$EXPECTED_CORRESPONDING_SOURCE_LOCK_SHA256" ]; then
    incomplete+=("reviewed-corresponding-source-lock-bytes")
  fi
  if ! grep -Fq '"url": "https://snapshot.ubuntu.com/ubuntu/20260905T000000Z"' \
    "$LOCK_ROOT/apt-runtime.lock.json"; then
    incomplete+=("immutable-ubuntu-snapshot")
  fi
  if [ "${#incomplete[@]}" -gt 0 ]; then
    printf 'Neutral bootstrap packaging refusal before network access: %s\n' \
      "${incomplete[*]}" >&2
    return 65
  fi
}

install_bootstrap() {
  verify_bootstrap_locks
  ca_bundle=/run/npa-host-ca-bundle.crt
  if [ ! -r "$ca_bundle" ] || [ ! -s "$ca_bundle" ]; then
    printf 'Neutral bootstrap packaging refusal: missing ephemeral TLS trust input\n' >&2
    return 65
  fi
  cat > /etc/apt/sources.list.d/ubuntu.sources <<EOF
Types: deb
URIs: $SNAPSHOT_URL
Suites: noble noble-updates noble-backports noble-security
Components: main universe
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
Check-Valid-Until: no
EOF
  rm -f /etc/apt/sources.list
  cat > /etc/apt/apt.conf.d/99npa-snapshot-tls <<EOF
Acquire::https::CaInfo "$ca_bundle";
Acquire::Retries "5";
EOF
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    "${BOOTSTRAP_APT_PACKAGES[@]}"
  observed=$(mktemp)
  dpkg-query -W \
    -f='${binary:Package}\t${Version}\t${source:Package}\t${source:Version}\t${Architecture}\n' \
    | LC_ALL=C sort > "$observed"
  observed_sha256=$(sha256sum "$observed" | cut -d' ' -f1)
  rm -f "$observed"
  if [ "$observed_sha256" != "$EXPECTED_FINAL_PACKAGE_MANIFEST_SHA256" ]; then
    printf 'Neutral bootstrap packaging refusal: installed package graph changed\n' >&2
    return 65
  fi
  rm -f /etc/apt/apt.conf.d/99npa-snapshot-tls /etc/ssh/ssh_host_*
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
