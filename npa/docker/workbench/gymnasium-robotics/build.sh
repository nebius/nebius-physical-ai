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
  "libegl1=1.7.0-1build1"
  "libopengl0=1.7.0-1build1"
  "openssh-server=1:9.6p1-3ubuntu13.19"
  "python3-boto3=1.34.46+dfsg-1ubuntu1"
  "python3.12=3.12.3-1ubuntu0.16"
  "python3.12-venv=3.12.3-1ubuntu0.16"
  "rsync=3.2.7-1ubuntu1.5"
  "sudo=1.9.15p5-3ubuntu5.24.04.2"
)
EXPECTED_APT_LOCK_SHA256="edcfbf171c6d13b2bfddabc596c0adc29bda8bebd696ec497be6265d863c28b2"
EXPECTED_CORRESPONDING_SOURCE_LOCK_SHA256="3c238567f6acebd2a393037a05803b2dee6be350808f81bfdcaca47c1cad957f"
EXPECTED_RUNTIME_FETCH_MANIFEST_SHA256="72597bda8363f54ef1d243d6f8b2a7660aebdaf54875fd161b11b0097444f957"
EXPECTED_FINAL_PACKAGE_MANIFEST_SHA256="a6aa62688771732686c8f74ac73487f0532c4c238d874ad3ce99df47590d4b7e"
SNAPSHOT_URL="https://snapshot.ubuntu.com/ubuntu/20260905T000000Z"

verify_bootstrap_locks() {
  incomplete=()
  for name in apt-runtime.lock.json corresponding-source.lock.json runtime-fetch-manifest.json; do
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
  if [ "$(sha256sum "$LOCK_ROOT/runtime-fetch-manifest.json" | cut -d' ' -f1)" != "$EXPECTED_RUNTIME_FETCH_MANIFEST_SHA256" ]; then
    incomplete+=("reviewed-runtime-fetch-manifest-bytes")
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
  # Build timestamps and inode caches are not runtime inputs. Normalize them
  # before this layer commits so independent rebuilds have identical contents.
  truncate -s 0 /var/log/apt/history.log /var/log/apt/term.log \
    /var/log/dpkg.log /var/cache/ldconfig/aux-cache
  python3 - <<'PY'
from pathlib import Path

shadow = Path("/etc/shadow")
records = [line.split(":") for line in shadow.read_text().splitlines()]
for record in records:
    if record[0] == "sshd":
        record[2] = "0"
shadow.write_text("\n".join(":".join(record) for record in records) + "\n")
PY
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
