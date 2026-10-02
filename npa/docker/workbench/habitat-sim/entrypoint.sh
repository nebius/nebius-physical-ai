#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  exec /bin/bash
fi

runtime_cache_root=""

cleanup_runtime_cache() {
  local status=$?
  trap - EXIT
  if [[ -n "$runtime_cache_root" ]]; then
    rm -rf -- "$runtime_cache_root" || true
  fi
  exit "$status"
}

runtime_fetch_optional_scientific_stack() {
  local lock_file=/usr/share/doc/npa-habitat-sim/requirements-runtime.lock
  local cache_parent lock_digest final_dir staging
  [[ -f "$lock_file" ]] || {
    printf '%s\n' 'runtime-fetch lock is missing' >&2
    return 78
  }
  [[ "$(id -u)" == 1000 ]] || {
    printf '%s\n' 'runtime-fetch requires the non-root workload user' >&2
    return 78
  }
  cache_parent="$(mktemp -d "${TMPDIR:-/tmp}/npa-habitat-runtime.XXXXXX")"
  runtime_cache_root="$cache_parent"
  chmod 0700 "$cache_parent"
  [[ "$(stat -c '%u:%a' "$cache_parent")" == "$(id -u):700" ]] || {
    printf '%s\n' 'runtime-fetch cache parent ownership is invalid' >&2
    return 78
  }
  lock_digest="$(sed -n 's/^# runtime-fetch: //p' "$lock_file" | sha256sum | cut -d' ' -f1)"
  [[ "$lock_digest" =~ ^[0-9a-f]{64}$ ]] || {
    printf '%s\n' 'runtime-fetch lock digest is invalid' >&2
    return 78
  }
  final_dir="$cache_parent/payload-$lock_digest"
  staging="$(mktemp -d "$cache_parent/.staging.XXXXXX")"
  chmod 0700 "$staging"
  sed -n 's/^# runtime-fetch: //p' "$lock_file" > "$staging/requirements.lock"
  [[ -s "$staging/requirements.lock" ]] || {
    printf '%s\n' 'runtime-fetch lock has no optional packages' >&2
    return 78
  }
  mkdir "$staging/payload"
  /opt/venv/bin/python -m pip install \
    --disable-pip-version-check --no-cache-dir --only-binary=:all: --no-deps \
    --require-hashes --target "$staging/payload" -r "$staging/requirements.lock"
  if find "$staging/payload" -type l -print -quit | grep -q .; then
    printf '%s\n' 'runtime-fetch payload contains a symlink' >&2
    return 78
  fi
  (
    cd "$staging/payload"
    while IFS= read -r -d '' member; do
      [[ -f "$member" ]] || exit 78
      sha256sum -- "$member"
    done < <(find . -type f -print0 | sort -z)
  ) > "$staging/payload.sha256"
  (
    cd "$staging/payload"
    sha256sum -c ../payload.sha256 >/dev/null
  ) || {
    printf '%s\n' 'runtime-fetch payload integrity verification failed' >&2
    return 78
  }
  printf '%s\n' "$lock_digest" > "$staging/.ready"
  chmod 0600 "$staging/.ready"
  [[ ! -e "$final_dir" && ! -L "$final_dir" ]] || {
    printf '%s\n' 'runtime-fetch cache destination unexpectedly exists' >&2
    return 78
  }
  mv -- "$staging" "$final_dir"
  chmod 0700 "$final_dir"
  export PYTHONPATH="$final_dir/payload${PYTHONPATH:+:$PYTHONPATH}"
}

if [[ "$(basename -- "$1")" == python* && "${2:-}" == "-m" &&
      "${3:-}" == "npa.workflows.habitat_sim_smoke" ]]; then
  skip_runtime_fetch=false
  for argument in "$@"; do
    [[ "$argument" == "--help" ]] && skip_runtime_fetch=true
  done
  if [[ "$skip_runtime_fetch" == false ]]; then
    trap cleanup_runtime_cache EXIT
    runtime_fetch_optional_scientific_stack
    "$@"
    status=$?
    exit "$status"
  fi
fi

exec "$@"
