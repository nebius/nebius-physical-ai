#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  exec /bin/bash
fi

runtime_fetch_optional_scientific_stack() {
  local lock_file=/usr/share/doc/npa-habitat-sim/requirements-runtime.lock
  local cache_parent="${HOME}/.cache/npa-habitat-runtime"
  local lock_digest final_dir staging
  [[ -f "$lock_file" ]] || {
    printf '%s\n' 'runtime-fetch lock is missing' >&2
    return 78
  }
  [[ "$(id -u)" == 1000 ]] || {
    printf '%s\n' 'runtime-fetch requires the non-root workload user' >&2
    return 78
  }
  mkdir -p "$cache_parent"
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
  if [[ -e "$final_dir" || -L "$final_dir" ]]; then
    [[ -d "$final_dir/payload" && -f "$final_dir/.ready" ]] || {
      printf '%s\n' 'runtime-fetch cache entry is incomplete' >&2
      return 78
    }
    [[ "$(cat "$final_dir/.ready")" == "$lock_digest" ]] || {
      printf '%s\n' 'runtime-fetch cache lock digest mismatch' >&2
      return 78
    }
    [[ "$(stat -c '%u:%a' "$final_dir")" == "$(id -u):700" ]] || {
      printf '%s\n' 'runtime-fetch cache entry ownership is invalid' >&2
      return 78
    }
    export PYTHONPATH="$final_dir/payload${PYTHONPATH:+:$PYTHONPATH}"
    return 0
  fi
  staging="$(mktemp -d "$cache_parent/.staging.XXXXXX")"
  chmod 0700 "$staging"
  cleanup_staging() {
    local status=$?
    rm -rf -- "$staging" || true
    return "$status"
  }
  trap cleanup_staging RETURN
  sed -n 's/^# runtime-fetch: //p' "$lock_file" > "$staging/requirements.lock"
  [[ -s "$staging/requirements.lock" ]] || {
    printf '%s\n' 'runtime-fetch lock has no optional packages' >&2
    return 78
  }
  mkdir "$staging/payload"
  /opt/venv/bin/python -m pip install \
    --disable-pip-version-check --no-cache-dir --only-binary=:all: --no-deps \
    --require-hashes --target "$staging/payload" -r "$staging/requirements.lock"
  printf '%s\n' "$lock_digest" > "$staging/.ready"
  chmod 0600 "$staging/.ready"
  if ! mkdir "$final_dir"; then
    [[ -d "$final_dir/payload" && -f "$final_dir/.ready" ]] || {
      printf '%s\n' 'runtime-fetch cache claim collided with an invalid entry' >&2
      return 78
    }
    [[ "$(cat "$final_dir/.ready")" == "$lock_digest" ]] || {
      printf '%s\n' 'runtime-fetch concurrent cache digest mismatch' >&2
      return 78
    }
  else
    mv "$staging/payload" "$final_dir/payload"
    mv "$staging/.ready" "$final_dir/.ready"
    chmod 0700 "$final_dir"
  fi
  export PYTHONPATH="$final_dir/payload${PYTHONPATH:+:$PYTHONPATH}"
}

if [[ "$(basename -- "$1")" == python* && "${2:-}" == "-m" &&
      "${3:-}" == "npa.workflows.habitat_sim_smoke" ]]; then
  skip_runtime_fetch=false
  for argument in "$@"; do
    [[ "$argument" == "--help" ]] && skip_runtime_fetch=true
  done
  if [[ "$skip_runtime_fetch" == false ]]; then
    runtime_fetch_optional_scientific_stack
  fi
fi

exec "$@"
