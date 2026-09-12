#!/bin/sh
set -eu

runtime_root=${LIBERO_RUNTIME_ROOT:?LIBERO_RUNTIME_ROOT is required}
test -x "$runtime_root/venv/bin/python"
test -f "$runtime_root/.complete.json"
test ! -e "$runtime_root/source/libero/libero/assets"
test ! -e "$runtime_root/source/.git"
export BYOF_REPO_ROOT="$runtime_root/source"
export LIBERO_CONFIG_PATH="$runtime_root/libero-config"
mkdir -p "$LIBERO_CONFIG_PATH"
printf '%s\n' \
  "benchmark_root: $runtime_root/source/libero/libero" \
  "bddl_files: $runtime_root/source/libero/libero/bddl_files" \
  "init_states: $runtime_root/source/libero/libero/init_files" \
  "datasets: $runtime_root/data" \
  "assets: $runtime_root/source/libero/libero/assets" \
  > "$LIBERO_CONFIG_PATH/config.yaml"
exec "$runtime_root/venv/bin/python" /opt/npa/libero/libero_smoke.py
