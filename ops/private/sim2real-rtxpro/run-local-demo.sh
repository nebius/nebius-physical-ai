#!/usr/bin/env bash
# Customer demo entrypoint — cluster on Nebius, laptop = sync + Rerun only.
exec "$(cd "$(dirname "$0")" && pwd)/run-demo.sh" "$@"
