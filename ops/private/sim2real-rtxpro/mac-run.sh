#!/usr/bin/env bash
# Back-compat alias — use operator-run.sh
exec bash "$(cd "$(dirname "$0")" && pwd)/operator-run.sh" "$@"
