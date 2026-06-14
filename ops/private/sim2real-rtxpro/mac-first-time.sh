#!/usr/bin/env bash
# Back-compat alias — use first-time-setup.sh
exec bash "$(cd "$(dirname "$0")" && pwd)/first-time-setup.sh" "$@"
