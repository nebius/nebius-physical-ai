#!/bin/sh
set -eu

umask 077
mkdir -p /workspace/.cache/npa/libero /workspace/byof-runs
chmod 1770 /workspace/byof-runs
exec "$@"
