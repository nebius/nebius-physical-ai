#!/bin/sh
set -eu

umask 077
mkdir -p /workspace/.cache/npa/libero /workspace/byof-runs
exec "$@"
