#!/bin/sh
set -eu

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

exec /bin/bash
