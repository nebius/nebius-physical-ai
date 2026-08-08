#!/usr/bin/env bash
set -euo pipefail

wan-runtime health
wan-runtime version
/opt/wan-base/bin/python -c 'import wan; print(wan.__file__)'
set +e
wan-runtime ensure >/tmp/wan-out 2>/tmp/wan-err
rc=$?
set -e
test "$rc" = 78
grep -q "Nothing has been downloaded" /tmp/wan-err
test -z "$(find /workspace/.cache/npa/wan2-2/runtime -mindepth 1 -print -quit)"
