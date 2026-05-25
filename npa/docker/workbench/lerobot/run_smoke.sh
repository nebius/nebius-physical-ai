#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

VERSION="$(
  cd "$NPA_ROOT"
  python3 - <<'PY'
from pathlib import Path
import re

try:
    import tomllib
except ModuleNotFoundError:
    text = Path("pyproject.toml").read_text()
    section = text.split("[tool.npa.supported-tools]", 1)[1]
    match = re.search(r'^lerobot\s*=\s*"([^"]+)"', section, re.MULTILINE)
    if not match:
        raise SystemExit("Could not find [tool.npa.supported-tools].lerobot")
    print(match.group(1))
else:
    with Path("pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    print(data["tool"]["npa"]["supported-tools"]["lerobot"])
PY
)"

IMAGE="${1:-npa-lerobot:${VERSION}}"

set +e
docker run --rm --gpus all --entrypoint /bin/bash "$IMAGE" -lc '
set +e
python -m npa.smoke.test_lerobot_env
env_code=$?
python -m npa.smoke.test_lerobot_functional
functional_code=$?

echo "ENV_SMOKE_EXIT_CODE=${env_code}"
echo "FUNCTIONAL_SMOKE_EXIT_CODE=${functional_code}"

if [ "$env_code" -ne 0 ]; then
  exit "$env_code"
fi
exit "$functional_code"
'
code=$?
set -e

if [ "$code" -eq 0 ]; then
  echo "Smoke tests passed for $IMAGE"
else
  echo "Smoke tests failed for $IMAGE (exit $code)" >&2
fi
exit "$code"
