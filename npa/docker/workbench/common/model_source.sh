#!/usr/bin/env bash
# Stage reviewed upstream code without examples, model files, or Git credentials.
set -euo pipefail

source_url="$1"
source_ref="$2"
shift 2
source_dir="$(mktemp -d)"
trap 'rm -rf "$source_dir"' EXIT
GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout "$source_url" "$source_dir"
GIT_LFS_SKIP_SMUDGE=1 git -C "$source_dir" checkout --detach "$source_ref"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$source_ref"
mv /opt/byof /opt/npa/wan2-2/source
mkdir /opt/byof
for source_path in "$@"; do
  test -e "$source_dir/$source_path"
  cp -a "$source_dir/$source_path" /opt/byof/
done
/opt/wan-base/bin/python - "$source_url" "$source_ref" <<'PY'
import json
import sys
from pathlib import Path

Path('/opt/byof/npa_source_metadata.json').write_text(json.dumps({
    'source': 'oss-byof', 'repo': sys.argv[1], 'ref': sys.argv[2],
}, indent=2) + '\n')
PY
ln -s /workspace/.cache/npa/wan2-2/runtime/current/venv /opt/byof/.venv
