#!/usr/bin/env bash
# Bootstrap the dedicated non-root setup interpreter used by the controller.
set -euo pipefail

base_python=${1:?usage: setup-venv.sh BASE_PYTHON SETUP_VENV_ROOT}
setup_venv=${2:?usage: setup-venv.sh BASE_PYTHON SETUP_VENV_ROOT}
readonly pyyaml_requirement='PyYAML==6.0.3'

case "$setup_venv" in
  /*) ;;
  *) printf '%s\n' 'setup venv root must be absolute' >&2; exit 64 ;;
esac
test -x "$base_python"

"$base_python" -m venv --system-site-packages "$setup_venv"
setup_python="$setup_venv/bin/python"
test -x "$setup_python"
"$setup_python" -m pip --version
"$setup_python" -m pip install --no-cache-dir "$pyyaml_requirement"
"$setup_python" -c 'import yaml; assert yaml.__version__ == "6.0.3"'
