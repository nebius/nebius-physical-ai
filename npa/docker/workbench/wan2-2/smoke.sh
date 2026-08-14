#!/usr/bin/env bash
set -euo pipefail

wan-runtime health
wan-runtime version
/opt/wan-base/bin/python /opt/npa/wan2-2/dependency_closure.py verify-report \
  --baked-inventory /opt/byof/npa_baked_python_inventory.txt \
  --runtime-requirements /opt/npa/wan2-2/runtime-requirements.txt \
  --report /opt/byof/npa_dependency_closure_report.json
PYTHONPATH=/opt/byof /opt/wan-base/bin/python - <<'PY'
import importlib.util
import runpy

from npa_wan_input_contract import resolve_wan_input_contract

assert importlib.util.find_spec("wan") is not None
assert importlib.util.find_spec("torch") is None
contract = resolve_wan_input_contract(
    context_image_uri="",
    declared_capability="wan2.2_ti2v_5b_text_to_video",
    declared_artifact="wan2_2_ti2v_5b_text_to_video.json",
)
assert contract.task == "text-to-video"
easy_dict = runpy.run_path("/opt/byof/wan/utils/easydict.py")["EasyDict"]
assert easy_dict({"nested": {"value": 7}}).nested.value == 7
PY
test -n "$(find /opt/byof/wan -type f -path '*/__pycache__/*.pyc' -print -quit)"
set +e
wan-runtime ensure >/tmp/wan-out 2>/tmp/wan-err
rc=$?
set -e
test "$rc" = 78
grep -q "Nothing has been downloaded" /tmp/wan-err
test -z "$(find /workspace/.cache/npa/wan2-2/runtime -mindepth 1 -print -quit)"
