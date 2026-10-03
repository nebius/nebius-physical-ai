"""Prove the standalone SkyPilot bridge retains the shared scalar contract."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from npa.burst import _sky_api
from npa.literal_values import require_boolean


@pytest.mark.parametrize(
    "value", [True, False, None, 0, 1, 0.0, "true", "false", [], {}]
)
def test_standalone_boolean_adapter_matches_shared_contract(value):
    if type(value) is bool:
        assert _sky_api._json_boolean({"flag": value}, "flag", default=False) is (
            require_boolean(value, field="flag")
        )
        return
    with pytest.raises(ValueError, match="flag must be a literal boolean"):
        require_boolean(value, field="flag")
    with pytest.raises(ValueError, match="flag must be a JSON boolean"):
        _sky_api._json_boolean({"flag": value}, "flag", default=False)


@pytest.mark.parametrize("action,flag", [("queue", "refresh"), ("logs", "follow")])
@pytest.mark.parametrize("value", [None, "false", 1])
def test_bridge_runs_without_installed_npa_and_refuses_before_sky_calls(
    tmp_path, action, flag, value
):
    # -S excludes installed packages; only the fake SkyPilot module is supplied.
    # Executing the real script by path matches core._run_sky_api's packaging.
    (tmp_path / "sky.py").write_text(
        "import importlib.util\n"
        "assert importlib.util.find_spec('npa') is None\n"
        "def __getattr__(name):\n"
        "    raise AssertionError('SkyPilot called before validation')\n"
    )
    result = subprocess.run(
        [sys.executable, "-S", _sky_api.__file__, action],
        input=json.dumps({"job_id": 42, flag: value}),
        env={"PYTHONPATH": str(tmp_path)},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == f"{flag} must be a JSON boolean"
