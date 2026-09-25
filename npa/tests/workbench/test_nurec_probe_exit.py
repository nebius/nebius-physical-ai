"""Protect failed native scan qualification against Kit's zero-status shutdown."""

import os
from pathlib import Path
import subprocess
import sys

from npa.workbench.nurec import navigation_probe


def test_native_shutdown_cannot_hide_probe_failure(tmp_path):
    script = tmp_path / "failed_probe.py"
    script.write_text(
        "import atexit, os, sys, types\n"
        "from npa.workbench.nurec import navigation_probe as probe\n"
        "atexit.register(lambda: os._exit(0))\n"
        "sys.modules['isaacsim'] = types.SimpleNamespace(\n"
        "    SimulationApp=lambda *args: types.SimpleNamespace(close=lambda: None))\n"
        "def reject(*args):\n"
        "    raise ValueError('measured probe rejected')\n"
        "probe._verify_and_publish = reject\n"
        "sys.argv = ['probe', '--input-path', 'unused', '--output-path', 'unused',\n"
        "            '--runtime-image', 'registry.example.invalid/isaac@sha256:' + 'a' * 64]\n"
        "probe.main()\n"
    )
    source = Path(navigation_probe.__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, str(script)],
        env={**os.environ, "PYTHONPATH": str(source)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "measured probe rejected" in result.stderr
