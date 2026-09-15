"""Exercise the dispatched scan's shell boundary with a hostile image string."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def test_dispatched_image_is_passed_as_data(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load(
        (root / ".github/workflows/image-security-scan.yml").read_text()
    )
    step = next(
        step for step in workflow["jobs"]["omniverse-payload-scan"]["steps"]
        if step["name"].startswith("Scan the public development image")
    )
    payload = 'example.invalid/image:tag$(touch injected)"; touch injected; #'
    expression = "${{ inputs.omniverse_payload_scan_image }}"
    assert step["env"]["SCAN_IMAGE"] == expression
    executable = tmp_path / "python"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        "pathlib.Path('arguments.json').write_text(json.dumps(sys.argv[1:]))\n"
    )
    executable.chmod(0o700)
    environment = {**os.environ, "PATH": f"{tmp_path}:/usr/bin:/bin", "SCAN_IMAGE": payload}
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"].replace(expression, payload)],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "arguments.json").read_text())[1] == payload
    assert not (tmp_path / "injected").exists()
