"""Exercise private GHCR destination refusal before any image is built or pushed."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
PREFLIGHT = "Prove destination cannot expose existing private bytes"


@pytest.mark.parametrize(
    ("status", "visibility", "pages", "allowed"),
    [
        (404, None, None, True),
        (403, None, None, False),
        (200, "public", None, True),
        (200, "internal", None, False),
        (200, "private", [[]], True),
        (200, "private", [[], []], True),
        (200, "private", [[{"metadata": {"container": {"tags": []}}}]], False),
        (200, "private", [[{"metadata": {"container": {"tags": ["old"]}}}]], False),
        (200, "private", [[], [{"id": 123}]], False),
        (200, "private", [], False),
        (200, "private", [[], None], False),
        (200, "private", {}, False),
        (200, "private", None, False),
        (200, "private", "invalid-json", False),
        (200, "private", "request-failed", False),
    ],
)
def test_actual_destination_preflight(tmp_path, status, visibility, pages, allowed):
    workflow = yaml.safe_load((ROOT / ".github/workflows/publish-public-images.yml").read_text())
    steps = workflow["jobs"]["build-development"]["steps"]
    script = next(step["run"] for step in steps if step.get("name") == PREFLIGHT)
    binary = tmp_path / "bin"
    binary.mkdir()
    gh = binary / "gh"
    gh.write_text(f"#!{sys.executable}\n" + '''
import json, os, pathlib, sys
args = sys.argv[1:]
state = json.loads(pathlib.Path(os.environ["FAKE_REGISTRY"]).read_text())
assert args[0] == "api"
if "--include" in args:
    print("HTTP/2 " + str(state["status"]) + " response")
    raise SystemExit(0 if state["status"] == 200 else 1)
if "--jq" in args:
    assert args[-1] == ".visibility"
    print(state["visibility"])
else:
    assert "--paginate" in args and "--slurp" in args
    assert args[-1].endswith("/versions?per_page=100")
    value = state["pages"]
    if value == "request-failed":
        raise SystemExit(9)
    print("malformed" if value == "invalid-json" else json.dumps(value))
''')
    gh.chmod(0o755)
    interpreter = tmp_path / "npa/.venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    helper = tmp_path / "npa/scripts/public_registry_destination.py"
    helper.parent.mkdir(parents=True)
    helper.write_bytes((ROOT / "npa/scripts/public_registry_destination.py").read_bytes())
    state = tmp_path / "registry.json"
    state.write_text(json.dumps({"status": status, "visibility": visibility, "pages": pages}))
    env = {
        **os.environ,
        "PATH": str(binary) + os.pathsep + os.defpath,
        "IMAGE": "ghcr.io/example/project/npa-tool:dev-" + "a" * 40,
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
        "FAKE_REGISTRY": str(state),
        "TMPDIR": str(tmp_path),
    }
    result = subprocess.run(
        ["bash", "-c", script + "\necho destination-approved\n"],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=False,
    )
    assert (result.returncode == 0) is allowed, result.stderr
    assert ("destination-approved" in result.stdout) is allowed
    assert '"tags"' not in result.stdout + result.stderr


def test_destination_inventory_gate_precedes_build_and_push():
    workflow = yaml.safe_load((ROOT / ".github/workflows/publish-public-images.yml").read_text())
    steps = workflow["jobs"]["build-development"]["steps"]
    names = [step.get("name") for step in steps]
    preflight = steps[names.index(PREFLIGHT)]
    assert names.index(PREFLIGHT) < names.index("Build immutable development image locally")
    assert names.index(PREFLIGHT) < names.index("Push only after every pre-publication gate passes")
    assert "continue-on-error" not in preflight
    assert "set -euo pipefail" in preflight["run"]
