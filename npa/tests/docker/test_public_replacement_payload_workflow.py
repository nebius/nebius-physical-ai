"""Exercise replacement-image payload gates in the trusted publication shell."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github/workflows/publish-public-images.yml"
PRE = "Enforce runtime, revision, bootstrap, config, and history contracts"
POST = "Verify pushed bytes, revision, payload, visibility, and anonymous pull"


@pytest.mark.parametrize("phase", [PRE, POST])
@pytest.mark.parametrize("tool", ["openpi", "robocasa", "cosmos3-nano-video", "cosmos3-super-benchmark"])
@pytest.mark.parametrize("failure", [False, True])
def test_replacement_verifier_controls_publication_continuation(tmp_path, phase, tool, failure):
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["build-development"]["steps"]
    script = next(step["run"] for step in steps if step.get("name") == phase)
    opening = (
        'if [[ "$TOOL" == openpi || "$TOOL" == robocasa ]]; then'
        if tool in {"openpi", "robocasa"}
        else 'if [[ "$TOOL" == cosmos3-serving || "$TOOL" == cosmos3-nano-video || "$TOOL" == cosmos3-super-benchmark ]]; then'
    )
    start = script.index(opening)
    block = script[start:script.index("\nfi", start) + 3]
    binary = tmp_path / "bin"
    binary.mkdir()
    docker = binary / "docker"
    docker.write_text(f"#!{sys.executable}\nprint('sha256:' + 'a' * 64)\n")
    docker.chmod(0o755)
    interpreter = tmp_path / "npa/.venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text(f"#!{sys.executable}\n" + '''
import json, os, pathlib, sys
args = sys.argv[1:]
key = "--docker-save" if "--docker-save" in args else "--tarball"
assert pathlib.Path(args[args.index(key) + 1]).read_bytes() == b"exact saved image"
pathlib.Path(os.environ["CALL_LOG"]).write_text(json.dumps(args))
raise SystemExit(17 if os.environ["FAIL_GATE"] == "1" else 0)
''')
    interpreter.chmod(0o755)
    suffix = "-pushed" if phase == POST else ""
    (tmp_path / f"{tool}{suffix}.tar").write_bytes(b"exact saved image")
    env = {**os.environ, "TOOL": tool, "RUNNER_TEMP": str(tmp_path), "IMAGE": "candidate",
           "PATH": str(binary) + os.pathsep + os.defpath,
           "CALL_LOG": str(tmp_path / "call.json"), "FAIL_GATE": str(int(failure))}
    result = subprocess.run(
        ["bash", "-c", 'set -euo pipefail\nexact="candidate@sha256:resolved"\n' + block + '\necho accepted\n'],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=False,
    )
    args = json.loads((tmp_path / "call.json").read_text())
    expected = (f"npa/docker/workbench/{tool}/verify_image.py" if tool in {"openpi", "robocasa"}
                else "npa/scripts/scan_image_cosmos3_serving_payload.py")
    assert args[0] == expected
    assert result.returncode == (17 if failure else 0), result.stderr
    assert ("accepted" in result.stdout) is not failure
    assert args[args.index("--json") + 1].startswith(str(tmp_path / f"{tool}{suffix}-"))


def test_exact_archive_gates_surround_the_only_push():
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["build-development"]["steps"]
    names = [step.get("name") for step in steps]
    push = "Push only after every pre-publication gate passes"
    assert names.index(PRE) < names.index(push) < names.index(POST)
    for phase in (PRE, POST):
        step = steps[names.index(phase)]
        assert "continue-on-error" not in step
        assert "sha256sum" in step["run"] and "-archive.sha256" in step["run"]


def test_sanitized_receipts_preserve_prepublication_failure_evidence():
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["build-development"]["steps"]
    names = [step.get("name") for step in steps]
    pre = names.index("Preserve sanitized pre-publication receipt")
    post = names.index("Preserve sanitized post-publication receipt")
    assert names.index("Generate pre-publication SBOM") < pre
    assert pre < names.index("Push only after every pre-publication gate passes")
    assert names.index(POST) < post
    upload = next(step for step in steps if step.get("name") == "Upload sanitized exact-image validation receipts")
    assert "always()" in upload["if"]
    assert "steps.receipt-pre.outcome == 'success'" in upload["if"]
    assert "steps.receipt-post.outcome == 'success'" in upload["if"]
    assert upload["with"]["path"] == "${{ runner.temp }}/public-image-receipts/*.json"
    assert upload["with"]["if-no-files-found"] == "error"
    assert upload["with"]["overwrite"] is False
