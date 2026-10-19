"""Exercise RoboCasa publication identity, failure, and private-scanner contracts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from test_image_byte_publish_contract import POST, PRE, named, steps


def robocasa_block(name):
    script = named(name)["run"]
    opening = 'if [ "$TOOL" = robocasa ]; then\n'
    start = script.index(opening)
    end = script.index("\nfi", start) + len("\nfi")
    return script[start:end]


@pytest.fixture
def shell_environment(tmp_path):
    checkout = tmp_path / "checkout"
    interpreter = checkout / "npa/.venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    # Stubs model previously tested prerequisites. The actual workflow block
    # still controls invocation order, path selection, failures and cleanup.
    interpreter.write_text(
        f"#!{sys.executable}\n"
        + r"""
import json, os, pathlib, stat, sys
args = sys.argv[1:]
script = pathlib.Path(args[0]).name
def option(name):
    return args[args.index(name) + 1]
operation = args[1] if script == "prepare.py" else script
with open(os.environ["GATE_LOG"], "a") as handle:
    handle.write(json.dumps({"operation": operation, "argv": args}) + "\n")
if os.environ.get("FAIL_OPERATION") == operation:
    raise SystemExit(17)
if script == "docker_save_verification.py":
    archive = pathlib.Path(option("--archive"))
    assert archive.read_bytes() == b"saved image"
    assert archive.is_relative_to(pathlib.Path(option("--analysis-root")))
    assert option("--trusted-root") == os.environ["GITHUB_WORKSPACE"]
    assert option("--expected-image-id") == "sha256:" + "a" * 64
    out = pathlib.Path(option("--output-dir"))
    out.mkdir(mode=0o700)
    (out / "verification.json").write_text(json.dumps({"valid": True, "expected_image_id": option("--expected-image-id")}))
elif operation == "authorize":
    root = pathlib.Path(option("--analysis-root"))
    archive = pathlib.Path(option("--archive"))
    graph = pathlib.Path(option("--verification-report"))
    assert archive.is_relative_to(root) and graph.is_relative_to(root)
    assert archive.read_bytes() == b"saved image"
    assert json.loads(graph.read_text())["valid"] is True
    assert stat.S_IMODE(archive.stat().st_mode) == 0o600
    assert stat.S_IMODE(graph.stat().st_mode) == 0o600
    assert stat.S_IMODE(archive.parent.stat().st_mode) == 0o700
    assert option("--policy-mode") == "ci-regex"
    assert option("--trusted-root") == os.environ["GITHUB_WORKSPACE"]
    assert option("--expected-image-id") == "sha256:" + "a" * 64
    out = pathlib.Path(option("--output-dir"))
    out.mkdir(mode=0o700)
    (out / "authorization.json").write_text('{"bound":true}')
elif script == "scan_image_bytes.py":
    assert json.loads(pathlib.Path(option("--authorization")).read_text())["bound"]
    assert option("--trusted-root") == os.environ["GITHUB_WORKSPACE"]
    out = pathlib.Path(option("--output-dir"))
    out.mkdir(mode=0o700)
    assert "--public-native-policy" not in args
    assert "--public-native-policy-sha256" not in args
    (out / "report.json").write_text('{"complete":true,"valid":false,"findings":1}')
    (out / "public-policy-acceptance.json").write_text('{"accepted":true,"raw_scan_valid":false,"accepted_native_occurrences":1}')
else:
    raise SystemExit(29)
"""
    )
    interpreter.chmod(0o755)
    binary = tmp_path / "bin"
    binary.mkdir()
    docker = binary / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        + r"""
import os, sys
assert len(sys.argv) == 6
assert sys.argv[5] == os.environ["EXPECTED_INSPECT_IMAGE"]
assert sys.argv[1:5] == ["image", "inspect", "--format", "{{.Id}}"]
print("sha256:" + "a" * 64)
"""
    )
    docker.chmod(0o755)
    jq = binary / "jq"
    jq.write_text(
        f"#!{sys.executable}\n"
        + r"""
import json, pathlib, sys
assert sys.argv[1:3] == ["-er", ".expected_image_id"]
print(json.loads(pathlib.Path(sys.argv[3]).read_text())["expected_image_id"])
"""
    )
    jq.chmod(0o755)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    analysis = runtime / "analysis"
    analysis.mkdir(mode=0o700)
    env = {
        **os.environ,
        "PATH": str(binary) + os.pathsep + os.defpath,
        "TOOL": "robocasa",
        "IMAGE": "local-image",
        "RUNNER_TEMP": str(runtime),
        "ROBOCASA_BYTE_GATE_ROOT": str(analysis),
        "GITHUB_WORKSPACE": str(checkout),
        "GATE_LOG": str(tmp_path / "gate.jsonl"),
        "FAIL_OPERATION": "",
    }
    return checkout, runtime, analysis, env


@pytest.mark.parametrize(
    "step_name,phase,suffix", [(PRE, "pre", ""), (POST, "post", "-pushed")]
)
@pytest.mark.parametrize(
    "failure",
    ["", "docker_save_verification.py", "authorize", "scan_image_bytes.py"],
)
def test_robocasa_complete_byte_gate_is_pre_and_post_push_and_fail_closed(
    shell_environment, step_name, phase, suffix, failure
):
    checkout, runtime, analysis, env = shell_environment
    env.update(
        {
            "TOOL": "robocasa",
            "ROBOCASA_BYTE_GATE_ROOT": str(analysis),
            "CUSTOMER_DENYLIST": "customer-pattern",
            "INFRA_DENYLIST": "infra-pattern",
            "FAIL_OPERATION": failure,
        }
    )
    env["EXPECTED_INSPECT_IMAGE"] = (
        env["IMAGE"] if phase == "pre" else "local-image@sha256:fixture"
    )
    original = runtime / f"robocasa{suffix}.tar"
    original.write_bytes(b"saved image")
    script = 'set -euo pipefail\nexact="local-image@sha256:fixture"\n'
    script += robocasa_block(step_name) + '\nprintf "gate completed\\n"\n'
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    calls = [
        json.loads(line) for line in Path(env["GATE_LOG"]).read_text().splitlines()
    ]
    expected = ["docker_save_verification.py", "authorize", "scan_image_bytes.py"]
    if failure:
        assert result.returncode == 17, result.stderr
        assert "gate completed" not in result.stdout
        assert [call["operation"] for call in calls] == expected[
            : expected.index(failure) + 1
        ]
        assert (analysis / phase / "image.tar").read_bytes() == b"saved image"
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout == "gate completed\n"
        assert [call["operation"] for call in calls] == expected
        assert not original.exists()
        assert not (analysis / phase / "image.tar").exists()


def test_robocasa_policy_is_required_before_build_and_secrets_are_scoped():
    names = [step.get("name") for step in steps()]
    check = "Validate required RoboCasa confidentiality policy before building"
    prepare = "Prepare and test the RoboCasa complete-byte scanner"
    build = "Build immutable development image locally"
    assert names.index(check) < names.index(prepare) < names.index(build)
    assert names.index(build) < names.index(PRE)
    assert named(check)["if"] == "matrix.tool == 'robocasa'"
    assert named(prepare)["if"] == "matrix.tool == 'robocasa'"
    assert "--policy-mode ci-regex" in named(check)["run"]
    assert {"CUSTOMER_DENYLIST", "INFRA_DENYLIST"} <= named(check)["env"].keys()
    for role in (PRE, POST):
        for key in ("CUSTOMER_DENYLIST", "INFRA_DENYLIST"):
            assert "matrix.tool == 'robocasa'" in named(role)["env"][key]


def test_robocasa_native_gate_requires_cpython312_and_executes_all_prerequisites():
    setup = next(
        step
        for step in steps()
        if step.get("uses", "").startswith("actions/setup-python@")
    )
    assert "matrix.tool == 'robocasa'" in setup["with"]["python-version"]
    script_step = named("Prepare and test the RoboCasa complete-byte scanner")
    assert not script_step.get("continue-on-error")
    script = script_step["run"]
    assert "set -euo pipefail" in script and "umask 077" in script
    assert script.index("go_helper/build.py") < script.index("prepare.py dependencies")
    assert script.index("prepare.py dependencies") < script.index(
        "real_helper_checks.py"
    )
    assert 'mktemp -d "$RUNNER_TEMP/' in script
    assert "|| true" not in script and "--skip" not in script
