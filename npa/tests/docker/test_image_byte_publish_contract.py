"""Execute publication shell boundaries without Docker, secrets, or native tools."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
PUBLISH = ROOT / ".github/workflows/publish-public-images.yml"
SECURITY = ROOT / ".github/workflows/image-security-scan.yml"
PRE = "Enforce runtime, revision, bootstrap, config, and history contracts"
POST = "Verify pushed bytes, revision, payload, visibility, and anonymous pull"


def steps():
    return yaml.safe_load(PUBLISH.read_text())["jobs"]["build-development"]["steps"]


def named(name):
    return next(step for step in steps() if step.get("name") == name)


def curobo_block(name):
    script = named(name)["run"]
    opening = 'if [ "$TOOL" = curobo ]; then\n'
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
    interpreter.write_text(f"#!{sys.executable}\n" + r'''
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
if script == "verify_image.py":
    assert pathlib.Path(option("--docker-save")).read_bytes() == b"saved image"
    assert option("--expected-image-id") == "sha256:" + "a" * 64
    pathlib.Path(option("--json")).write_text(json.dumps({"valid": True, "expected_image_id": option("--expected-image-id")}))
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
    (out / "report.json").write_text('{"complete":true,"valid":true}')
else:
    raise SystemExit(29)
''')
    interpreter.chmod(0o755)
    binary = tmp_path / "bin"
    binary.mkdir()
    docker = binary / "docker"
    docker.write_text(f"#!{sys.executable}\n" + r'''
import sys
assert sys.argv[1:5] == ["image", "inspect", "--format", "{{.Id}}"]
print("sha256:" + "a" * 64)
''')
    docker.chmod(0o755)
    jq = binary / "jq"
    jq.write_text(f"#!{sys.executable}\n" + r'''
import json, pathlib, sys
assert sys.argv[1:3] == ["-er", ".expected_image_id"]
print(json.loads(pathlib.Path(sys.argv[3]).read_text())["expected_image_id"])
''')
    jq.chmod(0o755)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    analysis = runtime / "analysis"
    analysis.mkdir(mode=0o700)
    env = {
        **os.environ,
        "PATH": str(binary) + os.pathsep + os.defpath,
        "TOOL": "curobo",
        "IMAGE": "local-image",
        "RUNNER_TEMP": str(runtime),
        "CUROBO_BYTE_GATE_ROOT": str(analysis),
        "GITHUB_WORKSPACE": str(checkout),
        "GATE_LOG": str(tmp_path / "gate.jsonl"),
        "FAIL_OPERATION": "",
    }
    return checkout, runtime, analysis, env


@pytest.mark.parametrize("step_name,phase,suffix", [(PRE, "pre", ""), (POST, "post", "-pushed")])
@pytest.mark.parametrize("failure", ["", "verify_image.py", "authorize", "scan_image_bytes.py"])
def test_actual_publication_block_retains_failed_inputs_and_stops_later_actions(
    shell_environment, step_name, phase, suffix, failure
):
    checkout, runtime, analysis, env = shell_environment
    original = runtime / f"curobo{suffix}.tar"
    original.write_bytes(b"saved image")
    env["FAIL_OPERATION"] = failure
    # The post-push block receives the already resolved immutable reference.
    script = 'set -euo pipefail\nexact="local-image@sha256:fixture"\n'
    script += curobo_block(step_name) + '\nprintf "gate completed\\n"\n'
    result = subprocess.run(
        ["bash", "-c", script], cwd=checkout, env=env, capture_output=True, text=True,
    )
    calls = [json.loads(line) for line in Path(env["GATE_LOG"]).read_text().splitlines()]
    expected = ["verify_image.py", "authorize", "scan_image_bytes.py"]
    if failure:
        assert result.returncode == 17, result.stderr
        assert "gate completed" not in result.stdout
        assert [call["operation"] for call in calls] == expected[:expected.index(failure) + 1]
        retained = original if failure == "verify_image.py" else analysis / phase / "image.tar"
        assert retained.read_bytes() == b"saved image"
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout == "gate completed\n"
        assert [call["operation"] for call in calls] == expected
        assert not original.exists()
        assert not (analysis / phase / "image.tar").exists()
        assert json.loads((analysis / phase / "scan/report.json").read_text())["valid"]


@pytest.mark.parametrize("step_name", [PRE, POST])
def test_other_images_do_not_enter_curobo_policy_or_native_scan(shell_environment, step_name):
    checkout, _, _, env = shell_environment
    env["TOOL"] = "unrelated-image"
    del env["CUROBO_BYTE_GATE_ROOT"]
    result = subprocess.run(
        ["bash", "-euc", curobo_block(step_name)], cwd=checkout, env=env,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not Path(env["GATE_LOG"]).exists()


def test_required_policy_precedes_build_and_secret_environment_is_scoped():
    all_steps = steps()
    names = [step.get("name") for step in all_steps]
    check = "Validate required cuRobo confidentiality policy before building"
    prepare = "Prepare and test the cuRobo complete-byte scanner"
    build = "Build immutable development image locally"
    push = "Push only after every pre-publication gate passes"
    assert names.index(check) < names.index(prepare) < names.index(build) < names.index(PRE)
    assert names.index(PRE) < names.index(push) < names.index(POST)
    assert named(check)["if"] == "matrix.tool == 'curobo'"
    assert named(prepare)["if"] == "matrix.tool == 'curobo'"
    for step in all_steps:
        env = step.get("env", {})
        if "CUSTOMER_DENYLIST" in env or "INFRA_DENYLIST" in env:
            assert step["name"] in {check, PRE, POST}
            assert {"CUSTOMER_DENYLIST", "INFRA_DENYLIST"} <= env.keys()
            if step["name"] != check:
                assert all("matrix.tool == 'curobo'" in env[key] for key in ("CUSTOMER_DENYLIST", "INFRA_DENYLIST"))
    assert "--policy-mode ci-regex" in named(check)["run"]
    assert "--build-arg" in named(build)["run"]
    assert "DENYLIST" not in named(build)["run"]


def test_native_check_is_an_executed_gate_with_separate_private_dependencies():
    publish = yaml.safe_load(PUBLISH.read_text())
    build_steps = publish["jobs"]["build-development"]["steps"]
    setup = next(step for step in build_steps if step.get("uses") == "actions/setup-python@v6")
    assert setup["with"]["python-version"] == "${{ matrix.tool == 'curobo' && '3.12' || '3.11' }}"
    for name, job in publish["jobs"].items():
        if name == "build-development":
            continue
        for step in job.get("steps", []):
            if step.get("uses") == "actions/setup-python@v6":
                assert step["with"]["python-version"] == "3.11"
    security = yaml.safe_load(SECURITY.read_text())
    job = security["jobs"]["complete-byte-native-integration"]
    assert "if" not in job
    native_step = next(step for step in job["steps"] if "real_helper_checks.py" in step.get("run", ""))
    for step in [native_step, named("Prepare and test the cuRobo complete-byte scanner")]:
        assert not step.get("continue-on-error")
        script = step["run"]
        assert "set -euo pipefail" in script and "umask 077" in script
        assert script.index("go_helper/build.py") < script.index("prepare.py dependencies")
        assert script.index("prepare.py dependencies") < script.index("real_helper_checks.py")
        assert 'mktemp -d "$RUNNER_TEMP/' in script
        assert "|| true" not in script and "--skip" not in script


def test_minimal_curobo_base_gets_the_unchanged_critical_vulnerability_gate():
    spec = yaml.safe_load(SECURITY.read_text())
    job = spec["jobs"]["base-image-cve-scan"]
    base = (ROOT / "npa/docker/workbench/curobo/Dockerfile").read_text().split("FROM ", 1)[1].splitlines()[0]
    entries = [entry for entry in job["strategy"]["matrix"]["include"] if entry["image"] == base]
    assert len(entries) == 1 and entries[0]["purge_linux_libc_dev"] is False
    gate = next(step for step in job["steps"] if step.get("name") == "Trivy image scan")
    assert gate["with"]["severity"] == "CRITICAL"
    assert gate["with"]["exit-code"] == "1"
