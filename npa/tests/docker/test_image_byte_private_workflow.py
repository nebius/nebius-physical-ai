"""Execute trusted workflow boundaries with synthetic pins and gate processes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

CHECKOUT = Path(__file__).resolve().parents[3]
WORKFLOW = CHECKOUT / ".github/workflows/publish-public-images.yml"
GATE = "npa/scripts/image_byte_scan/private_review_gate.py"
PRE = "Enforce runtime, revision, bootstrap, config, and history contracts"
POST = "Verify pushed bytes, revision, payload, visibility, and anonymous pull"
PIN = "b" * 64
SOURCE = "a" * 40


def _workflow():
    return yaml.safe_load(WORKFLOW.read_text())


def _steps():
    return _workflow()["jobs"]["build-development"]["steps"]


def _named(name):
    return next(step for step in _steps() if step.get("name") == name)


def _plan_program():
    steps = _workflow()["jobs"]["resolve"]["steps"]
    body = next(step["run"] for step in steps if step.get("id") == "plan")
    return body.split("\n", 1)[1].rsplit("\nPY", 1)[0]


def _resolve(pin, tools):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CHECKOUT, text=True).strip()
    environment = {**os.environ, "TARGET": "ghcr.io/nebius/nebius-physical-ai",
                   "NPA_PUBLIC_REGISTRY": "ghcr.io/nebius/nebius-physical-ai",
                   "DEVELOPMENT_SHA": head, "BUILD_TOOLS": tools, "CLEANUP_TOOLS": "",
                   "REVIEW_PUBLIC_KEY_SHA256": pin}
    return subprocess.run([sys.executable, "-c", _plan_program()], cwd=CHECKOUT,
                          env=environment, capture_output=True, text=True, check=False)


@pytest.mark.parametrize("tools", ["curobo", " curobo "])
def test_exact_private_pin_resolves_only_the_single_curobo_build(tools):
    result = _resolve(PIN, tools)
    assert result.returncode == 0, result.stderr
    outputs = dict(line.split("=", 1) for line in result.stdout.splitlines())
    assert outputs["build_count"] == "1"
    assert [entry["tool"] for entry in json.loads(outputs["build_matrix"])] == ["curobo"]


@pytest.mark.parametrize("pin", ["a" * 63, "a" * 65, "A" * 64, "g" * 64,
                                 " " + PIN, PIN + "\n", "$(touch synthetic-sentinel)"])
def test_malformed_dispatch_pin_cannot_emit_a_build_matrix(pin):
    result = _resolve(pin, "curobo")
    assert result.returncode != 0
    assert "private review requires an exact public-key digest and curobo-only build" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("tools", ["", "openpi", "curobo,openpi", "openpi curobo"])
def test_private_authority_cannot_cover_empty_wrong_or_mixed_build_selection(tools):
    result = _resolve(PIN, tools)
    assert result.returncode != 0 and result.stdout == ""
    assert "curobo-only build" in result.stderr


def test_empty_pin_preserves_existing_multiple_public_build_selection():
    result = _resolve("", "curobo,openpi")
    assert result.returncode == 0, result.stderr
    outputs = dict(line.split("=", 1) for line in result.stdout.splitlines())
    assert [entry["tool"] for entry in json.loads(outputs["build_matrix"])] == ["curobo", "openpi"]


def _gate_fragment(name):
    body = _named(name)["run"]
    start = body.index("npa/.venv/bin/python " + GATE)
    end = body.index('rm -f "$phase/image.tar"', start) + len('rm -f "$phase/image.tar"')
    return body[start:end]


def _gate_environment(tmp_path, phase, pin, exit_code):
    interpreter = tmp_path / "npa/.venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text(f"#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\n"
        "Path(os.environ['CALLS']).write_text(json.dumps(sys.argv[1:]))\n"
        "raise SystemExit(int(os.environ['GATE_EXIT']))\n")
    interpreter.chmod(0o700)
    root = tmp_path / "private gate"
    selected = root / phase
    selected.mkdir(parents=True)
    (selected / "image.tar").write_bytes(b"synthetic retained archive")
    return {**os.environ, "CUROBO_BYTE_GATE_ROOT": str(root), "phase": str(selected),
            "GITHUB_WORKSPACE": str(CHECKOUT), "DEVELOPMENT_SHA": SOURCE,
            "CUROBO_REVIEW_PUBLIC_KEY_SHA256": pin, "CUROBO_PUBLIC_NATIVE_POLICY_SHA256": "c" * 64,
            "CALLS": str(tmp_path / "calls.json"), "GATE_EXIT": str(exit_code)}


@pytest.mark.parametrize("phase,name", [("pre", PRE), ("post", POST)])
@pytest.mark.parametrize("pin", ["", PIN, "literal $(touch should-not-exist) pin"])
@pytest.mark.parametrize("exit_code", [0, 17])
def test_both_real_gate_calls_forward_authority_and_stop_before_archive_deletion(tmp_path, phase, name, pin, exit_code):
    environment = _gate_environment(tmp_path, phase, pin, exit_code)
    command = 'set -euo pipefail\n' + _gate_fragment(name) + '\nprintf "completed\\n"\n'
    result = subprocess.run(["bash", "-c", command], cwd=tmp_path, env=environment,
                            capture_output=True, text=True, check=False)
    argv = json.loads((tmp_path / "calls.json").read_text())
    assert argv[0] == GATE
    assert dict(zip(argv[1::2], argv[2::2], strict=True)) == {
        "--analysis-root": environment["CUROBO_BYTE_GATE_ROOT"], "--trusted-root": str(CHECKOUT),
        "--authorization": environment["phase"] + "/authorization/authorization.json",
        "--output-dir": environment["phase"] + "/scan", "--source-sha": SOURCE, "--phase": phase,
        "--review-public-key-sha256": pin,
        "--public-native-policy": str(CHECKOUT / "npa/scripts/image_byte_scan/public_policies/curobo-v2.json"),
        "--public-native-policy-sha256": "c" * 64}
    assert result.returncode == exit_code
    assert result.stdout == ("completed\n" if exit_code == 0 else "")
    assert (Path(environment["phase"]) / "image.tar").exists() == (exit_code != 0)
    assert not (tmp_path / "should-not-exist").exists()


def test_private_gates_preserve_graph_authorization_and_publication_order():
    steps = _steps()
    names = [step.get("name") for step in steps]
    push = names.index("Push only after every pre-publication gate passes")
    assert names.index(PRE) < names.index("Preserve sanitized pre-publication receipt") < push
    assert push < names.index(POST) < names.index("Preserve sanitized post-publication receipt")
    for name in (PRE, POST):
        body = _named(name)["run"]
        assert body.index("curobo/verify_image.py") < body.index("prepare.py authorize") < body.index(GATE)
        assert "--policy-mode ci-regex" in body
        assert "|| true" not in _gate_fragment(name) and "continue-on-error" not in _named(name)
    post = _named(POST)["run"]
    assert post.index('docker pull "$exact"') < post.index(GATE)
    assert post.index('test "$(docker image inspect') < post.index(GATE)
    assert post.index(GATE) < post.index('test "$(gh api "$package_api" --jq .visibility)" = public')
    assert post.index(GATE) < post.index('crane manifest "$IMAGE"')


@pytest.mark.parametrize("phase", ["pre", "post"])
def test_public_receipt_uses_the_same_dispatch_authority_in_each_phase(phase):
    body = _named(f"Preserve sanitized {phase}-publication receipt")["run"]
    assert 'receipt_args+=(--curobo-review-public-key-sha256 "$CUROBO_REVIEW_PUBLIC_KEY_SHA256")' in body
    assert body.index('if [ "$TOOL" = curobo ]; then') < body.index("--curobo-review-public-key-sha256")
    assert '"${receipt_args[@]}"' in body
    job = _workflow()["jobs"]["build-development"]
    assert job["env"]["CUROBO_REVIEW_PUBLIC_KEY_SHA256"] == "${{ inputs.curobo_private_review_public_key_sha256 }}"


@pytest.mark.parametrize("pin,archives,directory,expected", [
    ("", 0, False, 0), (PIN, 0, False, 1), (PIN, 1, False, 0),
    (PIN, 2, False, 1), (PIN, 1, True, 1),
])
def test_signature_build_requires_one_existing_verified_toolchain_archive(tmp_path, pin, archives, directory, expected):
    environment = _gate_environment(tmp_path, "pre", pin, 0)
    root = Path(environment["CUROBO_BYTE_GATE_ROOT"])
    environment["scan_root"] = str(root)
    for ordinal in range(archives):
        archive = root / f"tools/go-build-synthetic-{ordinal}/go.tar.gz"
        archive.parent.mkdir(parents=True)
        if directory:
            archive.mkdir()
        else:
            archive.write_bytes(b"synthetic input for separately tested pinned hash verifier")
    body = _named("Prepare and test the cuRobo complete-byte scanner")["run"]
    fragment = body[body.index('if [ -n "$CUROBO_REVIEW_PUBLIC_KEY_SHA256" ]; then'):]
    result = subprocess.run(["bash", "-c", "set -euo pipefail\n" + fragment], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == expected
    calls = tmp_path / "calls.json"
    assert calls.exists() == (bool(pin) and expected == 0)
    if calls.exists():
        argv = json.loads(calls.read_text())
        assert argv == ["npa/scripts/image_byte_scan/review_signature_build.py",
            "--analysis-root", str(root), "--trusted-root", str(CHECKOUT),
            "--output-dir", str(root / "review-signature"),
            "--toolchain-archive", str(root / "tools/go-build-synthetic-0/go.tar.gz")]
