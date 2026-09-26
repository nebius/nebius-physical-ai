"""Private RoboTwin evidence transport keeps its exact scope and failed verdict."""

from __future__ import annotations

import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import httpx
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa/docker/workbench/robotwin"
SPEC = importlib.util.spec_from_file_location(
    "robotwin_private_upload", IMAGE / "private_failure_upload.py"
)
UPLOAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UPLOAD)
SIGNED_URL = "https://example.invalid/owned-object?signature=synthetic-private-value"


@pytest.fixture
def phase(tmp_path):
    (tmp_path / "scan").mkdir()
    for name in UPLOAD.MEMBERS:
        (tmp_path / name).write_bytes(("exact " + name).encode())
    (tmp_path / "authorization.json").write_text("synthetic-private-authorization")
    (tmp_path / "policy.json").write_text("synthetic-private-policy")
    return tmp_path


@pytest.mark.parametrize("status", [200, 302, 403])
def test_exact_bundle_single_put_tls_and_no_redirects(phase, monkeypatch, status):
    calls = []
    client_type = httpx.Client

    def receive(request):
        calls.append(request)
        assert request.method == "PUT" and str(request.url) == SIGNED_URL
        assert (
            "authorization" not in request.headers and "cookie" not in request.headers
        )
        body = request.read()
        assert request.headers["content-length"] == str(len(body))
        assert request.headers["content-type"] == "application/x-tar"
        with tarfile.open(fileobj=io.BytesIO(body)) as archive:
            assert archive.getnames() == list(UPLOAD.MEMBERS)
            for member in archive:
                assert member.isfile() and member.mode == 0o600
                assert (
                    archive.extractfile(member).read()
                    == (phase / member.name).read_bytes()
                )
        return httpx.Response(
            status, headers={"Location": "https://example.invalid/unauthorized"}
        )

    def client(**kwargs):
        assert kwargs == {"verify": True, "follow_redirects": False, "trust_env": False}
        return client_type(**kwargs, transport=httpx.MockTransport(receive))

    monkeypatch.setattr(UPLOAD.httpx, "Client", client)
    if status == 200:
        UPLOAD.upload_failure(phase, SIGNED_URL)
    else:
        with pytest.raises(httpx.HTTPStatusError):
            UPLOAD.upload_failure(phase, SIGNED_URL)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "url",
    [
        "http://example.invalid/object",
        "https://user:pass@example.invalid/object",
        "https://example.invalid/object#fragment",
        "not-a-url",
    ],
)
def test_invalid_destination_never_starts_transport(phase, monkeypatch, url):
    monkeypatch.setattr(
        UPLOAD.httpx, "Client", lambda **kwargs: pytest.fail("unexpected transport")
    )
    with pytest.raises(ValueError):
        UPLOAD.upload_failure(phase, url)


@pytest.mark.parametrize("mutation", ["missing", "symlink"])
def test_incomplete_bundle_never_starts_transport(phase, monkeypatch, mutation):
    source = phase / "scan/records.jsonl"
    source.unlink()
    if mutation == "symlink":
        source.symlink_to(phase / "authorization.json")
    monkeypatch.setattr(
        UPLOAD.httpx, "Client", lambda **kwargs: pytest.fail("unexpected transport")
    )
    with pytest.raises(OSError):
        UPLOAD.upload_failure(phase, SIGNED_URL)


def test_main_does_not_print_private_transport_exception(phase, monkeypatch, capsys):
    def fail(*args):
        raise httpx.ConnectError(SIGNED_URL + " synthetic-private-response")

    monkeypatch.setattr(UPLOAD, "upload_failure", fail)
    monkeypatch.setattr(sys, "argv", ["private_failure_upload.py", str(phase)])
    monkeypatch.setenv(UPLOAD.UPLOAD_ENV, SIGNED_URL)
    assert UPLOAD.main() == 1
    assert capsys.readouterr() == ("", "")


def _fake_gate_interpreter(interpreter):
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text(
        f"#!{sys.executable}\n"
        + """
import os, pathlib, sys
script = pathlib.Path(sys.argv[1]).name
if script == "scan_image_bytes.py":
    raise SystemExit(int(os.environ["SCAN_STATUS"]))
if script == "private_failure_upload.py":
    assert os.environ["ROBOTWIN_PRIVATE_FAILURE_UPLOAD_URL"] not in sys.argv
    pathlib.Path(os.environ["UPLOAD_CALLED"]).touch()
    print(os.environ["ROBOTWIN_PRIVATE_FAILURE_UPLOAD_URL"])
    print("synthetic-private-response", file=sys.stderr)
    raise SystemExit(int(os.environ["UPLOAD_STATUS"]))
"""
    )
    interpreter.chmod(0o755)
    docker = interpreter.parent / "docker"
    docker.write_text("#!/bin/sh\nprintf '%s\\n' sha256:synthetic\n")
    docker.chmod(0o755)


@pytest.mark.parametrize(
    "scan_status,configured,upload_status",
    [
        (0, False, 0),
        (0, True, 0),
        (23, False, 0),
        (23, True, 0),
        (23, True, 71),
    ],
)
def test_actual_gate_uploads_only_failure_and_preserves_exit(
    tmp_path, scan_status, configured, upload_status
):
    interpreter = tmp_path / "npa/.venv/bin/python"
    _fake_gate_interpreter(interpreter)
    analysis = tmp_path / "analysis"
    analysis.mkdir(mode=0o700)
    saved = tmp_path / "saved.tar"
    saved.write_bytes(b"synthetic saved image")
    called = tmp_path / "called"
    env = {
        **os.environ,
        "PATH": str(interpreter.parent) + os.pathsep + os.defpath,
        "ROBOTWIN_BYTE_GATE_ROOT": str(analysis),
        "ROBOTWIN_PUBLIC_NATIVE_POLICY_SHA256": "a" * 64,
        "SCAN_STATUS": str(scan_status),
        "UPLOAD_STATUS": str(upload_status),
        "UPLOAD_CALLED": str(called),
    }
    env.pop(UPLOAD.UPLOAD_ENV, None)
    if configured:
        env[UPLOAD.UPLOAD_ENV] = SIGNED_URL
    result = subprocess.run(
        ["bash", str(IMAGE / "byte_gate.sh"), str(saved), "synthetic-image", "pre"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == scan_status
    assert called.exists() is (scan_status != 0 and configured)
    assert SIGNED_URL not in result.stdout + result.stderr
    assert "synthetic-private-response" not in result.stdout + result.stderr
    assert result.stderr == ""
    expected = ""
    if scan_status and configured:
        expected = (
            "RoboTwin private failure evidence upload "
            + ("completed" if upload_status == 0 else "failed")
            + "\n"
        )
    assert result.stdout == expected


def test_workflow_secret_is_scoped_to_robotwin_byte_gate_steps():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/publish-public-images.yml").read_text()
    )
    selected = []
    for step in workflow["jobs"]["build-development"]["steps"]:
        if UPLOAD.UPLOAD_ENV in step.get("env", {}):
            selected.append(step["name"])
            assert (
                step["env"][UPLOAD.UPLOAD_ENV]
                == "${{ matrix.tool == 'robotwin' && secrets.ROBOTWIN_PRIVATE_FAILURE_UPLOAD_URL || '' }}"
            )
    assert selected == [
        "Enforce runtime, revision, bootstrap, config, and history contracts",
        "Verify pushed bytes, revision, payload, visibility, and anonymous pull",
    ]
