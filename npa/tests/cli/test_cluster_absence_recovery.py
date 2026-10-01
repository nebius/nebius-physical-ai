"""Malformed terminal evidence has a stable, credential-free JSON failure."""

import json
import subprocess
import tarfile

import pytest
import yaml
from typer.testing import CliRunner
from npa.cli.main import app
from npa.cli.cluster import reconcile_absent as command


@pytest.mark.parametrize(
    "error",
    [
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        AttributeError,
        OSError,
        tarfile.ReadError,
        yaml.YAMLError,
        subprocess.CalledProcessError,
    ],
)
def test_failure_envelope_never_discloses_evidence(monkeypatch, tmp_path, error):
    def fail(*args, **kwargs):
        if error is subprocess.CalledProcessError:
            raise error(1, ["private-provider-credential"])
        raise error("private-provider-credential")

    monkeypatch.setattr(command, "reconcile_absent", fail)
    result = CliRunner().invoke(
        app,
        [
            "cluster",
            "reconcile-absent",
            "--evidence-file",
            str(tmp_path / "missing.json"),
        ],
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "status": "verification-unavailable",
        "reconciled": False,
    }
    assert "private-provider-credential" not in result.output


def test_verification_deadline_is_forwarded(monkeypatch, tmp_path):
    calls = []

    def verify(path, **kwargs):
        calls.append(kwargs)
        return {"status": "test"}

    monkeypatch.setattr(command, "reconcile_absent", verify)
    for value in ("0", "0.25"):
        result = CliRunner().invoke(
            app,
            [
                "cluster",
                "reconcile-absent",
                "--evidence-file",
                str(tmp_path / "input"),
                "--verification-timeout-seconds",
                value,
            ],
        )
        assert result.exit_code == 0, result.output
    assert calls == [
        {"verification_timeout_seconds": 0.0},
        {"verification_timeout_seconds": 0.25},
    ]
