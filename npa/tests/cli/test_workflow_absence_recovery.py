"""Prove the registered absence command defaults to verification and sanitizes errors."""

import json

import pytest

from typer.testing import CliRunner

from npa.cli.main import app
from npa.cli.workbench.workflow import absence_recovery


def test_registered_absence_command_requires_explicit_apply(monkeypatch, tmp_path):
    calls = []

    def verify(path, *, apply, verification_timeout_seconds):
        assert verification_timeout_seconds == 120
        calls.append((path, apply))
        return {
            "status": "reconciled-absent" if apply else "absence-verified-not-applied"
        }

    monkeypatch.setattr(absence_recovery, "reconcile_absent", verify)
    command = [
        "workbench",
        "workflow",
        "reconcile-absent",
        "--evidence-file",
        str(tmp_path / "record.json"),
    ]
    for apply in (False, True):
        result = CliRunner().invoke(app, command + (["--apply"] if apply else []))
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["status"] == (
            "reconciled-absent" if apply else "absence-verified-not-applied"
        )
    assert [apply for _, apply in calls] == [False, True]


def test_registered_absence_failure_does_not_print_private_error(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise ValueError("private-provider-error-content")

    monkeypatch.setattr(absence_recovery, "reconcile_absent", unavailable)
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "workflow",
            "reconcile-absent",
            "--evidence-file",
            "/private/record.json",
        ],
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "status": "verification-unavailable",
        "reconciled": False,
    }
    assert "private-provider-error-content" not in result.output


@pytest.mark.parametrize("value", ["inf", "nan"])
def test_nonfinite_deadline_is_an_operator_input_error(tmp_path, value):
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "workflow",
            "reconcile-absent",
            "--evidence-file",
            str(tmp_path / "missing"),
            "--verification-timeout-seconds",
            value,
        ],
    )
    assert result.exit_code == 2
    assert "finite" in result.output
    assert "verification-unavailable" not in result.output
