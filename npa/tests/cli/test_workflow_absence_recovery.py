"""Prove the registered absence command defaults to verification and sanitizes errors."""

import json

from typer.testing import CliRunner

from npa.cli.main import app
from npa.cli.workbench.workflow import absence_recovery


def test_registered_absence_command_requires_explicit_apply(monkeypatch, tmp_path):
    calls = []

    def verify(path, *, apply):
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
