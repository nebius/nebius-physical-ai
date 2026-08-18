from __future__ import annotations

from typer.testing import CliRunner

from npa.cli.agent import app
from npa.clients import nebius_vm_auth


def test_auth_profile_cli_routes_to_safe_vm_helper(monkeypatch) -> None:
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return nebius_vm_auth.ProfileVerification("operator", True, True)

    monkeypatch.setattr(nebius_vm_auth, "run_vm_profile_auth", fake_run)
    result = CliRunner().invoke(
        app,
        [
            "auth-profile",
            "--ssh-host",
            "operator.example",
            "--ssh-user",
            "ubuntu",
            "--profile",
            "operator",
            "--auth-timeout-seconds",
            "17",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured == {
        "ssh_host": "operator.example",
        "ssh_user": "ubuntu",
        "identity_file": "",
        "profile": "operator",
        "auth_timeout_seconds": 17,
    }


def test_auth_profile_cli_reports_safe_failure_without_secret(monkeypatch) -> None:
    monkeypatch.setattr(
        nebius_vm_auth,
        "run_vm_profile_auth",
        lambda **_kwargs: (_ for _ in ()).throw(
            nebius_vm_auth.VmAuthError("authorization: Bearer exception-secret")
        ),
    )
    result = CliRunner().invoke(app, ["auth-profile", "--ssh-host", "operator.example"])
    assert result.exit_code == 1
    assert "Authentication failed safely: authorization: [REDACTED]" in result.output
    assert "exception-secret" not in result.output
