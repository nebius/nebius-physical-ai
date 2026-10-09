"""Exercise CLI serialization and the shared authenticated SDK transport."""

import json

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from npa.cli.workbench import team as cli
from npa.workbench.team.client import TeamClient, load_bearer_token
from npa.workbench.team.errors import AuthenticationError, TeamError
from npa.workbench.team.models import SubmitRequest


def test_sdk_sends_token_only_in_header_and_preserves_idempotency(workflow):
    observed = []

    def transport(request):
        observed.append(request)
        return httpx.Response(202, json={"id": "run-" + "1" * 32})

    client = TeamClient(
        "https://team.example.test",
        "test-token",
        transport=httpx.MockTransport(transport),
    )
    request = SubmitRequest(
        workspace="robotics",
        cluster="east",
        idempotency_key="retained-key",
        workflow=workflow,
    )
    try:
        client.submit(request)
        client.submit(request)
    finally:
        client.close()
    assert observed[0].headers["Authorization"] == "Bearer test-token"
    assert "test-token" not in str(observed[0].url)
    assert json.loads(observed[0].content)["idempotency_key"] == "retained-key"
    assert observed[0].content == observed[1].content


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://team.example.test",
        "https://user:secret@team.example.test",
        "https://team.example.test?token=secret",
    ],
)
def test_sdk_rejects_insecure_or_credential_bearing_endpoints(endpoint):
    with pytest.raises(TeamError):
        TeamClient(endpoint, "token")


def test_sdk_does_not_follow_token_redirects():
    seen = []

    def transport(request):
        seen.append(request)
        return httpx.Response(
            307, headers={"Location": "https://untrusted.example.test"}
        )

    client = TeamClient(
        "https://team.example.test", "token", transport=httpx.MockTransport(transport)
    )
    try:
        with pytest.raises(TeamError):
            client.list("robotics")
    finally:
        client.close()
    assert len(seen) == 1


def test_cli_and_sdk_share_the_submission_contract(tmp_path, workflow, monkeypatch):
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(workflow))
    token_file = tmp_path / "access-key"
    token_file.write_text("test-token\n")
    token_file.chmod(0o600)
    calls = []

    class Client:
        def __init__(self, endpoint, token):
            assert endpoint == "https://team.example.test" and token == "test-token"

        def submit(self, request):
            calls.append(request)
            return {"id": "run-" + "1" * 32, "status": "accepted"}

        def close(self):
            return None

    monkeypatch.setattr(cli, "TeamClient", Client)
    result = CliRunner().invoke(
        cli.app,
        [
            "submit",
            "--spec",
            str(path),
            "--workspace",
            "robotics",
            "--cluster",
            "east",
            "--idempotency-key",
            "first",
            "--endpoint",
            "https://team.example.test",
            "--token-file",
            str(token_file),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "accepted"
    assert calls[0].workflow == workflow
    assert "test-token" not in result.output


def test_cli_whoami_uses_the_authenticated_client(tmp_path, monkeypatch):
    token_file = tmp_path / "access-key"
    token_file.write_text("external-token\n")
    token_file.chmod(0o600)

    class Client:
        def __init__(self, endpoint, token):
            assert endpoint == "https://team.example.test"
            assert token == "external-token"

        def whoami(self):
            return {"subject": "verified", "workspaces": []}

        def close(self):
            return None

    monkeypatch.setattr(cli, "TeamClient", Client)
    result = CliRunner().invoke(
        cli.app,
        [
            "whoami",
            "--endpoint",
            "https://team.example.test",
            "--token-file",
            str(token_file),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["subject"] == "verified"
    assert "external-token" not in result.output


def test_cli_list_and_run_accept_the_same_private_token_file(tmp_path, monkeypatch):
    token_file = tmp_path / "access-key"
    token_file.write_text("personal-token\n")
    token_file.chmod(0o600)
    observed = []

    class Client:
        def __init__(self, endpoint, token):
            assert endpoint == "https://team.example.test"
            assert token == "personal-token"

        def list(self, workspace):
            observed.append(("list", workspace))
            return {"runs": []}

        def run(self, run_id, action):
            observed.append(("run", run_id, action))
            return {"id": run_id, "status": "succeeded"}

        def close(self):
            return None

    monkeypatch.setattr(cli, "TeamClient", Client)
    runner = CliRunner()
    common = [
        "--endpoint",
        "https://team.example.test",
        "--token-file",
        str(token_file),
    ]
    listed = runner.invoke(cli.app, ["list", "--workspace", "robotics", *common])
    run_id = "run-" + "1" * 32
    logged = runner.invoke(cli.app, ["run", run_id, "--action", "logs", *common])
    assert listed.exit_code == 0, listed.output
    assert logged.exit_code == 0, logged.output
    assert observed == [("list", "robotics"), ("run", run_id, "logs")]
    assert "personal-token" not in listed.output + logged.output


def test_private_token_file_rejects_unsafe_permissions(tmp_path):
    token_file = tmp_path / "access-key"
    token_file.write_text("personal-token\n")
    token_file.chmod(0o644)
    with pytest.raises(AuthenticationError, match="mode-0600"):
        load_bearer_token("NPA_TEAM_TOKEN", token_file)


def test_render_cli_writes_private_files(config, tmp_path):
    source = tmp_path / "team.yaml"
    source.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    result = CliRunner().invoke(
        cli.app,
        ["render", "--config", str(source), "--output-dir", str(tmp_path / "rendered")],
    )
    assert result.exit_code == 0, result.output
    document = json.loads(result.stdout)
    assert document["credentials_exported"] is False
    assert (tmp_path / "rendered").stat().st_mode & 0o077 == 0
