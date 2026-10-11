"""Verify secret-safe CLI login, explicit profiles, refresh, and child cleanup."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys

import pytest

from npa.workbench.team import nebius_login
from npa.workbench.team.errors import AuthenticationError

_BROWSER_URL = (
    "https://auth.nebius.com/oauth/authorize?"
    "redirect_uri=http%3A%2F%2Flocalhost%3A45123%2Fcallback&state=test-state"
)
_FAKE_CLI = r"""
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
profile = args[args.index("--profile") + 1] if "--profile" in args else ""
if "profile" in args and "list" in args:
    print("personal\noperator [default]")
elif "config" in args and "auth-type" in args:
    if profile == "missing":
        print("Authorization: should-never-appear", file=sys.stderr)
        sys.exit(1)
    print("service account" if profile == "operator" else "federation")
elif "profile" in args and "create" in args:
    Path(args[args.index("--config") + 1]).write_text("official-cli-created-config\n")
else:
    visible = [key for key in ("NEBIUS_IAM_TOKEN", "NEBIUS_IAM_TOKEN_FILE",
                               "NPA_NEBIUS_IAM_TOKEN") if key in os.environ]
    Path(os.environ["NPA_TEST_LOGIN_RECEIPT"]).write_text(json.dumps({
        "argv": args, "visible_tokens": visible, "pid": os.getpid()}))
    mode = os.environ.get("NPA_TEST_LOGIN_MODE", "success")
    print("Authorization: should-never-appear", file=sys.stderr, flush=True)
    if mode == "browser":
        print(os.environ["NPA_TEST_LOGIN_URL"], file=sys.stderr, flush=True)
        time.sleep(0.2)
    elif mode == "failure":
        print("https://evil.example/callback?access_token=should-never-appear",
              file=sys.stderr, flush=True)
        print("should-never-appear")
        sys.exit(1)
    elif mode == "block":
        time.sleep(60)
    elif mode == "malformed":
        print("token with whitespace")
        sys.exit(0)
    elif mode == "oversize":
        print("x" * 150000, file=sys.stderr, flush=True)
    print("test-human-token")
"""


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    executable = tmp_path / "nebius"
    executable.write_text(f"#!{sys.executable}\n{_FAKE_CLI}")
    executable.chmod(0o700)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("NPA_TEST_LOGIN_RECEIPT", str(tmp_path / "receipt.json"))
    monkeypatch.setenv("NPA_TEST_LOGIN_URL", _BROWSER_URL)
    return tmp_path / "receipt.json"


def test_profile_metadata_is_secret_free_and_does_not_activate(fake_cli):
    profiles = nebius_login.list_profiles()
    assert profiles == [
        nebius_login.NebiusProfile("personal", "federation", False),
        nebius_login.NebiusProfile("operator", "service account", True),
    ]
    assert not fake_cli.exists()


def test_login_selects_explicit_profile_scrubs_tokens_and_captures_output(
    fake_cli, monkeypatch, capsys
):
    for name in ("NEBIUS_IAM_TOKEN", "NEBIUS_IAM_TOKEN_FILE", "NPA_NEBIUS_IAM_TOKEN"):
        monkeypatch.setenv(name, "ambient-secret")
    monkeypatch.setenv("NEBIUS_PROFILE", "operator")
    output = io.StringIO()
    token = nebius_login.login_token("personal", output=output)
    receipt = json.loads(fake_cli.read_text())
    assert token == "test-human-token"
    assert receipt["visible_tokens"] == []
    assert receipt["argv"][receipt["argv"].index("--profile") + 1] == "personal"
    assert "--no-browser" not in receipt["argv"]
    assert "--debug=false" in receipt["argv"]
    assert "--insecure=false" in receipt["argv"]
    assert output.getvalue() == ""
    assert capsys.readouterr() == ("", "")


def test_remote_login_relays_only_official_url_and_exact_callback(
    fake_cli, monkeypatch
):
    monkeypatch.setenv("NPA_TEST_LOGIN_MODE", "browser")
    output = io.StringIO()
    assert nebius_login.login_token("personal", ssh_host="dev-vm", output=output)
    receipt = json.loads(fake_cli.read_text())
    assert "--no-browser" in receipt["argv"]
    assert _BROWSER_URL in output.getvalue()
    assert "ssh -N -L 45123:127.0.0.1:45123 dev-vm" in output.getvalue()
    assert "should-never-appear" not in output.getvalue()
    assert "test-human-token" not in output.getvalue()


def test_fresh_token_refreshes_without_browser_or_token_persistence(fake_cli, capsys):
    assert nebius_login.fresh_token("personal") == "test-human-token"
    receipt = json.loads(fake_cli.read_text())
    assert "--no-browser" in receipt["argv"]
    assert receipt["argv"][receipt["argv"].index("--auth-timeout") + 1] == "30s"
    assert capsys.readouterr() == ("", "")


def test_refresh_requiring_browser_fails_promptly_and_reaps_child(
    fake_cli, monkeypatch, capsys
):
    monkeypatch.setenv("NPA_TEST_LOGIN_MODE", "browser")
    with pytest.raises(AuthenticationError, match="run npa login again"):
        nebius_login.fresh_token("personal")
    receipt = json.loads(fake_cli.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(receipt["pid"], 0)
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("mode", ["failure", "malformed", "oversize"])
def test_cli_failure_never_exposes_stdout_stderr_or_tokens(fake_cli, monkeypatch, mode):
    monkeypatch.setenv("NPA_TEST_LOGIN_MODE", mode)
    output = io.StringIO()
    with pytest.raises(AuthenticationError) as raised:
        nebius_login.login_token("personal", output=output)
    assert "should-never-appear" not in str(raised.value)
    assert "evil.example" not in str(raised.value)
    assert output.getvalue() == ""


@pytest.mark.parametrize(
    "profile", ["operator", "missing", "--option", "bad\nname", ""]
)
def test_invalid_service_or_missing_profiles_never_mint(fake_cli, profile):
    with pytest.raises(AuthenticationError):
        nebius_login.login_token(profile)
    assert not fake_cli.exists()


def test_cancellation_terminates_and_reaps_owned_child(fake_cli, monkeypatch):
    monkeypatch.setenv("NPA_TEST_LOGIN_MODE", "block")
    processes = []
    start = nebius_login._start_cli

    def capture_start(command):
        process = start(command)
        processes.append(process)
        return process

    def interrupt(selector, capture):
        raise KeyboardInterrupt

    monkeypatch.setattr(nebius_login, "_start_cli", capture_start)
    monkeypatch.setattr(nebius_login, "_read_ready", interrupt)
    with pytest.raises(KeyboardInterrupt):
        nebius_login.login_token("personal")
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert processes[0].stdout.closed and processes[0].stderr.closed


def test_refresh_network_stall_has_bounded_cleanup(fake_cli, monkeypatch):
    start = nebius_login._start_cli
    processes = []

    def capture_start(*args, **kwargs):
        process = start(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setenv("NPA_TEST_LOGIN_MODE", "block")
    monkeypatch.setattr(nebius_login, "_REFRESH_SECONDS", 0.1)
    monkeypatch.setattr(nebius_login, "_start_cli", capture_start)
    with pytest.raises(AuthenticationError, match="refresh timed out"):
        nebius_login.fresh_token("personal")
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert processes[0].stdout.closed and processes[0].stderr.closed


def test_isolated_creation_and_refresh_preserve_explicit_configuration(
    fake_cli, tmp_path
):
    config_file = tmp_path / "private" / "nebius.yaml"
    nebius_login.create_human_profile("personal", config_file)
    assert config_file.stat().st_mode & 0o777 == 0o600
    assert config_file.parent.stat().st_mode & 0o777 == 0o700
    assert config_file.read_text() == "official-cli-created-config\n"
    assert nebius_login.fresh_token("personal", config_file=config_file)
    receipt = json.loads(fake_cli.read_text())
    assert receipt["argv"][receipt["argv"].index("--config") + 1] == str(config_file)
    with pytest.raises(AuthenticationError, match="new private"):
        nebius_login.create_human_profile("personal", config_file)
    assert config_file.read_text() == "official-cli-created-config\n"


def test_new_profile_rejects_public_and_symlink_directories(fake_cli, tmp_path):
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    with pytest.raises(AuthenticationError, match="private directory"):
        nebius_login.create_human_profile("personal", public / "nebius.yaml")
    link = tmp_path / "linked"
    link.symlink_to(public, target_is_directory=True)
    with pytest.raises(AuthenticationError, match="private directory"):
        nebius_login.create_human_profile("personal", link / "nebius.yaml")


def test_failed_profile_creation_leaves_no_configuration_and_allows_retry(
    fake_cli, tmp_path, monkeypatch
):
    config_file = tmp_path / "private" / "nebius.yaml"
    initialize = nebius_login._initialize_profile

    def unavailable(*args):
        raise AuthenticationError("CLI unavailable")

    monkeypatch.setattr(nebius_login, "_initialize_profile", unavailable)
    with pytest.raises(AuthenticationError, match="CLI unavailable"):
        nebius_login.create_human_profile("personal", config_file)
    assert not config_file.exists()
    assert list(config_file.parent.iterdir()) == []
    monkeypatch.setattr(nebius_login, "_initialize_profile", initialize)
    nebius_login.create_human_profile("personal", config_file)
    assert config_file.exists()


def test_remote_login_rejects_option_like_ssh_host(fake_cli):
    with pytest.raises(AuthenticationError, match="SSH host alias"):
        nebius_login.login_token("personal", ssh_host="-oProxyCommand=unexpected")
    assert not fake_cli.exists()


def test_missing_cli_and_subprocess_errors_do_not_expose_exception_output(monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("private-path-secret")

    monkeypatch.setattr(nebius_login.subprocess, "run", unavailable)
    with pytest.raises(
        AuthenticationError, match="Install the official Nebius CLI"
    ) as raised:
        nebius_login.list_profiles()
    assert "private-path-secret" not in str(raised.value)

    def failed(*args, **kwargs):
        raise subprocess.TimeoutExpired("secret-command", 15, output="secret-token")

    monkeypatch.setattr(nebius_login.subprocess, "run", failed)
    with pytest.raises(AuthenticationError) as raised:
        nebius_login.list_profiles()
    assert "secret" not in str(raised.value)


@pytest.mark.parametrize(
    "diagnostic",
    [
        "Error: missing configuration: open /private/config.yaml: no such file or directory\n",
        "Error: config: get profile: no profile configured\n",
    ],
)
def test_profile_discovery_allows_fresh_cli_configuration(monkeypatch, diagnostic):
    monkeypatch.setattr(
        nebius_login.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 4, "", diagnostic),
    )
    assert nebius_login.list_profiles() == []


@pytest.mark.parametrize(
    "diagnostic",
    [
        "Error: malformed configuration: private-detail",
        "Error: missing configuration: open /private/config.yaml: permission denied",
    ],
)
def test_unreadable_configuration_is_not_treated_as_first_login(
    monkeypatch, diagnostic
):
    monkeypatch.setattr(
        nebius_login.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 4, "", diagnostic),
    )
    with pytest.raises(AuthenticationError, match="could not be read"):
        nebius_login.list_profiles()
