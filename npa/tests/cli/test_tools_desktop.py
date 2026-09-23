"""Verify desktop command isolation, transport safety, and authenticated gateway configuration."""

import json
import subprocess
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from npa.cli.tools import app
from npa.tools import desktop
from npa.tools.desktop import remote


@pytest.mark.parametrize("action", ["setup", "optimize"])
def test_desktop_dry_run_is_offline(monkeypatch, action):
    monkeypatch.setattr(
        subprocess, "run", Mock(side_effect=AssertionError("network call"))
    )
    result = CliRunner().invoke(
        app, ["desktop", action, "--ssh-host", "dev-host", "--dry-run", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["planned"] is True


@pytest.mark.parametrize(
    "host", ["-oProxyCommand=bad", "host;touch /tmp/bad", "host\nother", "$(id)"]
)
def test_ssh_options_cannot_be_injected(host, monkeypatch):
    transport = Mock()
    monkeypatch.setattr(subprocess, "run", transport)
    with pytest.raises(ValueError, match="SSH"):
        desktop.operate(host, "status")
    transport.assert_not_called()


def test_remote_options_travel_as_data_over_stdin(monkeypatch):
    transport = Mock(
        return_value=subprocess.CompletedProcess([], 0, '{"services": {}}', "")
    )
    monkeypatch.setattr(subprocess, "run", transport)
    desktop.operate(
        "dev-host",
        "setup",
        repository_path="~/space ' and quotes",
        dpi=192,
        geometry="2880x1800",
    )
    args, kwargs = transport.call_args
    assert args[0][-3:] == ["dev-host", "/usr/bin/python3", "-"]
    assert "~/space" not in " ".join(args[0])
    compile(kwargs["input"], "remote-operation", "exec")


def test_failed_transport_never_echoes_secrets(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        Mock(
            return_value=subprocess.CompletedProcess(
                [], 1, "private credential", "private credential"
            )
        ),
    )
    with pytest.raises(RuntimeError) as error:
        desktop.operate("dev-host", "status")
    assert "private credential" not in str(error.value)


@pytest.mark.parametrize(
    "options", [{"dpi": 0}, {"geometry": "100x100;id"}, {"geometry": "10x999999"}]
)
def test_display_arguments_rejected_before_ssh(options, monkeypatch):
    transport = Mock()
    monkeypatch.setattr(subprocess, "run", transport)
    with pytest.raises(ValueError):
        desktop.operate("dev-host", "display", **options)
    transport.assert_not_called()


def test_gateway_auth_applies_to_page_websocket_and_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(remote, "_STATE", tmp_path)
    config = remote._gateway_config(
        {"https_port": 8443, "public_ip": "203.0.113.10"}, tls=True
    )
    assert 'auth_basic "Development desktop";' in config
    assert "auth_basic off" not in config
    assert "proxy_pass http://127.0.0.1:6080" in config
    assert "location = /desktop-credentials.json" in config
    assert "default 0;" in config
    assert '"https://203.0.113.10:8443" 1;' in config
    assert "return 403" in config
    assert 'Cache-Control "no-store"' in config
    assert "return 308 https://203.0.113.10:8443$request_uri;" in config
    assert "https://$host" not in config
    assert "location /.well-known/acme-challenge/" in config


def test_http_gateway_only_serves_certificate_challenges():
    config = remote._gateway_config({}, tls=False)
    assert "location / { return 404; }" in config
    assert "proxy_pass" not in config
    assert "vnc-credentials" not in config


@pytest.mark.parametrize("responses", [["401", "401", "401", "200"], ["200"]])
def test_gateway_verification_requires_auth_without_argv_secrets(
    tmp_path, monkeypatch, responses
):
    monkeypatch.setattr(remote, "_STATE", tmp_path)
    (tmp_path / "public-password").write_text("private-test-credential")
    command = Mock(side_effect=responses)
    monkeypatch.setattr(remote, "_command", command)
    config = {"public_ip": "203.0.113.10", "https_port": 8443}
    if len(responses) == 1:
        with pytest.raises(RuntimeError, match="authentication"):
            remote._verify_gateway(config)
    else:
        remote._verify_gateway(config)
        assert "private-test-credential" in command.call_args.kwargs["data"]
    for call in command.call_args_list:
        assert "private-test-credential" not in " ".join(call.args[0])
        assert "--insecure" not in call.args[0]


def test_passwords_survive_repeated_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "_STATE", tmp_path)
    monkeypatch.setattr(remote, "_HOME", tmp_path)
    (tmp_path / ".vnc").mkdir()
    (tmp_path / ".vnc/passwd").write_bytes(b"existing hash")
    (tmp_path / "password").write_text("existing")
    (tmp_path / "keyring-password").write_text("existing keyring")
    remote._credentials()
    assert (tmp_path / "password").read_text() == "existing"
    assert (tmp_path / "keyring-password").read_text() == "existing keyring"
    assert (tmp_path / ".vnc/passwd").read_bytes() == b"existing hash"


def test_gateway_refuses_unowned_nginx(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "_STATE", tmp_path)
    monkeypatch.setattr(remote.shutil, "which", lambda _: "/usr/sbin/nginx")
    mutation = Mock()
    monkeypatch.setattr(remote, "_packages", mutation)
    with pytest.raises(RuntimeError, match="not owned"):
        remote._gateway_packages({"https_port": 8443})
    mutation.assert_not_called()


def test_gateway_refuses_occupied_ports(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "_STATE", tmp_path)
    monkeypatch.setattr(remote.shutil, "which", lambda _: None)
    monkeypatch.setattr(remote, "_command", lambda *args, **kwargs: "LISTEN")
    mutation = Mock()
    monkeypatch.setattr(remote, "_packages", mutation)
    with pytest.raises(RuntimeError, match="listener"):
        remote._gateway_packages({"https_port": 8443})
    mutation.assert_not_called()


def test_tls_certificate_renewal_reloads_gateway():
    assert "OnCalendar=*-*-* 00,12:00:00" in remote._RENEW_TIMER
    assert "--deploy-hook" in remote._RENEW_SERVICE
    assert "systemctl reload npa-desktop-gateway.service" in remote._RENEW_SERVICE


def test_new_viewer_uses_density_and_readable_scale_controls():
    assert "rfb.scaleViewport=true;rfb.resizeSession=true" in remote._VIEWER
    assert "More space" in remote._VIEWER
    assert "location.reload()" in remote._VIEWER
    assert "localStorage.setItem('npaDesktopScale'" in remote._VIEWER
    assert "localStorage.setItem('password'" not in remote._VIEWER


def test_viewer_refresh_installs_matching_assets_without_restarting_services(
    tmp_path, monkeypatch
):
    commands = Mock(side_effect=AssertionError("unexpected process change"))
    monkeypatch.setattr(remote, "_command", commands)
    assets = desktop._operation_assets("chat-setup")["viewer_assets"]
    remote._write_viewer(tmp_path, assets)
    remote._write_viewer(tmp_path, assets)
    assert (tmp_path / "desktop_clipboard.js").read_text() == assets[
        "desktop_clipboard.js"
    ]
    assert (tmp_path / "desktop.html").read_text() == remote._VIEWER
    assert (tmp_path / "desktop_clipboard.js").stat().st_mode & 0o777 == 0o644
    commands.assert_not_called()


def test_viewer_assets_cannot_write_arbitrary_paths(tmp_path):
    with pytest.raises(ValueError, match="Unexpected desktop viewer assets"):
        remote._write_viewer(tmp_path, {"../outside": "untrusted"})
    assert not list(tmp_path.iterdir())


def test_open_public_desktop_does_not_create_tunnel(monkeypatch):
    monkeypatch.setattr(
        desktop,
        "operate",
        lambda *_: {"public_url": "https://desktop.example.test/desktop.html"},
    )
    tunnel = Mock(side_effect=AssertionError("unexpected SSH tunnel"))
    monkeypatch.setattr(desktop, "_tunnel", tunnel)
    browser = Mock()
    monkeypatch.setattr(desktop.webbrowser, "open", browser)
    assert (
        desktop.open_desktop("dev-host") == "https://desktop.example.test/desktop.html"
    )
    tunnel.assert_not_called()
    browser.assert_called_once()
