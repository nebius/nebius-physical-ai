"""Keep desktop Codex launchable after installed extensions replace their paths."""

import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from npa.tools.desktop import chat_setup, codex_executable, local_runtime


def _binary(root, directory, version, architecture):
    path = root / directory / "extensions" / f"openai.chatgpt-{version}"
    path = path / "bin" / architecture / "codex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    path.chmod(0o700)
    return path


@pytest.fixture
def installation(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(codex_executable.shutil, "which", lambda _: None)
    return tmp_path


@pytest.mark.parametrize(
    "system,machine,architecture,directory",
    [
        ("Darwin", "arm64", "macos-aarch64", ".vscode"),
        ("Darwin", "x86_64", "macos-x86_64", ".vscode-insiders"),
        ("Linux", "x86_64", "linux-x86_64", ".vscode-server"),
        ("Linux", "aarch64", "linux-aarch64", ".vscode-server-insiders"),
    ],
)
def test_removed_extension_discovers_matching_replacement(
    installation, monkeypatch, system, machine, architecture, directory
):
    monkeypatch.setattr(codex_executable.platform, "system", lambda: system)
    monkeypatch.setattr(codex_executable.platform, "machine", lambda: machine)
    older = _binary(installation, directory, "older", architecture)
    replacement = _binary(installation, directory, "replacement", architecture)
    _binary(installation, directory, "wrong-architecture", "unsupported")
    os.utime(older, (1, 1))
    os.utime(replacement, (2, 2))
    assert codex_executable.resolve_executable(installation / "removed") == str(
        replacement
    )
    assert codex_executable.resolve_executable(older) == str(older)


def test_existing_explicit_executable_is_preserved(installation):
    configured = installation / "custom-codex"
    configured.write_text("#!/bin/sh\n")
    configured.chmod(0o700)
    assert codex_executable.resolve_executable(configured) == str(configured)


def test_missing_installation_has_actionable_error(installation):
    with pytest.raises(RuntimeError, match="Install Codex"):
        codex_executable.resolve_executable(installation / "removed")


def test_path_installation_is_supported(installation, monkeypatch):
    monkeypatch.setattr(codex_executable.shutil, "which", lambda _: "/opt/codex")
    assert codex_executable.resolve_executable(installation / "removed") == "/opt/codex"


def test_launcher_survives_replacement_and_preserves_literal_arguments(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HOME", str(tmp_path))
    system = "macos" if sys.platform == "darwin" else "linux"
    machine = "aarch64" if os.uname().machine in {"arm64", "aarch64"} else "x86_64"
    old = _binary(tmp_path, ".vscode", "old", f"{system}-{machine}")
    command = [sys.executable, codex_executable.__file__, str(old), "app-server"]
    arguments = ["space in argument", "$(must-not-execute)", "`literal`"]
    first = subprocess.run(
        command + arguments, check=True, capture_output=True, text=True
    )
    old.unlink()
    _binary(tmp_path, ".vscode", "new", f"{system}-{machine}")
    second = subprocess.run(
        command + arguments, check=True, capture_output=True, text=True
    )
    assert (
        first.stdout.splitlines()
        == second.stdout.splitlines()
        == ["app-server", *arguments]
    )


def test_cloud_service_uses_launcher_without_restarting_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(chat_setup, "_ROOT", tmp_path)
    monkeypatch.setattr(chat_setup, "_UNITS", tmp_path / "units")
    run = Mock()
    monkeypatch.setattr(chat_setup, "_run", run)
    chat_setup._services({"binary": "/removed/codex", "socket": "/private/codex.sock"})
    service = (tmp_path / "units/npa-codex-server.service").read_text()
    assert f"{tmp_path}/codex_executable.py /removed/codex" in service
    assert not any(
        "restart" in call.args and "npa-codex-server.service" in call.args
        for call in run.call_args_list
    )


def test_local_launcher_upgrade_preserves_active_turns(monkeypatch, tmp_path):
    from npa.tools.desktop import chat_native

    (tmp_path / "native").mkdir()
    (tmp_path / "codex_executable.py").write_text("# installed launcher\n")
    monkeypatch.setattr(local_runtime, "service_running", lambda _: True)
    connection = Mock()
    connection.call.side_effect = RuntimeError("Wait for active turns")
    monkeypatch.setattr(chat_native, "native_connection", lambda *args: connection)
    config = {"binary": "/removed/codex", "native_version": "old"}
    with pytest.raises(RuntimeError, match="active turns"):
        local_runtime._prepare_engine(config, tmp_path)
    connection.connection.close.assert_called_once()
    assert config["native_version"] == "old"
