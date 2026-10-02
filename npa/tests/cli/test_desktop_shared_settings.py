"""Verify desktop and Remote SSH clients use the same Codex runtime."""

import json
from pathlib import Path

from npa.tools.desktop import chat_setup


def test_shared_runtime_configures_desktop_and_remote_ssh(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(chat_setup, "_ROOT", tmp_path / "runtime")
    paths = [
        tmp_path / ".config/Code/User/settings.json",
        tmp_path / ".vscode-server/data/Machine/settings.json",
        tmp_path / ".vscode-server-insiders/data/Machine/settings.json",
    ]
    for path in paths:
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"editor.fontSize": 16}))
    chat_setup._connect_vscode()
    launcher = str(tmp_path / ".local/bin/npa-codex-shared")
    for path in paths:
        assert json.loads(path.read_text()) == {
            "editor.fontSize": 16,
            "chatgpt.cliExecutable": launcher,
        }
    assert len(list((tmp_path / "runtime").glob("*settings-before-chat.json"))) == 3


def test_repeated_setup_preserves_original_client_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(chat_setup, "_ROOT", tmp_path / "runtime")
    settings = tmp_path / ".vscode-server/data/Machine/settings.json"
    settings.parent.mkdir(parents=True)
    original = {"chatgpt.cliExecutable": "/opt/codex", "editor.fontSize": 15}
    settings.write_text(json.dumps(original))
    chat_setup._connect_vscode()
    chat_setup._connect_vscode()
    backup = tmp_path / "runtime/vscode-ssh-settings-before-chat.json"
    assert json.loads(backup.read_text()) == original
    assert json.loads(settings.read_text())["editor.fontSize"] == 15
    assert not (tmp_path / ".vscode-server-insiders").exists()
