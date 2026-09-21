"""Install the shared Codex runtime and mobile service in private operator state."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from urllib.parse import urlsplit

_ROOT = Path.home() / ".local/share/nebius-desktop/codex-chat"
_UNITS = Path.home() / ".config/systemd/user"


def _run(*args):
    subprocess.run(args, check=True)


def _write(path, value, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.chmod(mode)
    temporary.replace(path)


def _runtime_config():
    config_path = _ROOT / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    public = json.loads((_ROOT.parent / "public-access.json").read_text())
    parsed = urlsplit(public["url"])
    extensions = sorted(
        Path.home().glob(".vscode/extensions/openai.chatgpt-*/bin/linux-x86_64/codex")
    )
    binary = str(extensions[-1]) if extensions else shutil.which("codex")
    if not binary:
        raise RuntimeError("Install Codex and sign in on the VDI first.")
    listeners = subprocess.check_output(
        ["ss", "-H", "-ltn", "sport", "=", ":6090"], text=True
    )
    if listeners.strip():
        raise RuntimeError("The mobile service port is already occupied.")
    return {
        "binary": binary,
        "socket": str(_ROOT / "app-server.sock"),
        "origin": f"{parsed.scheme}://{parsed.netloc}",
        "username": public["username"],
        "password_file": public["password_file"],
        "cwd": str(Path.home() / "nebius-physical-ai"),
        "port": 6090,
    }


def _unit(command, description, after="network-online.target"):
    return (
        f"[Unit]\nDescription={description}\nAfter={after}\n[Service]\nType=simple\n"
        f"ExecStart={shlex.join(command)}\nRestart=always\nRestartSec=3\nUMask=0077\n"
        "[Install]\nWantedBy=default.target\n"
    )


def _services(config):
    _write(
        _UNITS / "npa-codex-server.service",
        _unit(
            [
                config["binary"],
                "-c",
                "features.code_mode_host=true",
                "app-server",
                "--listen",
                "unix://" + config["socket"],
            ],
            "Shared Codex runtime for desktop and mobile",
        ),
    )
    _write(
        _UNITS / "npa-codex-chat.service",
        _unit(
            [
                str(_ROOT / "venv/bin/python"),
                str(_ROOT / "chat_server.py"),
                str(_ROOT / "config.json"),
            ],
            "Authenticated mobile Codex chat",
            "npa-codex-server.service",
        ),
    )
    _run("systemctl", "--user", "daemon-reload")
    _run("systemctl", "--user", "enable", "--now", "npa-codex-server.service")
    _run("systemctl", "--user", "enable", "npa-codex-chat.service")
    _run("systemctl", "--user", "restart", "npa-codex-chat.service")


def _connect_vscode():
    launcher = Path.home() / ".local/bin/npa-codex-shared"
    command = [
        str(_ROOT / "venv/bin/python"),
        str(_ROOT / "chat_proxy.py"),
        str(_ROOT / "config.json"),
    ]
    _write(launcher, "#!/bin/sh\nexec " + shlex.join(command) + ' "$@"\n', 0o700)
    settings = Path.home() / ".config/Code/User/settings.json"
    value = json.loads(settings.read_text()) if settings.exists() else {}
    backup = _ROOT / "vscode-settings-before-chat.json"
    if not backup.exists():
        _write(backup, json.dumps(value, indent=2))
    value["chatgpt.cliExecutable"] = str(launcher)
    _write(settings, json.dumps(value, indent=2) + "\n")


def main(connect_vscode=False):
    """Install chat without restarting the shared engine or existing IDE sessions.

    Args:
        connect_vscode: Configure the IDE to share the runtime on its next reload.
    Returns:
        None.
    Raises:
        RuntimeError: Required software or an owned listener is unavailable.
        OSError: Private configuration cannot be written.
        subprocess.CalledProcessError: Installation fails.
    """
    os.umask(0o077)
    config = _runtime_config()
    _write(_ROOT / "config.json", json.dumps(config))
    _run("/usr/bin/python3", "-m", "venv", str(_ROOT / "venv"))
    _run(str(_ROOT / "venv/bin/pip"), "install", "websockets==15.0.1")
    _services(config)
    if connect_vscode:
        _connect_vscode()
    _write(
        _ROOT.parent / "chat.json",
        json.dumps(
            {
                "url": config["origin"] + "/chat/",
                "shared_vscode_configured": connect_vscode,
            }
        ),
    )


if __name__ == "__main__":
    main(json.load(sys.stdin).get("connect_vscode", False))
