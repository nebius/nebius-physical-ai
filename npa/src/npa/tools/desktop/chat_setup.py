"""Install the shared Codex runtime and mobile service in private operator state."""

import json
from http.client import HTTPConnection, HTTPException
import os
from pathlib import Path
import shlex
import shutil
import secrets
import subprocess
import sys
import time
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


def _authentication(config):
    if not config.get("auth_port"):
        listeners = subprocess.check_output(
            ["ss", "-H", "-ltn", "sport", "=", ":6092"], text=True
        )
        if listeners.strip():
            raise RuntimeError("The desktop sign-in port is already occupied.")
    config.setdefault("session_secret", secrets.token_urlsafe(48))
    config.setdefault("auth_port", 6092)
    _write(_ROOT / "config.json", json.dumps(config))
    _write(
        _UNITS / "npa-desktop-auth.service",
        _unit(
            [
                str(_ROOT / "venv/bin/python"),
                str(_ROOT / "chat_auth.py"),
                str(_ROOT / "config.json"),
            ],
            "Desktop browser sign-in independent of Codex",
        ),
    )
    _run("systemctl", "--user", "daemon-reload")
    _run("systemctl", "--user", "enable", "npa-desktop-auth.service")
    _run("systemctl", "--user", "restart", "npa-desktop-auth.service")
    _wait_for_auth(config)


def _wait_for_auth(config):
    for attempt in range(30):
        if _auth_ready(config["auth_port"]):
            return
        if attempt < 29:
            time.sleep(0.2)
    raise RuntimeError("Desktop sign-in service did not become ready.")


def _auth_ready(port):
    # Readiness must stay on the local listener, without proxies or redirects.
    connection = HTTPConnection("127.0.0.1", port=port, timeout=2)
    try:
        connection.request("GET", "/chat/login")
        with connection.getresponse() as response:
            return response.status == 200
    except (OSError, HTTPException):
        return False
    finally:
        connection.close()


def _connect_vscode():
    launcher = Path.home() / ".local/bin/npa-codex-shared"
    command = [
        str(_ROOT / "venv/bin/python"),
        str(_ROOT / "chat_proxy.py"),
        str(_ROOT / "config.json"),
    ]
    _write(launcher, "#!/bin/sh\nexec " + shlex.join(command) + ' "$@"\n', 0o700)
    targets = {
        ".config/Code/User/settings.json": "vscode-settings-before-chat.json",
        ".vscode-server/data/Machine/settings.json": "vscode-ssh-settings-before-chat.json",
        ".vscode-server-insiders/data/Machine/settings.json": "vscode-ssh-insiders-settings-before-chat.json",
    }
    for relative, backup in targets.items():
        if (
            "insiders" not in relative
            or (Path.home() / ".vscode-server-insiders").exists()
        ):
            _configure_vscode(Path.home() / relative, _ROOT / backup, launcher)


def _configure_vscode(settings, backup, launcher):
    value = json.loads(settings.read_text()) if settings.exists() else {}
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
    _run(str(_ROOT / "venv/bin/pip"), "install", "websockets==16.1.1")
    _authentication(config)
    _services(config)
    if connect_vscode:
        _connect_vscode()
    _write(
        _ROOT.parent / "chat.json",
        json.dumps(
            {
                "url": config["origin"] + "/chat/",
                "shared_vscode_configured": connect_vscode,
                "browser_login": True,
            }
        ),
    )


if __name__ == "__main__":
    main(json.load(sys.stdin).get("connect_vscode", False))
