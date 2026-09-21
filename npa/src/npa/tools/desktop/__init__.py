"""Install and operate a persistent Linux development desktop over SSH."""

from __future__ import annotations

import ipaddress
import json
from pathlib import Path
import re
import subprocess
import webbrowser


def _validate_host(host: str) -> None:
    if not re.fullmatch(r"[a-zA-Z0-9_][a-zA-Z0-9_.@-]*", host):
        raise ValueError("Use an SSH config alias or a user@hostname, without options.")


def _configuration(action: str, options: dict) -> dict:
    if action not in {
        "setup", "status", "display", "public-access", "chat-setup", "optimize"
    }:
        raise ValueError("Unsupported desktop action.")
    result = {"action": action, **options}
    if "dpi" in result and not 96 <= result["dpi"] <= 240:
        raise ValueError("Desktop DPI must be between 96 and 240.")
    if "geometry" in result and not re.fullmatch(
        r"[1-9][0-9]{2,3}x[1-9][0-9]{2,3}", result["geometry"]
    ):
        raise ValueError("Geometry must use WIDTHxHEIGHT in pixels.")
    if action == "public-access":
        address = ipaddress.ip_address(result["public_ip"])
        if address.version != 4 or not address.is_global:
            raise ValueError("Public access requires the VM's public IPv4 address.")
        if not 1024 <= result["https_port"] <= 65535:
            raise ValueError("Use a dedicated HTTPS port between 1024 and 65535.")
    return result


def _chat_assets():
    names = [
        "chat_setup.py",
        "chat_server.py",
        "chat_rpc.py",
        "chat_proxy.py",
        "chat_history.py",
        "chat_models.py",
        "chat_delivery.py",
        "chat_session.py",
        "chat.html",
        "chat.css",
        "chat.js",
    ]
    return {name: (Path(__file__).parent / name).read_text() for name in names}


def operate(host: str, action: str, *, dry_run: bool = False, **options) -> dict:
    """Run a desktop action on an explicitly selected existing SSH host.

    Args:
        host: Existing SSH alias or user@hostname using the operator's SSH keys.
        action: Setup, status, display, public-access, chat-setup, or optimize.
        dry_run: Return the intended action without connecting or writing files.
        options: Validated action-specific configuration; never credentials.
    Returns:
        A status document without passwords or private keys.
    Raises:
        ValueError: An option is invalid.
        RuntimeError: Remote execution or its response failed.
    """
    _validate_host(host)
    config = _configuration(action, options)
    if dry_run:
        return {"planned": True, "ssh_host": host, **config}
    if action == "chat-setup":
        config["chat_assets"] = _chat_assets()
    source = (Path(__file__).parent / "remote.py").read_text()
    payload = source + "\n_run(" + repr(config) + ")\n"
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "--", host, "/usr/bin/python3", "-"],
        input=payload,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "Desktop operation failed; inspect ~/.local/share/nebius-desktop/setup.log on the VM."
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Desktop operation returned an invalid response.") from exc


def open_desktop(host: str, *, local_port: int = 16080, chat: bool = False) -> str:
    """Open the authenticated public desktop or establish a private SSH tunnel.

    Args:
        host: Existing SSH alias or user@hostname.
        local_port: Loopback port for the SSH-only viewer.
        chat: Open the mobile chat interface on the existing HTTPS gateway.
    Returns:
        The browser URL, without credentials.
    Raises:
        ValueError: Host or port is invalid.
        RuntimeError: Desktop status or tunnel creation fails.
    """
    _validate_host(host)
    if not 1024 <= local_port <= 65535:
        raise ValueError("Local port must be between 1024 and 65535.")
    state = operate(host, "status")
    if chat:
        url = state.get("chat", {}).get("url")
        if not url:
            raise RuntimeError("Run desktop chat-setup before opening mobile chat.")
        webbrowser.open(url)
        return url
    url = state.get("public_url")
    if not url:
        _tunnel(host, local_port)
        url = f"http://127.0.0.1:{local_port}/desktop.html"
    webbrowser.open(url)
    return url


def _tunnel(host: str, port: int) -> None:
    import hashlib

    root = Path.home() / ".local/share/nebius-desktop"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    socket = root / (
        "ssh-" + hashlib.sha256(f"{host}:{port}".encode()).hexdigest()[:16]
    )
    check = subprocess.run(
        ["ssh", "-S", str(socket), "-O", "check", host], capture_output=True
    )
    if check.returncode == 0:
        return
    result = subprocess.run(
        [
            "ssh",
            "-M",
            "-S",
            str(socket),
            "-fNT",
            "-o",
            "BatchMode=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-L",
            f"127.0.0.1:{port}:127.0.0.1:6080",
            host,
        ],
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError("Could not establish the desktop SSH tunnel.")
