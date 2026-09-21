"""Manage private macOS chat and outbound gateway services through launchd."""

import os
from pathlib import Path
import plistlib
import subprocess
import time


def install_service(label, command, root):
    """Install or refresh one owned LaunchAgent.

    Args:
        label: Owned service label.
        command: Argument array, without passwords or tokens.
        root: Private directory for logs.
    Returns:
        None.
    Raises:
        OSError: The property list cannot be written.
        subprocess.CalledProcessError: launchd rejects the service.
    """
    path = Path.home() / "Library/LaunchAgents" / (label + ".plist")
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "Label": label,
        "ProgramArguments": command,
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 5,
        "WorkingDirectory": str(root),
        "StandardOutPath": str(root / (label + ".log")),
        "StandardErrorPath": str(root / (label + ".error.log")),
        "Umask": 0o077,
    }
    data = plistlib.dumps(value)
    target = f"gui/{os.getuid()}/{label}"
    loaded = subprocess.run(["launchctl", "print", target], capture_output=True)
    if path.exists() and path.read_bytes() == data and loaded.returncode == 0:
        return
    if loaded.returncode == 0:
        subprocess.run(
            ["launchctl", "bootout", target], check=True, capture_output=True
        )
    previous = path.read_bytes() if path.exists() else None
    path.write_bytes(data)
    path.chmod(0o600)
    try:
        _bootstrap(path)
    except subprocess.CalledProcessError:
        if previous:
            path.write_bytes(previous)
            _bootstrap(path)
        raise


def _bootstrap(path):
    command = ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)]
    for attempt in range(30):
        result = subprocess.run(command, capture_output=True)
        if result.returncode == 0:
            return
        # bootout can return before launchd has removed its service record.
        if result.returncode != 5 or attempt == 29:
            result.check_returncode()
        time.sleep(0.2)


def service_running(label):
    """Report whether an owned LaunchAgent is running.

    Args:
        label: Service label.
    Returns:
        A boolean without exposing the service environment.
    Raises:
        OSError: launchctl cannot execute.
    """
    result = subprocess.run(
        ["launchctl", "print", f"gui/{os.getuid()}/{label}"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and "state = running" in result.stdout


def tunnel_command(host, local_port, gateway_port):
    """Build an outbound SSH tunnel exposing only the remote loopback listener.

    Args:
        host: Validated operator SSH alias.
        local_port: Mac chat service port.
        gateway_port: Dedicated gateway loopback port.
    Returns:
        SSH arguments without credentials.
    Raises:
        None.
    """
    return [
        "/usr/bin/ssh",
        "-NT",
        "-o",
        "BatchMode=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-R",
        f"127.0.0.1:{gateway_port}:127.0.0.1:{local_port}",
        "--",
        host,
    ]
