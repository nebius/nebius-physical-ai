"""Send operator commands to an owned cloud controller over pinned SSH."""

from __future__ import annotations

import ipaddress
import json
from pathlib import Path
import subprocess
import time

from ci_cpu_runner_cloud import _load, _nebius, _nebius_items, _owned

_COMMANDS = {"up", "status", "enable", "disable", "down"}


def _ssh_command(root: Path, remote: dict, command: str) -> list[str]:
    if command not in _COMMANDS:
        raise ValueError("Unsupported remote controller command")
    host = str(ipaddress.ip_address(remote["host"]))
    key = (root / "controller-key").resolve(strict=True)
    known_hosts = (root / "controller-known-hosts").resolve(strict=True)
    if key.parent != root or key.stat().st_mode & 0o077:
        raise ValueError("Controller SSH key must stay in private operator state")
    if known_hosts.parent != root:
        raise ValueError("Controller host keys must stay in private operator state")
    return [
        "ssh",
        "-i",
        str(key),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        f"ubuntu@{host}",
        "sudo",
        "/usr/local/sbin/npa-ci-controller",
        command,
    ]


def _controller_instance(root: Path, config: dict, remote: dict) -> dict:
    instance = _nebius(
        root, config, "compute instance", "get", {"id": remote["instance_id"]}
    )
    _owned(instance, config)
    if instance["metadata"].get("labels", {}).get("npa-role") != "controller":
        raise ValueError("Remote control target is not this pool's controller")
    return instance


def _wake_controller(root: Path, config: dict, remote: dict, instance: dict) -> None:
    if instance["status"]["state"] == "STOPPED":
        _nebius(
            root, config, "compute instance", "start", {"id": remote["instance_id"]}
        )
    probe = _ssh_command(root, remote, "status")
    while True:
        result = subprocess.run(probe, capture_output=True, check=False)
        if result.returncode == 0:
            return
        retryable = (
            b"Connection refused",
            b"Connection timed out",
            b"Operation timed out",
            b"Connection closed",
        )
        if result.returncode != 255 or not any(
            message in result.stderr for message in retryable
        ):
            raise RuntimeError(
                "Controller SSH identity, authentication, or status check failed"
            )
        current = _controller_instance(root, config, remote)
        if current["status"]["state"] not in {"STARTING", "RUNNING"}:
            raise RuntimeError("Cloud controller did not start")
        time.sleep(5)


def _verify_remote_shutdown(root: Path, config: dict, remote: dict) -> None:
    instance = _controller_instance(root, config, remote)
    while instance["status"]["state"] == "STOPPING":
        time.sleep(5)
        instance = _controller_instance(root, config, remote)
    workers = _nebius_items(root, config, "compute instance", config["project_id"])
    owned = [
        item
        for item in workers
        if item["metadata"].get("labels", {}).get("npa-owner") == config["owner"]
        and item["metadata"]["id"] != remote["instance_id"]
    ]
    if instance["status"]["state"] != "STOPPED" or owned:
        raise RuntimeError(
            "SSH disconnected before worker removal and controller stop were verified"
        )
    print("No workers remain; cloud controller stopped.")


def _run_remote(root: Path, config: dict, command: str) -> None:
    remote = _load(root / "remote-controller.json")
    instance = _controller_instance(root, config, remote)
    if command == "status" and instance["status"]["state"] == "STOPPED":
        print(json.dumps({"controller_running": False, "controller_state": "STOPPED"}))
        return
    if command in {"up", "down"}:
        _wake_controller(root, config, remote, instance)
    result = subprocess.run(_ssh_command(root, remote, command), check=False)
    if command == "down" and result.returncode == 255:
        _verify_remote_shutdown(root, config, remote)
    else:
        result.check_returncode()


def _stop_controller(root: Path, config: dict) -> None:
    identity = config.get("controller_instance_id")
    if not identity:
        return
    _controller_instance(root, config, {"instance_id": identity})
    marker = root / "controller-stopped"
    marker.touch(mode=0o600)
    try:
        _nebius(root, config, "compute instance", "stop", {"id": identity})
    except (RuntimeError, OSError):
        marker.unlink(missing_ok=True)
        raise
