"""Prepare a dedicated controller VM and an explicit, credential-bounded handoff bundle."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import tarfile

from ci_cpu_runner_auth import _private_file
from ci_cpu_runner_cloud import _instance_request, _load, _owned

_STATE = "/var/lib/npa-ci-runners"
_SCRIPTS = (
    "ci_cpu_runners.py",
    "ci_cpu_runner_cloud.py",
    "ci_cpu_runner_auth.py",
    "ci_cpu_runner_remote.py",
)
_INSTALL_FILES = (
    "controller-install.sh",
    "controller-command.sh",
    "npa-ci-runners.service",
)


def _controller_request(
    config: dict, account: str, firewall: str, public_key: str
) -> dict:
    name = f"npa-ci-controller-{config['owner']}"
    request = _instance_request(config, name, "")
    request["metadata"]["labels"]["npa-role"] = "controller"
    spec = request["spec"]
    spec["hostname"] = "npa-ci-controller"
    spec["service_account_id"] = account
    spec["resources"]["preset"] = "2vcpu-8gb"
    spec["recovery_policy"] = "RECOVER"
    spec["network_interfaces"][0]["security_groups"] = [{"id": firewall}]
    spec["cloud_init_user_data"] = "#cloud-config\n" + json.dumps(
        {
            "ssh_authorized_keys": [public_key.strip()],
            "ssh_pwauth": False,
            "runcmd": [
                ["systemctl", "disable", "--now", "docker.service", "docker.socket"],
                ["rm", "-f", "/etc/sudoers.d/runner"],
                ["userdel", "--remove", "runner"],
                ["rm", "-rf", "/opt/actions-runner", "/opt/hostedtoolcache"],
                [
                    "sh",
                    "-c",
                    "printf 'NPA_CONTROLLER_HOST_KEY='; cat /etc/ssh/ssh_host_ed25519_key.pub",
                ],
            ],
        }
    )
    return request


def _runtime_config(config: dict, identity: str) -> dict:
    return {
        **config,
        "profile": "ci-controller",
        "github_auth": "app",
        "quota_scope": "project",
        "supervisor": "systemd",
        "controller_instance_id": identity,
    }


def _add_bytes(archive, name: str, content: bytes, mode=0o600) -> None:
    entry = tarfile.TarInfo(name)
    entry.size = len(content)
    entry.mode = mode
    archive.addfile(entry, io.BytesIO(content))


def _add_regular(archive, source: Path, name: str) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError("Controller bundles accept only explicit regular files")
    _add_bytes(archive, name, source.read_bytes())


def _add_state(archive, root: Path, config: dict) -> None:
    _add_bytes(archive, "state/config.json", json.dumps(config).encode())
    # An installed bundle must not race the previous controller after a reboot.
    _add_bytes(archive, "state/controller-stopped", b"Awaiting controller handoff\n")
    if (root / "routing.json").exists():
        _add_regular(archive, root / "routing.json", "state/routing.json")
    for folder in ("workers", "retired"):
        for source in sorted((root / folder).glob("*.json")):
            _add_regular(archive, source, f"state/{folder}/{source.name}")
    if (root / "github-app.json").exists():
        app = _load(root / "github-app.json")
        key = _private_file(Path(app["private_key_file"]))
        _add_regular(archive, key, "state/github-app.pem")
        app["private_key_file"] = f"{_STATE}/github-app.pem"
        _add_bytes(archive, "state/github-app.json", json.dumps(app).encode())


def _write_bundle(
    repository: Path, root: Path, config: dict, destination: Path
) -> None:
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for name in _SCRIPTS:
                _add_regular(
                    archive, repository / "npa/scripts" / name, f"npa/scripts/{name}"
                )
            for name in _INSTALL_FILES:
                _add_regular(
                    archive,
                    repository / ".github/ci-runners" / name,
                    f".github/ci-runners/{name}",
                )
            _add_state(archive, root, config)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--instance-receipt", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    root = args.state_dir.resolve()
    if root.stat().st_mode & 0o077:
        raise ValueError("Controller state must be private (chmod 700)")
    config = _load(root / "config.json")
    instance = _load(args.instance_receipt)
    _owned(instance, config)
    if instance["metadata"]["labels"].get("npa-role") != "controller":
        raise ValueError("A controller creation receipt is required")
    runtime = _runtime_config(config, instance["metadata"]["id"])
    _write_bundle(Path(__file__).resolve().parents[2], root, runtime, args.output_path)
    print(
        "Private controller bundle prepared; activation remains disabled until handoff."
    )


if __name__ == "__main__":
    _main()
