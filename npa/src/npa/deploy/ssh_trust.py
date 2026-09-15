"""Authenticate fresh VM host keys through provider-owned serial logs."""

from __future__ import annotations

import argparse
import base64
import hashlib
import ipaddress
import json
import os
import re
import tempfile
from pathlib import Path

import paramiko

from npa.clients import nebius
from npa.clients.config import NPA_CONFIG_DIR


class HostTrustError(RuntimeError):
    """Provider identity or host-key evidence is missing or inconsistent."""


class HostKeyPending(HostTrustError):
    """A verified new instance has not published its host key yet."""


def known_hosts_path(host: str) -> Path:
    identity = hashlib.sha256(host.encode()).hexdigest()
    return NPA_CONFIG_DIR / "ssh" / "hosts" / f"{identity}.known_hosts"


def _atomic_private_file(path: Path, content: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise HostTrustError("SSH trust directory must not be a symlink")
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _host_key_from_logs(raw: str, *, instance_id: str, nonce: str) -> str:
    keys = set()
    marker = re.compile(
        rf"(?:^|\n)NPA_SSH_HOST_KEY {re.escape(nonce)} (ssh-ed25519 [A-Za-z0-9+/]+=*)"
        r"(?:\s|$)"
    )
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            raise HostTrustError("Provider serial logs returned malformed JSONL") from None
        if not isinstance(row, dict) or not isinstance(row.get("labels"), dict):
            raise HostTrustError("Provider serial logs lack resource identity")
        labels = row["labels"]
        if labels.get("sp_resource_id") != instance_id or labels.get("__bucket__") != "sp_serial":
            raise HostTrustError("Provider serial log identity does not match the instance")
        message = row.get("message")
        if not isinstance(message, str):
            raise HostTrustError("Provider serial log message is malformed")
        keys.update(marker.findall(message))
    if not keys:
        raise HostKeyPending("Verified instance has not published its SSH host key yet")
    if len(keys) != 1:
        raise HostTrustError("Conflicting SSH host keys for the deployment nonce")
    key = keys.pop()
    try:
        paramiko.Ed25519Key(data=base64.b64decode(key.split()[1], validate=True))
    except Exception:
        raise HostTrustError("Provider serial log contains an invalid SSH host key") from None
    return key


def _existing_operator_pin(host: str) -> Path:
    selected = os.environ.get("NPA_SSH_KNOWN_HOSTS", "")
    path = Path(selected).expanduser() if selected else Path.home() / ".ssh" / "known_hosts"
    try:
        keys = paramiko.HostKeys(str(path))
    except (OSError, ValueError, paramiko.SSHException):
        raise HostTrustError("Existing VM requires an independently verified known_hosts file") from None
    if not keys.lookup(host):
        raise HostTrustError("Existing VM has no independently verified SSH host-key entry")
    return path.resolve()


def verify_host_key(*, project_id: str, instance_id: str, host: str, nonce: str) -> Path:
    """Pin one exact instance's public key before any SSH authentication occurs.

    This performs one authenticated provider query, without changing IAM. A
    caller may retry HostKeyPending during its existing VM readiness loop;
    identity, schema, permission and conflicting-key failures are terminal.
    """
    if not all(re.fullmatch(r"[a-z][a-z0-9-]{2,127}", value) for value in (project_id, instance_id)):
        raise HostTrustError("Exact project and instance identifiers are required")
    if nonce and not re.fullmatch(r"[a-f0-9]{64}", nonce):
        raise HostTrustError("A fresh deployment SSH host-key nonce is required")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        raise HostTrustError("The SSH endpoint must be the provider-verified IP address") from None
    instance = nebius._run_json(["compute", "instance", "get", "--id", instance_id])
    metadata = instance.get("metadata", {})
    if not isinstance(metadata, dict) or metadata.get("id") != instance_id or metadata.get("parent_id") != project_id:
        raise HostTrustError("Provider instance identity does not match the deployment")
    status = instance.get("status")
    spec = instance.get("spec")
    if not isinstance(status, dict) or not isinstance(spec, dict):
        raise HostTrustError("Provider instance response lacks its status or specification")
    interfaces = status.get("network_interfaces", [])
    if not isinstance(interfaces, list) or any(not isinstance(item, dict) for item in interfaces):
        raise HostTrustError("Provider instance network interfaces are malformed")
    addresses = {
        str(interface.get(field, {}).get("address", "")).split("/")[0]
        for interface in interfaces for field in ("public_ip_address", "ip_address")
        if isinstance(interface.get(field), dict)
    }
    if str(address) not in addresses:
        raise HostTrustError("Provider instance does not own the requested SSH address")
    if not nonce:
        return _existing_operator_pin(host)
    user_data = spec.get("cloud_init_user_data", "")
    if not isinstance(user_data, str):
        raise HostTrustError("Provider instance boot metadata is malformed")
    if f"# NPA_SSH_HOST_KEY_NONCE={nonce}\n" not in user_data:
        raise HostTrustError("Provider instance does not contain the deployment SSH nonce")
    path = known_hosts_path(host)
    receipt_path = path.with_suffix(".json")
    identity = {"project_id": project_id, "instance_id": instance_id, "nonce": nonce}
    if receipt_path.exists():
        try:
            receipt = json.loads(receipt_path.read_text())
        except (ValueError, OSError):
            raise HostTrustError("Saved SSH host-key receipt is unreadable") from None
        if not isinstance(receipt, dict):
            raise HostTrustError("Saved SSH host-key receipt is malformed")
        if all(receipt.get(field) == value for field, value in identity.items()):
            if path.is_symlink() or not path.is_file() or (
                hashlib.sha256(path.read_bytes()).hexdigest() != receipt.get("sha256")
            ):
                raise HostTrustError("The saved SSH host key changed for the verified deployment")
            # Serial logs expire. A durable authenticated pin remains valid;
            # the SSH handshake still rejects an endpoint presenting a new key.
            return path
    query = f'{{sp_resource_id="{instance_id}"}} |= "NPA_SSH_HOST_KEY {nonce} "'
    raw = nebius._run([
        "logging", "query", query, "--project-id", project_id,
        "--bucket", "sp_serial", "--format", "json",
    ])
    key = _host_key_from_logs(raw, instance_id=instance_id, nonce=nonce)
    content = f"{host} {key}\n"
    _atomic_private_file(path, content)
    receipt = {**identity, "sha256": hashlib.sha256(content.encode()).hexdigest()}
    _atomic_private_file(receipt_path, json.dumps(receipt, sort_keys=True) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-id", "instance-id", "host", "nonce"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    try:
        path = verify_host_key(**vars(args))
    except HostKeyPending:
        return 75
    except (HostTrustError, nebius.NebiusError) as exc:
        # Neither provider log rows nor key bytes are included in errors.
        parser.exit(1, f"SSH trust verification failed: {exc}\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
