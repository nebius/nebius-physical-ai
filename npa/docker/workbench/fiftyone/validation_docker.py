"""Record local FiftyOne validation commands and retire only their owned containers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import stat
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


class _ValidationError(RuntimeError):
    """A required image check or ownership observation did not pass."""


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class _Container:
    role: str
    name: str
    image_id: str
    nonce: str
    container_id: str = ""
    created: str = ""


@dataclass
class _Commands:
    root: Path
    source: Path
    records: list[dict] = field(default_factory=list)

    def run(self, name: str, argv: list[str], *, check: bool = True) -> bytes:
        record = {"name": name, "argv": argv, "started_at": _now()}
        _write_json(self.root / f"{name}.command.json", record)
        environment = self._environment()
        with (self.root / f"{name}.stdout").open("wb") as stdout:
            with (self.root / f"{name}.stderr").open("wb") as stderr:
                result = subprocess.run(
                    argv,
                    cwd=self.source,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
        record.update(exit_code=result.returncode, ended_at=_now())
        self.records.append(record)
        _write_json(self.root / f"{name}.command.json", record)
        _write_json(self.root / "commands.json", self.records)
        if check and result.returncode:
            raise _ValidationError(
                f"{name} exited {result.returncode}; see its private log"
            )
        return (self.root / f"{name}.stdout").read_bytes()

    def _environment(self) -> dict[str, str]:
        return {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": os.environ["HOME"],
            "LANG": "C.UTF-8",
            "DOCKER_CONFIG": str(self.root / "docker-config"),
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _inspect(commands: _Commands, name: str, reference: str) -> dict:
    payload = commands.run(name, ["docker", "inspect", reference])
    try:
        rows = json.loads(payload)
    except (ValueError, UnicodeDecodeError) as error:
        raise _ValidationError("Docker inspection was not structured JSON") from error
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise _ValidationError("Docker inspection did not identify one object")
    return rows[0]


def _require_owner(container: _Container, observed: dict) -> None:
    config = observed.get("Config") or {}
    labels = config.get("Labels") or {}
    if labels.get("npa.fiftyone.validation") != container.nonce:
        raise _ValidationError("Container ownership label changed")
    if observed.get("Name") != "/" + container.name:
        raise _ValidationError("Container name changed")
    if (
        observed.get("Image") != container.image_id
        or config.get("Image") != container.image_id
    ):
        raise _ValidationError("Container image identity changed")
    identity = observed.get("Id", "")
    if not re.fullmatch(r"[0-9a-f]{64}", identity):
        raise _ValidationError("Container identity is missing or malformed")
    if container.container_id and identity != container.container_id:
        raise _ValidationError("Container identity changed")
    if container.created and observed.get("Created") != container.created:
        raise _ValidationError("Container creation lifetime changed")


def _private_mounts(commands, container, source, checks, output) -> list[str]:
    descriptor, name = tempfile.mkstemp(prefix="npa-source-receipt-", dir=output)
    with os.fdopen(descriptor, "wb"):
        pass
    receipt = Path(name)
    # The unpredictable backing file stays beneath the existing private 0700 parent.
    # Keep host readback possible after initial setup assigns the runtime user's owner.
    receipt.chmod(0o666)
    info = receipt.stat()
    _write_json(
        commands.root / f"{container.role}-receipt.json",
        {"path": str(receipt), "device": info.st_dev, "inode": info.st_ino},
    )
    return [
        f"type=bind,src={checks},dst=/validation,readonly",
        f"type=bind,src={source},dst=/tmp/npa-src",
        f"type=bind,src={receipt},dst=/tmp/npa-src-root",
        f"type=bind,src={output},dst=/evidence",
    ]


def _mount_fields(specification: str) -> dict:
    fields = dict(
        part.split("=", 1) for part in specification.split(",") if "=" in part
    )
    return {
        "Type": fields["type"],
        "Source": fields["src"],
        "Destination": fields["dst"],
        "RW": "readonly" not in specification.split(","),
    }


def _creation_arguments(container, mounts, argv, network) -> list[str]:
    contract = (
        f"NPA_FIFTYONE_VALIDATION_CONTRACT=/validation/{container.role}-mounts.json"
    )
    arguments = [
        "docker",
        "create",
        "--pull=never",
        "--name",
        container.name,
        "--label",
        f"npa.fiftyone.validation={container.nonce}",
        "--network",
        network,
        "--env",
        contract,
    ]
    for specification in mounts:
        arguments.extend(["--mount", specification])
    return [*arguments, container.image_id, *argv]


def _verified_mounts(observed: dict, requested: list[str]) -> list[dict]:
    actual = observed.get("Mounts")
    if not isinstance(actual, list) or len(actual) != len(requested):
        raise _ValidationError("Container mounts are missing or unexpected")
    matched = []
    for specification in requested:
        expected = _mount_fields(specification)
        rows = [
            row for row in actual if row.get("Destination") == expected["Destination"]
        ]
        if len(rows) != 1:
            raise _ValidationError(
                "Container mount destination is missing or ambiguous"
            )
        row = rows[0]
        if row.get("Type") != "bind" or row.get("RW") is not expected["RW"]:
            raise _ValidationError("Container mount type or write mode differs")
        if Path(row.get("Source", "")).resolve() != Path(expected["Source"]).resolve():
            raise _ValidationError(
                "Container mount source differs from its owned backing path"
            )
        matched.append(expected)
    return matched


def _record_mounts(commands, container, observed, requested, checks) -> None:
    mounted = _verified_mounts(observed, requested)
    selection = (
        f"NPA_FIFTYONE_VALIDATION_CONTRACT=/validation/{container.role}-mounts.json"
    )
    environment = observed.get("Config", {}).get("Env", [])
    values = [
        value
        for value in environment
        if value.startswith("NPA_FIFTYONE_VALIDATION_CONTRACT=")
    ]
    if values != [selection]:
        raise _ValidationError("Container mount-contract selection differs")
    record = {
        "schema": "npa.fiftyone.mounts.v1",
        "container_id": container.container_id,
        "created": container.created,
        "source": mounted[1],
        "receipt": mounted[2],
        "checks": mounted[0],
    }
    path = checks / f"{container.role}-mounts.json"
    _write_json(path, record)
    path.chmod(0o644)
    _write_json(commands.root / f"{container.role}-mounts.json", record)


def _create(commands, container, source, checks, output, argv, *, network: str) -> None:
    _write_json(commands.root / f"{container.role}-intent.json", vars(container))
    mounts = _private_mounts(commands, container, source, checks, output)
    commands.run(
        f"{container.role}-create",
        _creation_arguments(container, mounts, argv, network),
    )
    observed = _inspect(commands, f"{container.role}-created", container.name)
    _require_owner(container, observed)
    container.container_id = observed["Id"]
    container.created = observed["Created"]
    _write_json(commands.root / f"{container.role}-identity.json", vars(container))
    _record_mounts(commands, container, observed, mounts, checks)


def _verify_receipt(commands, container) -> None:
    record = json.loads((commands.root / f"{container.role}-receipt.json").read_text())
    path = Path(record["path"])
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != (
        record["device"],
        record["inode"],
    ):
        raise _ValidationError("Owned source receipt backing file was replaced")
    mounts = json.loads((commands.root / f"{container.role}-mounts.json").read_text())
    expected = mounts["source"]["Destination"] if container.role == "post" else ""
    if path.read_text() != expected:
        raise _ValidationError(
            "Owned source receipt does not match the original setup path"
        )
    _write_json(
        commands.root / f"{container.role}-receipt-after.json",
        {
            "same_backing_file": True,
            "contents_match_setup": True,
            "sha256": _digest(path),
        },
    )


def _retire(commands: _Commands, container: _Container) -> dict:
    reference = container.container_id or container.name
    observed = _inspect(commands, f"{container.role}-before-cleanup", reference)
    _require_owner(container, observed)
    identity = observed["Id"]
    if observed.get("State", {}).get("Running") is True:
        commands.run(f"{container.role}-stop", ["docker", "stop", identity])
        observed = _inspect(commands, f"{container.role}-stopped", identity)
        _require_owner(container, observed)
    state = observed.get("State") or {}
    if state.get("Running") is not False or state.get("Pid") != 0:
        raise _ValidationError("Owned container shutdown was not verified")
    commands.run(f"{container.role}-remove", ["docker", "rm", identity])
    remaining = commands.run(
        f"{container.role}-absence",
        ["docker", "ps", "-a", "--filter", f"id={identity}", "--format", "{{.ID}}"],
    )
    if remaining.strip():
        raise _ValidationError("Owned container removal was not verified")
    return {
        "role": container.role,
        "shutdown_exit_code": state.get("ExitCode"),
        "processes_stopped": True,
        "container_absent": True,
    }


def _cleanup(
    commands: _Commands, containers: list[_Container]
) -> tuple[list[dict], list[str]]:
    receipts, errors = [], []
    for container in reversed(containers):
        try:
            receipts.append(_retire(commands, container))
        except (OSError, ValueError, _ValidationError) as error:
            errors.append(f"{container.role}: {error}")
    _write_json(
        commands.root / "cleanup.json", {"receipts": receipts, "errors": errors}
    )
    return receipts, errors
