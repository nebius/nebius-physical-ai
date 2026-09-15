"""Record local FiftyOne validation commands and retire only their owned containers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
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


def _create(
    commands: _Commands,
    container: _Container,
    source: Path,
    checks: Path,
    output: Path,
    argv: list[str],
    *,
    network: str,
) -> None:
    _write_json(commands.root / f"{container.role}-intent.json", vars(container))
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
        "--mount",
        f"type=bind,src={checks},dst=/validation,readonly",
        "--mount",
        f"type=bind,src={source},dst=/tmp/npa-src",
        "--mount",
        f"type=bind,src={output},dst=/evidence",
        container.image_id,
        *argv,
    ]
    commands.run(f"{container.role}-create", arguments)
    observed = _inspect(commands, f"{container.role}-created", container.name)
    _require_owner(container, observed)
    container.container_id = observed["Id"]
    container.created = observed["Created"]
    _write_json(commands.root / f"{container.role}-identity.json", vars(container))


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
