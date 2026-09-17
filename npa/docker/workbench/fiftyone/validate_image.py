"""Validate one local FiftyOne image before publication using isolated Docker workloads."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import signal
import tarfile
import uuid
from pathlib import Path, PurePosixPath

from validation_checks import _phase_checks
from validation_docker import (
    _Commands,
    _Container,
    _ValidationError,
    _cleanup,
    _create,
    _digest,
    _inspect,
    _require_owner,
    _write_json,
    _verify_receipt,
)


_PYTHON = "/opt/fiftyone/venv/bin/python"
_CHECKS = "/validation/validation_checks.py"


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-id", required=True, help="Exact already-loaded sha256 image ID"
    )
    parser.add_argument(
        "--revision", required=True, help="Full Git revision baked into the image"
    )
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="Checkout containing that revision",
    )
    parser.add_argument(
        "--output-path", required=True, type=Path, help="New private evidence directory"
    )
    return parser.parse_args(argv)


def _prepare(arguments: argparse.Namespace) -> _Commands:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", arguments.image_id):
        raise _ValidationError("Supply an exact local sha256 image ID")
    if not re.fullmatch(r"[0-9a-f]{40}", arguments.revision):
        raise _ValidationError("Supply a full Git revision")
    arguments.source_root = arguments.source_root.resolve(strict=True)
    arguments.output_path = arguments.output_path.absolute()
    if any("," in str(path) for path in (arguments.source_root, arguments.output_path)):
        raise _ValidationError("Docker bind-mount paths must not contain commas")
    arguments.output_path.mkdir(mode=0o700, parents=False, exist_ok=False)
    (arguments.output_path / "docker-config").mkdir(mode=0o700)
    return _Commands(arguments.output_path, arguments.source_root)


def _image_contract(commands: _Commands, arguments: argparse.Namespace) -> dict:
    observed = _inspect(commands, "image-before", arguments.image_id)
    config = observed.get("Config") or {}
    labels = config.get("Labels") or {}
    if observed.get("Id") != arguments.image_id:
        raise _ValidationError("Local image identity differs from the requested ID")
    if labels.get("org.opencontainers.image.revision") != arguments.revision:
        raise _ValidationError("Image revision differs from the selected source")
    if config.get("User") != "ubuntu":
        raise _ValidationError("FiftyOne must retain its default non-root ubuntu user")
    if labels.get("org.nebius.npa.skypilot-bootstrap-contract") != "skypilot-0.12.2-v1":
        raise _ValidationError(
            "FiftyOne bootstrap declaration is missing or unsupported"
        )
    if config.get("Entrypoint") != ["/opt/npa/docker/workbench/fiftyone/entrypoint.sh"]:
        raise _ValidationError("FiftyOne command-forwarding entrypoint changed")
    return observed


def _unpack_source(archive_path: Path, destination: Path) -> list[dict]:
    destination.mkdir(mode=0o777)
    records = []
    with tarfile.open(archive_path) as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if (
                not name.parts
                or name.is_absolute()
                or ".." in name.parts
                or name.parts[0] != "npa"
            ):
                raise _ValidationError("Source archive contains an unexpected path")
            target = destination.joinpath(*name.parts[1:])
            if member.isdir():
                target.mkdir(mode=0o755, parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise _ValidationError(
                    "Source archive must contain only regular files/directories"
                )
            target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
            target.chmod(0o755 if member.mode & 0o111 else 0o644)
            records.append(
                {
                    "path": str(target.relative_to(destination)),
                    "sha256": _digest(target),
                    "executable": bool(member.mode & 0o111),
                }
            )
    # Editable installation creates metadata as the image UID, which can differ from the runner UID.
    for directory in [
        destination,
        *(path for path in destination.rglob("*") if path.is_dir()),
    ]:
        directory.chmod(0o777)
    return records


def _source_inputs(commands: _Commands, revision: str) -> tuple[Path, Path]:
    observed = (
        commands.run("source-revision", ["git", "rev-parse", "HEAD"]).decode().strip()
    )
    if observed != revision:
        raise _ValidationError("Checkout HEAD differs from the image revision")
    archive_path = commands.root / "source.tar"
    commands.run(
        "source-archive",
        ["git", "archive", "--format=tar", "-o", str(archive_path), revision, "npa"],
    )
    source = commands.root / "source"
    records = _unpack_source(archive_path, source)
    _write_json(
        commands.root / "source.json",
        {
            "revision": revision,
            "archive_sha256": _digest(archive_path),
            "files": records,
        },
    )
    _bind_validator(commands, source)
    checks = commands.root / "checks"
    checks.mkdir(mode=0o755)
    checks.chmod(0o755)
    for name in ("validation_checks.py",):
        original = source / "docker/workbench/fiftyone" / name
        target = checks / name
        shutil.copyfile(original, target)
        target.chmod(0o644)
    return source, checks


def _bind_validator(commands: _Commands, source: Path) -> None:
    records = []
    for name in ("validate_image.py", "validation_docker.py", "validation_checks.py"):
        executing = Path(__file__).with_name(name)
        committed = source / "docker/workbench/fiftyone" / name
        if executing.is_symlink() or executing.read_bytes() != committed.read_bytes():
            raise _ValidationError(
                "Executing validator differs from its selected committed source"
            )
        executable = bool(executing.stat().st_mode & 0o111)
        if executable != bool(committed.stat().st_mode & 0o111):
            raise _ValidationError(
                "Executing validator mode differs from its committed source"
            )
        records.append(
            {
                "path": f"npa/docker/workbench/fiftyone/{name}",
                "sha256": _digest(executing),
                "executable": executable,
            }
        )
    _write_json(commands.root / "validator-source.json", records)


def _verify_source(commands: _Commands, source: Path, phase: str = "complete") -> None:
    record = json.loads((commands.root / "source.json").read_text())
    if any(path.is_symlink() for path in source.rglob("*")):
        raise _ValidationError("Validation introduced a source symlink")
    for item in record["files"]:
        path = source / item["path"]
        if (
            _digest(path) != item["sha256"]
            or bool(path.stat().st_mode & 0o111) != item["executable"]
        ):
            raise _ValidationError(
                "Validation changed an original source-archive file or executable bit"
            )
    _write_json(
        commands.root / f"source-after-{phase}.json", {"original_files_unchanged": True}
    )


def _offline(commands: _Commands, container: _Container, name: str) -> None:
    observed = _inspect(commands, name, container.container_id)
    _require_owner(container, observed)
    networks = observed.get("NetworkSettings", {}).get("Networks")
    if not isinstance(networks, dict) or set(networks) - {"none"}:
        raise _ValidationError(
            "Functional validation still has an external network attachment"
        )


def _output_directory(commands: _Commands, phase: str) -> Path:
    output = commands.root / phase
    output.mkdir(mode=0o777)
    # The private parent is 0700; only its dedicated bind mount is writable by UID 1000.
    output.chmod(0o777)
    return output


def _bare(
    commands: _Commands,
    container: _Container,
    source: Path,
    checks: Path,
) -> None:
    output = _output_directory(commands, "bare")
    _create(
        commands,
        container,
        source,
        checks,
        output,
        [_PYTHON, _CHECKS, "bare", "literal argument", "--forwarded-option"],
        network="none",
    )
    _offline(commands, container, "bare-offline")
    commands.run(
        "bare-payload", ["docker", "start", "--attach", container.container_id]
    )
    observed = _inspect(commands, "bare-payload-finished", container.container_id)
    _require_owner(container, observed)
    if observed.get("State", {}).get("ExitCode") != 0:
        raise _ValidationError("Bare-image payload did not succeed")
    _require_phase(output, "bare")


def _exec(commands: _Commands, container: _Container, phase: str) -> None:
    commands.run(
        phase,
        [
            "docker",
            "exec",
            "--env",
            "NPA_SRC_S3_URI=",
            "--env",
            "NPA_SRC_OVERLAY=",
            container.container_id,
            _PYTHON,
            _CHECKS,
            phase,
        ],
    )


def _post_install(
    commands: _Commands,
    container: _Container,
    source: Path,
    checks: Path,
) -> None:
    output = _output_directory(commands, "post")
    _create(
        commands,
        container,
        source,
        checks,
        output,
        ["/bin/bash", "-c", "exec sleep infinity"],
        network="bridge",
    )
    commands.run("post-start", ["docker", "start", container.container_id])
    _exec(commands, container, "initial-install")
    _require_phase(output, "initial-install")
    commands.run(
        "post-disconnect",
        ["docker", "network", "disconnect", "bridge", container.container_id],
    )
    _offline(commands, container, "post-offline")
    _exec(commands, container, "post-install")
    _require_phase(output, "post-install")


def _require_phase(output: Path, phase: str) -> None:
    record = json.loads((output / f"{phase}.json").read_text())
    if record.get("phase") != phase or record.get("status") != "passed":
        raise _ValidationError(f"{phase} did not retain a successful payload receipt")
    checks = record.get("checks")
    expected = [name for name, _ in _phase_checks(phase)]
    if not isinstance(checks, list) or not all(
        isinstance(item, dict) for item in checks
    ):
        raise _ValidationError(f"{phase} has malformed checks")
    if [item.get("name") for item in checks] != expected:
        raise _ValidationError(f"{phase} has missing or unexpected checks")
    if any(item.get("ok") is not True for item in checks):
        raise _ValidationError(f"{phase} retains an unsuccessful check")


def _public_results(commands: _Commands, result: dict) -> dict:
    summary = {
        key: result[key]
        for key in ("status", "image_id", "revision", "source_archive_sha256")
    }
    summary.update(
        schema="npa.fiftyone.public-validation.v1", phases=[], package_versions={}
    )
    summary["source_archive_scope"] = "npa/"
    for directory, phase in (
        ("bare", "bare"),
        ("post", "initial-install"),
        ("post", "post-install"),
    ):
        path = commands.root / directory / f"{phase}.json"
        if not path.exists():
            continue
        _summarize_phase(summary, json.loads(path.read_text()), phase)
    summary["owned_cleanup"] = {
        "containers_removed": len(result["cleanup"]),
        "verified": not result["cleanup_errors"],
    }
    summary["raw_evidence_retention"] = (
        "runner directory only; excluded from public artifact"
    )
    return summary


def _summarize_phase(summary: dict, record: dict, phase: str) -> None:
    allowed = {name for name, _ in _phase_checks(phase)}
    checks = [
        item
        for item in record.get("checks", [])
        if isinstance(item, dict) and item.get("name") in allowed
    ]
    summary["phases"].append(
        {"phase": phase, "checks": [_public_check(item) for item in checks]}
    )
    for item in checks:
        if item.get("ok") is not True:
            continue
        details = item.get("result") or {}
        if not isinstance(details, dict):
            continue
        summary["package_versions"].update(_public_versions(details.get("versions")))
        if item["name"] == "mongodb-source-annex":
            annex = _public_annex(details)
            if annex:
                summary["source_annex"] = annex


def _public_versions(versions: object) -> dict:
    if not isinstance(versions, dict):
        return {}
    allowed = {
        "fiftyone",
        "voxel51-eta",
        "paramiko",
        "datasets",
        "pillow",
        "starlette",
        "boto3",
        "pyarrow",
        "huggingface-hub",
        "fiftyone-brain",
        "fiftyone-db",
        "numpy",
        "scikit-learn",
        "npa",
    }
    return {
        name: value
        for name, value in versions.items()
        if name in allowed
        and isinstance(value, str)
        and re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", value)
    }


def _public_annex(details: dict) -> dict:
    version, count = details.get("version"), details.get("source_members")
    if (
        details.get("source_delivery") != "verified"
        or type(count) is not int
        or count <= 0
    ):
        return {}
    if not isinstance(version, str) or not re.fullmatch(
        r"[0-9]+(?:\.[0-9]+)*", version
    ):
        return {}
    return {"verified": True, "source_members": count, "version": version}


def _public_check(item: dict) -> dict:
    result = {"name": item["name"], "passed": item.get("ok") is True}
    details = item.get("result") or {}
    if isinstance(details, dict):
        for key in ("checks_passed", "checks_total"):
            if type(details.get(key)) is int and 0 <= details[key] <= 5:
                result[key] = details[key]
    return result


def _validate(commands: _Commands, arguments: argparse.Namespace) -> dict:
    source, checks = _source_inputs(commands, arguments.revision)
    before = _image_contract(commands, arguments)
    nonce = uuid.uuid4().hex
    containers: list[_Container] = []
    failure = None
    try:
        for role, operation in (("bare", _bare), ("post", _post_install)):
            container = _Container(
                role,
                f"npa-fiftyone-validation-{nonce}-{role}",
                arguments.image_id,
                nonce,
            )
            containers.append(container)
            operation(commands, container, source, checks)
            _verify_receipt(commands, container)
            _verify_source(commands, source, role)
        after = _inspect(commands, "image-after", arguments.image_id)
        if after.get("Id") != before["Id"] or after.get("Config") != before.get(
            "Config"
        ):
            raise _ValidationError("The validated image changed")
    except (Exception, KeyboardInterrupt) as error:
        failure = error
    cleanup, errors = _cleanup(commands, containers)
    result = _validation_result(commands, arguments, failure, cleanup, errors)
    _write_json(commands.root / "result.json", result)
    _write_json(
        commands.root / "public-summary.json", _public_results(commands, result)
    )
    if failure or errors:
        raise _ValidationError(
            "FiftyOne validation failed; retained payload and cleanup receipts"
        ) from failure
    return result


def _validation_result(
    commands: _Commands,
    arguments: argparse.Namespace,
    failure: BaseException | None,
    cleanup: list,
    errors: list,
) -> dict:
    return {
        "status": "failed" if failure or errors else "passed",
        "image_id": arguments.image_id,
        "revision": arguments.revision,
        "payload_failure": str(failure) if failure else None,
        "cleanup_errors": errors,
        "cleanup": cleanup,
        "source_archive_sha256": _digest(commands.root / "source.tar"),
    }


def _interrupted(_signal: int, _frame: object) -> None:
    raise _ValidationError("Validation interrupted; exact owned cleanup is required")


def main(argv: list[str] | None = None) -> int:
    """Run the local exact-image prepublication checks.

    Args:
        argv: Command-line arguments, or None for the process arguments.
    Returns:
        Zero only when all payload checks and owned cleanup pass; one otherwise.
    Raises:
        None. Validation failures are reported through private evidence and exit status.
    """
    arguments = _arguments(argv)
    commands = None
    previous_handler = signal.signal(signal.SIGTERM, _interrupted)
    try:
        commands = _prepare(arguments)
        _validate(commands, arguments)
    except (Exception, KeyboardInterrupt) as error:
        if commands is not None:
            _write_json(
                commands.root / "failure.json",
                {"type": type(error).__name__, "reason": str(error)},
            )
            summary_path = commands.root / "public-summary.json"
            if not summary_path.exists():
                _write_json(
                    summary_path,
                    {"schema": "npa.fiftyone.public-validation.v1", "status": "failed"},
                )
        print(json.dumps({"status": "failed"}))
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    print((commands.root / "public-summary.json").read_text().strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
