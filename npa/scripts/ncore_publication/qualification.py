"""Run NCore qualification commands through one host-attested local image."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Sequence

from image_byte_scan import core as W, prepare as P

from . import artifact
from .process import file_sha, write_json


CANDIDATE_FORMAT = "npa.ncore.qualification-candidate-image.v1"
EXECUTION_FORMAT = "npa.ncore.host-container-execution.v1"
QUALIFICATION_FORMAT = "npa.ncore.candidate-qualification.v1"
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _private_file(path: Path, label: str) -> Path:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_uid != os.getuid()
        or path.stat().st_nlink != 1
        or path.stat().st_mode & 0o077
    ):
        raise ValueError(f"{label} must be a private owner-only regular file")
    return path


def _private_directory(path: Path, label: str, *, create: bool = False) -> Path:
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=False)
    if (
        path.is_symlink()
        or not path.is_dir()
        or path.stat().st_uid != os.getuid()
        or path.stat().st_mode & 0o077
    ):
        raise ValueError(f"{label} must be a private owner-only directory")
    return path


def bind_candidate_image(
    *,
    source_sha: str,
    analysis_root: Path,
    gate_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Recompute the checked OCI-to-local-image relationship."""
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ValueError("source SHA is invalid")
    analysis_root = _private_directory(analysis_root.absolute(), "analysis root")
    gate_dir = _private_directory(gate_dir.absolute(), "gate directory")
    if not gate_dir.is_relative_to(analysis_root):
        raise ValueError("gate directory is outside the analysis root")
    if not output_path.absolute().is_relative_to(analysis_root):
        raise ValueError("candidate receipt is outside the analysis root")
    expected = W.bound_json(P.binding(gate_dir / "inspection.json"))
    inspected = W.bound_json(P.binding(gate_dir / "loaded-image.json"))
    if (
        not isinstance(inspected, list)
        or len(inspected) != 1
        or not isinstance(inspected[0], dict)
    ):
        raise ValueError("loaded image inspection is invalid")
    relationship = artifact.verify_local_export(
        gate_dir / "loaded-image.tar", inspected[0], expected
    )
    stored_path = gate_dir / "local-image-binding.json"
    stored = W.bound_json(P.binding(stored_path))
    if stored != relationship:
        raise ValueError("local image binding changed after the checked image gates")
    build_path = analysis_root / "build/build.json"
    build = W.bound_json(P.binding(build_path))
    if (
        build.get("schema") != "npa.ncore.committed-oci-build.v1"
        or build.get("source_sha") != source_sha
        or build.get("image_digest") != relationship["image_digest"]
        or build.get("archive_sha256") != relationship["archive_sha256"]
    ):
        raise ValueError("build receipt differs from the checked local image")
    receipt = {
        "format": CANDIDATE_FORMAT,
        "status": "pass",
        "source_sha": source_sha,
        "local_image_id": relationship["local_image_id"],
        "identity_kind": relationship["identity_kind"],
        "image_digest": relationship["image_digest"],
        "platform_digest": relationship["platform_digest"],
        "config_digest": relationship["config_digest"],
        "archive_sha256": relationship["archive_sha256"],
        "build_receipt_sha256": file_sha(build_path),
        "inspection_receipt_sha256": file_sha(gate_dir / "inspection.json"),
        "local_image_binding_sha256": file_sha(stored_path),
        "local_export_sha256": relationship["export_sha256"],
    }
    write_json(output_path, receipt)
    return receipt


def _run(
    command: Sequence[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]],
) -> subprocess.CompletedProcess[bytes]:
    try:
        return runner(
            list(command),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=None,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("local candidate container operation failed") from exc


def _write_bytes(path: Path, value: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(value)


def _inspect(
    reference: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]],
) -> dict[str, Any]:
    result = _run(["docker", "container", "inspect", reference], runner=runner)
    if result.returncode != 0:
        raise ValueError("candidate container inspection failed")
    try:
        payload = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("candidate container inspection is invalid") from exc
    if (
        not isinstance(payload, list)
        or len(payload) != 1
        or not isinstance(payload[0], dict)
    ):
        raise ValueError("candidate container inspection is ambiguous")
    return payload[0]


def run_attested_container(
    *,
    role: str,
    run_id: str,
    candidate: dict[str, Any],
    command: Sequence[str],
    evidence_dir: Path,
    cache_dir: Path,
    env_file: Path,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> dict[str, Any]:
    """Create, inspect, run, and retire one exact pull-disabled container."""
    if _SAFE_NAME.fullmatch(role) is None or _SAFE_NAME.fullmatch(run_id) is None:
        raise ValueError("qualification execution name is invalid")
    evidence_dir = _private_directory(evidence_dir, "evidence directory")
    cache_dir = _private_directory(cache_dir, "cache directory")
    env_file = _private_file(env_file, "container environment file")
    image_id = str(candidate.get("local_image_id") or "")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise ValueError("candidate local image identity is invalid")
    candidate_path = evidence_dir / "candidate-image.json"
    if (
        candidate.get("format") != CANDIDATE_FORMAT
        or candidate.get("status") != "pass"
        or not candidate_path.is_file()
        or file_sha(candidate_path) != candidate.get("receipt_sha256")
    ):
        raise ValueError("candidate image receipt binding is invalid")
    name = f"{run_id}-{role}"
    label = f"npa.ncore.qualification={_canonical_sha([run_id, role, image_id])}"
    create = [
        "docker",
        "create",
        "--pull=never",
        "--name",
        name,
        "--label",
        label,
        "--env-file",
        str(env_file),
        "--mount",
        f"type=bind,src={evidence_dir},dst=/evidence",
        "--mount",
        f"type=bind,src={cache_dir},dst=/cache",
        image_id,
        *map(str, command),
    ]
    created_id = ""
    logs = evidence_dir / f"{role}-container"
    try:
        created = _run(create, runner=runner)
        _write_bytes(logs.with_suffix(".create.stdout"), created.stdout)
        _write_bytes(logs.with_suffix(".create.stderr"), created.stderr)
        if created.returncode != 0:
            raise ValueError("candidate container creation failed")
        created_id = created.stdout.decode(errors="strict").strip()
        if re.fullmatch(r"[0-9a-f]{64}", created_id) is None:
            raise ValueError("candidate container identity is invalid")
        before = _inspect(created_id, runner=runner)
        config = before.get("Config")
        labels = config.get("Labels") if isinstance(config, dict) else None
        if (
            before.get("Id") != created_id
            or before.get("Image") != image_id
            or not isinstance(config, dict)
            or config.get("Image") != image_id
            or not isinstance(labels, dict)
            or labels.get("npa.ncore.qualification") != label.split("=", 1)[1]
        ):
            raise ValueError("candidate container is not bound to the checked image")
        started = _run(["docker", "start", "--attach", created_id], runner=runner)
        _write_bytes(logs.with_suffix(".stdout"), started.stdout)
        _write_bytes(logs.with_suffix(".stderr"), started.stderr)
        after = _inspect(created_id, runner=runner)
        state = after.get("State")
        if (
            started.returncode != 0
            or not isinstance(state, dict)
            or state.get("Running") is not False
            or state.get("ExitCode") != 0
            or after.get("Image") != image_id
            or after.get("Created") != before.get("Created")
        ):
            raise ValueError("candidate container command did not pass")
        receipt = {
            "format": EXECUTION_FORMAT,
            "status": "pass",
            "role": role,
            "candidate_image_receipt_sha256": candidate["receipt_sha256"],
            "local_image_id": image_id,
            "command_sha256": _canonical_sha(list(map(str, command))),
            "container_identity_sha256": hashlib.sha256(
                created_id.encode()
            ).hexdigest(),
            "container_created_sha256": hashlib.sha256(
                str(before.get("Created") or "").encode()
            ).hexdigest(),
            "exit_code": 0,
            "pull_policy": "never",
        }
        write_json(evidence_dir / f"{role}-execution.json", receipt)
        return receipt
    finally:
        if created_id:
            removed = _run(
                ["docker", "container", "rm", "--force", created_id], runner=runner
            )
            _write_bytes(logs.with_suffix(".remove.stdout"), removed.stdout)
            _write_bytes(logs.with_suffix(".remove.stderr"), removed.stderr)
            if removed.returncode != 0:
                raise ValueError("candidate container retirement failed")


def run_candidate_qualification(
    *,
    run_id: str,
    candidate_path: Path,
    evidence_dir: Path,
    cache_dir: Path,
    env_file: Path,
    input_path: str,
    control_output_path: str,
    conversion_path: str,
    audit_output_path: str,
    expected_archive_sha256: str,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> dict[str, Any]:
    """Run the negative control, positive conversion, and independent audit."""
    candidate = W.bound_json(
        P.binding(_private_file(candidate_path, "candidate receipt"))
    )
    candidate["receipt_sha256"] = file_sha(candidate_path)
    commands = {
        "wrong-source": [
            "/opt/venv/bin/npa",
            "workbench",
            "nurec",
            "control-source",
            "--input-path",
            input_path,
            "--output-path",
            control_output_path,
            "--expected-archive-sha256",
            expected_archive_sha256,
            "--cache-dir",
            "/cache",
            "--receipt-path",
            "/evidence/wrong-source.json",
            "--output-format",
            "json",
        ],
        "convert": [
            "/opt/venv/bin/npa",
            "workbench",
            "nurec",
            "convert-colmap",
            "--input-path",
            input_path,
            "--output-path",
            conversion_path,
            "--expected-archive-sha256",
            expected_archive_sha256,
            "--cache-dir",
            "/cache",
            "--dataset-root",
            "struktur28",
            "--colmap-dir",
            "sparse/0",
            "--images-dir",
            "images",
            "--rig-mode",
            "derive",
            "--output-format",
            "json",
        ],
        "audit": [
            "/opt/venv/bin/npa",
            "workbench",
            "nurec",
            "audit-colmap",
            "--input-path",
            input_path,
            "--conversion-path",
            conversion_path,
            "--output-path",
            audit_output_path,
            "--expected-archive-sha256",
            expected_archive_sha256,
            "--cache-dir",
            "/cache",
            "--dataset-root",
            "struktur28",
            "--colmap-dir",
            "sparse/0",
            "--images-dir",
            "images",
            "--rig-mode",
            "derive",
            "--output-format",
            "json",
        ],
    }
    executions = [
        run_attested_container(
            role=role,
            run_id=run_id,
            candidate=candidate,
            command=command,
            evidence_dir=evidence_dir,
            cache_dir=cache_dir,
            env_file=env_file,
            runner=runner,
        )
        for role, command in commands.items()
    ]
    control = W.bound_json(P.binding(evidence_dir / "wrong-source.json"))
    if (
        control.get("format") != "npa_ncore_wrong_source_control_v1"
        or control.get("status") != "pass"
        or control.get("native_started") is not False
        or type(control.get("before_output_objects")) is not int
        or control["before_output_objects"] != 0
        or type(control.get("after_output_objects")) is not int
        or control["after_output_objects"] != 0
    ):
        raise ValueError("wrong-source control receipt differs")
    receipt = {
        "format": QUALIFICATION_FORMAT,
        "status": "pass",
        "candidate_image_receipt_sha256": candidate["receipt_sha256"],
        "local_image_id": candidate["local_image_id"],
        "observed_image_digest": candidate["image_digest"],
        "wrong_source_receipt_sha256": file_sha(evidence_dir / "wrong-source.json"),
        "executions": {
            item["role"]: file_sha(evidence_dir / f"{item['role']}-execution.json")
            for item in executions
        },
        "chronology": ["wrong-source", "convert", "audit"],
    }
    write_json(evidence_dir / "qualification-execution.json", receipt)
    return receipt
