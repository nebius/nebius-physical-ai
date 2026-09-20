from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))

from image_byte_scan import core as W  # noqa: E402
from ncore_publication import qualification  # noqa: E402


SOURCE_SHA = "a" * 40
LOCAL_ID = "sha256:" + "1" * 64
IMAGE_DIGEST = "sha256:" + "2" * 64
PLATFORM_DIGEST = "sha256:" + "3" * 64
CONFIG_DIGEST = "sha256:" + "4" * 64
ARCHIVE_SHA = "5" * 64


def _private(path: Path, value: object) -> Path:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    return path


def _relationship() -> dict:
    return {
        "local_image_id": LOCAL_ID,
        "identity_kind": "docker-v2-manifest",
        "image_digest": IMAGE_DIGEST,
        "platform_digest": PLATFORM_DIGEST,
        "config_digest": CONFIG_DIGEST,
        "layers": [],
        "archive_sha256": ARCHIVE_SHA,
        "inspection_sha256": "6" * 64,
        "export_sha256": "7" * 64,
    }


def test_candidate_binding_recomputes_checked_local_identity(
    monkeypatch, tmp_path: Path
) -> None:
    analysis = tmp_path / "analysis"
    gate = analysis / "check"
    gate.mkdir(mode=0o700, parents=True)
    (analysis / "build").mkdir(mode=0o700)
    _private(gate / "inspection.json", {"config_digest": CONFIG_DIGEST})
    _private(gate / "loaded-image.json", [{"Id": LOCAL_ID}])
    (gate / "loaded-image.tar").write_bytes(b"local export")
    (gate / "loaded-image.tar").chmod(0o600)
    _private(gate / "local-image-binding.json", _relationship())
    _private(
        analysis / "build/build.json",
        {
            "schema": "npa.ncore.committed-oci-build.v1",
            "source_sha": SOURCE_SHA,
            "image_digest": IMAGE_DIGEST,
            "archive_sha256": ARCHIVE_SHA,
        },
    )
    monkeypatch.setattr(
        qualification.artifact,
        "verify_local_export",
        lambda path, inspected, expected: _relationship(),
    )
    output = analysis / "candidate.json"
    with W.authorized_roots(tmp_path, ROOT):
        receipt = qualification.bind_candidate_image(
            source_sha=SOURCE_SHA,
            analysis_root=analysis,
            gate_dir=gate,
            output_path=output,
        )
    assert receipt["local_image_id"] == LOCAL_ID
    assert receipt["image_digest"] == IMAGE_DIGEST
    assert (
        receipt["local_image_binding_sha256"]
        == hashlib.sha256((gate / "local-image-binding.json").read_bytes()).hexdigest()
    )


def test_candidate_qualification_uses_three_inspected_pull_disabled_containers(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    cache = tmp_path / "cache"
    evidence.mkdir(mode=0o700)
    cache.mkdir(mode=0o700)
    env_file = tmp_path / "s3.env"
    env_file.write_text("AWS_ACCESS_KEY_ID=fixture\n")
    env_file.chmod(0o600)
    candidate = {
        "format": qualification.CANDIDATE_FORMAT,
        "status": "pass",
        "source_sha": SOURCE_SHA,
        "local_image_id": LOCAL_ID,
        "identity_kind": "docker-v2-manifest",
        "image_digest": IMAGE_DIGEST,
        "platform_digest": PLATFORM_DIGEST,
        "config_digest": CONFIG_DIGEST,
        "archive_sha256": ARCHIVE_SHA,
    }
    candidate_path = _private(evidence / "candidate-image.json", candidate)
    created = {}
    started: set[str] = set()
    commands = []

    def runner(command, **_kwargs):
        commands.append(command)
        if command[:2] == ["docker", "create"]:
            container_id = f"{len(created) + 1:064x}"
            name = command[command.index("--name") + 1]
            created[container_id] = {"name": name, "command": command}
            return subprocess.CompletedProcess(
                command, 0, container_id.encode() + b"\n", b""
            )
        if command[:3] == ["docker", "container", "inspect"]:
            container_id = command[3]
            record = created[container_id]
            payload = [
                {
                    "Id": container_id,
                    "Image": LOCAL_ID,
                    "Created": "immutable-created",
                    "Config": {
                        "Image": LOCAL_ID,
                        "Labels": {
                            "npa.ncore.qualification": command_label(record["command"])
                        },
                    },
                    "State": {
                        "Running": False,
                        "ExitCode": 0 if container_id in started else None,
                    },
                }
            ]
            return subprocess.CompletedProcess(
                command, 0, json.dumps(payload).encode(), b""
            )
        if command[:3] == ["docker", "start", "--attach"]:
            container_id = command[3]
            started.add(container_id)
            if "wrong-source" in created[container_id]["name"]:
                _private(
                    evidence / "wrong-source.json",
                    {
                        "format": "npa_ncore_wrong_source_control_v1",
                        "status": "pass",
                        "native_started": False,
                        "output_objects": 0,
                    },
                )
            return subprocess.CompletedProcess(command, 0, b'{"status":"ok"}\n', b"")
        if command[:3] == ["docker", "container", "rm"]:
            return subprocess.CompletedProcess(command, 0, b"removed\n", b"")
        raise AssertionError(command)

    def command_label(command):
        return command[command.index("--label") + 1].split("=", 1)[1]

    with W.authorized_roots(tmp_path, ROOT):
        receipt = qualification.run_candidate_qualification(
            run_id="qualification-run",
            candidate_path=candidate_path,
            evidence_dir=evidence,
            cache_dir=cache,
            env_file=env_file,
            input_path="s3://fixture/source.zip",
            control_output_path="s3://fixture/control/",
            conversion_path="s3://fixture/conversion/",
            audit_output_path="s3://fixture/audit.json",
            expected_archive_sha256="a" * 64,
            runner=runner,
        )
    creates = [command for command in commands if command[:2] == ["docker", "create"]]
    assert len(creates) == 3
    assert all("--pull=never" in command and LOCAL_ID in command for command in creates)
    assert receipt["chronology"] == ["wrong-source", "convert", "audit"]
    assert set(receipt["executions"]) == {"wrong-source", "convert", "audit"}


def test_candidate_execution_rejects_replaced_image_identity(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    cache = tmp_path / "cache"
    evidence.mkdir(mode=0o700)
    cache.mkdir(mode=0o700)
    env_file = _private(tmp_path / "s3.env", {"fixture": True})
    candidate = {
        "format": qualification.CANDIDATE_FORMAT,
        "status": "pass",
        "local_image_id": LOCAL_ID,
    }
    candidate_path = _private(evidence / "candidate-image.json", candidate)
    candidate["receipt_sha256"] = hashlib.sha256(
        candidate_path.read_bytes()
    ).hexdigest()

    def runner(command, **_kwargs):
        if command[:2] == ["docker", "create"]:
            return subprocess.CompletedProcess(command, 0, b"1" * 64 + b"\n", b"")
        if command[:3] == ["docker", "container", "inspect"]:
            payload = [
                {
                    "Id": "1" * 64,
                    "Image": "sha256:" + "9" * 64,
                    "Created": "immutable-created",
                    "Config": {"Image": LOCAL_ID, "Labels": {}},
                }
            ]
            return subprocess.CompletedProcess(
                command, 0, json.dumps(payload).encode(), b""
            )
        if command[:3] == ["docker", "container", "rm"]:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        raise AssertionError(command)

    with pytest.raises(ValueError, match="not bound"):
        qualification.run_attested_container(
            role="wrong-source",
            run_id="qualification-run",
            candidate=candidate,
            command=["/opt/venv/bin/npa", "--version"],
            evidence_dir=evidence,
            cache_dir=cache,
            env_file=env_file,
            runner=runner,
        )
