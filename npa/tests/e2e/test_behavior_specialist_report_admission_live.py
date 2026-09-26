"""Reject specialist reporting through the real CLI before any durable claim."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit

import pytest

from npa.clients.storage import StorageClient

pytestmark = pytest.mark.e2e
_CONFIG = "NPA_BEHAVIOR_SPECIALIST_ADMISSION_LIVE_CONFIG"


def _settings() -> dict:
    selected = os.environ.get(_CONFIG)
    if not selected:
        pytest.skip(f"requires operator-owned {_CONFIG}")
    path = Path(selected)
    assert path.is_file() and not path.is_symlink()
    assert path.stat().st_mode & 0o077 == 0, "live configuration must be owner-only"
    value = json.loads(path.read_bytes())
    required = {
        "panel_uri",
        "partition_uri",
        "absence_prefix",
        "upstream_root",
    }
    assert set(value) == required
    return value


def _assert_empty_prefix(settings: dict) -> None:
    target = urlsplit(settings["absence_prefix"])
    assert target.scheme == "s3" and target.netloc and target.path.strip("/")
    assert not target.query and not target.fragment
    response = StorageClient.from_environment().s3.list_objects_v2(
        Bucket=target.netloc,
        Prefix=target.path.strip("/").rstrip("/") + "/",
        MaxKeys=1,
    )
    assert response.get("KeyCount", 0) == 0


def _base_command(settings: dict, workspace: Path) -> list[str]:
    prefix = settings["absence_prefix"].rstrip("/")
    command = [
        sys.executable,
        "-m",
        "npa.workflows.behavior_challenge",
        "campaign-worker",
        "--panel-uri",
        settings["panel_uri"],
        "--partition-uri",
        settings["partition_uri"],
        "--worker-index",
        "0",
        "--workspace",
        str(workspace),
        "--output-path",
        f"{prefix}/state",
        "--worker-receipt-uri",
        f"{prefix}/worker.json",
    ]
    command.extend(_runtime_arguments(settings, workspace))
    return command


def _runtime_arguments(settings: dict, workspace: Path) -> list[str]:
    return [
        "--upstream-root",
        settings["upstream_root"],
        "--evaluator-python",
        sys.executable,
        "--data-root",
        str(workspace / "unused-data"),
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--policy-kind",
        "rlc-specialist",
        "--policy-execution-variant",
        "native",
    ]


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    source = Path(__file__).parents[2] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(source), environment.get("PYTHONPATH", "")) if item
    )
    return subprocess.run(command, text=True, capture_output=True, env=environment)


def _invalid_admission(path: Path) -> tuple[Path, str]:
    path.write_text('{"schema":"intentionally-invalid-negative-evidence"}\n')
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _shaped_equivalence(path: Path) -> tuple[Path, str]:
    value = {
        "schema": "npa.behavior.rlc-specialist-equivalence.v1",
        "status": "reviewed_legacy_v1_to_runtime_v2_equivalent",
        "legacy_policy": {},
        "legacy_authorization": {},
        "runtime_identity": {},
        "control_plane_changes": [],
        "claims": {},
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["receipt_sha256"] = hashlib.sha256(encoded).hexdigest()
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_live_cli_rejects_missing_and_mismatched_evidence_before_claim(tmp_path):
    """Download real declarations but reject before policy, evaluator, or storage writes."""
    settings = _settings()
    _assert_empty_prefix(settings)
    missing_workspace = tmp_path / "missing"
    missing = _run(_base_command(settings, missing_workspace))
    assert missing.returncode != 0
    assert "requires frozen equivalence and admission receipts" in missing.stderr
    assert sorted(path.name for path in missing_workspace.iterdir()) == [
        "panel.json",
        "partition.json",
    ]
    _assert_empty_prefix(settings)

    equivalent, equivalent_sha = _shaped_equivalence(tmp_path / "equivalence.json")
    invalid, invalid_sha = _invalid_admission(tmp_path / "invalid-admission.json")
    mismatched_workspace = tmp_path / "mismatched"
    command = _base_command(settings, mismatched_workspace)
    command.extend(
        [
            "--policy-specialist-equivalence-receipt",
            str(equivalent),
            "--policy-specialist-equivalence-sha256",
            equivalent_sha,
            "--policy-specialist-report-admission",
            str(invalid),
            "--policy-specialist-report-admission-sha256",
            invalid_sha,
        ]
    )
    mismatched = _run(command)
    assert mismatched.returncode != 0
    assert "Specialist report admission fields differ" in mismatched.stderr
    assert sorted(path.name for path in mismatched_workspace.iterdir()) == [
        "panel.json",
        "partition.json",
    ]
    _assert_empty_prefix(settings)
