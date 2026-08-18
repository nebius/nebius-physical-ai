from __future__ import annotations

import io
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from npa.workbench.antioch.dataset import AntiochDatasetError, validate_episode
from npa.workbench.antioch.manager import AntiochManager
from npa.workbench.antioch.project import (
    AntiochProjectError,
    deterministic_project_id,
    stage_project,
)
from npa.workbench.antioch.redaction import REDACTED, redact_payload, redact_text
from npa.workbench.antioch.schemas import (
    EpisodeProvenance,
    ProjectArchive,
    ProjectManifest,
    ResumeRequest,
    SubmitRequest,
)
from npa.workbench.antioch.storage import (
    StateStore,
    StoragePreconditionFailed,
    canonical_json,
    join_uri,
    sha256_bytes,
)
from npa.workbench.antioch.service import create_app
from npa.workbench.antioch.vendor_cli import AntiochCli, AntiochCliError


class MemoryStorage:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.counter = 0

    def read_bytes_with_etag(self, uri: str):  # noqa: ANN201
        return self.objects.get(uri)

    def put_bytes_conditional(
        self,
        payload: bytes,
        uri: str,
        *,
        if_match: str = "",
        if_none_match: bool = False,
        content_type: str = "",
    ) -> str:
        del content_type
        current = self.objects.get(uri)
        if if_none_match and current is not None:
            raise StoragePreconditionFailed(uri)
        if if_match and (current is None or current[1] != if_match):
            raise StoragePreconditionFailed(uri)
        self.counter += 1
        etag = f'"{self.counter}"'
        self.objects[uri] = (bytes(payload), etag)
        return etag


def _submit() -> SubmitRequest:
    return SubmitRequest(
        input_path="s3://safe/input",
        output_path="s3://safe/run",
        workflow_run="run-1",
        state_id="simulate",
        suite="smoke",
    )


def test_redaction_covers_nested_credentials_and_signed_urls() -> None:
    payload = redact_payload(
        {
            "token": "top-secret",
            "nested": {"Authorization": "Bearer abc.def.ghi"},
            "url": "https://x/a?X-Amz-Signature=secret&ok=1",
        }
    )
    rendered = json.dumps(payload)
    assert (
        "top-secret" not in rendered
        and "abc.def.ghi" not in rendered
        and "secret" not in rendered
    )
    assert REDACTED in rendered
    assert "eyJhbGciOiJIUzI1NiJ9.e30.signature" not in redact_text(
        "eyJhbGciOiJIUzI1NiJ9.e30.signature"
    )


@pytest.mark.parametrize(
    "status,retryable", [(401, False), (429, True), (500, True), (503, True)]
)
def test_structured_cli_error_classification(
    monkeypatch: pytest.MonkeyPatch, status: int, retryable: bool
) -> None:
    error = {
        "error": {
            "type": "http",
            "message": "Bearer sensitive",
            "http_status": status,
            "exit_code": 1,
        }
    }
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, "", json.dumps(error)),
    )
    with pytest.raises(AntiochCliError) as raised:
        AntiochCli("antioch").show(Path("."), kind="scenario", remote_id="r")
    assert raised.value.retryable is retryable
    assert "sensitive" not in str(raised.value)


def test_cli_rejects_malformed_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, "not-json", ""),
    )
    with pytest.raises(AntiochCliError, match="malformed"):
        AntiochCli("antioch").show(Path("."), kind="scenario", remote_id="r")


def _project_bundle(
    tmp_path: Path, member_name: str = "project/antioch.yaml"
) -> tuple[bytes, bytes]:
    archive = tmp_path / "project.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        content = b"id: original\nname: Synthetic\nservices: {}\n"
        info = tarfile.TarInfo(member_name)
        info.size = len(content)
        bundle.addfile(info, io.BytesIO(content))
    raw = archive.read_bytes()
    manifest = ProjectManifest(
        archive=ProjectArchive(size_bytes=len(raw), sha256=sha256_bytes(raw)),
        source_name="synthetic-cartpole",
        source_revision="1",
        source_license="CC0-1.0",
        source_sha256="a" * 64,
    )
    return canonical_json(manifest.model_dump(mode="json")), raw


class ProjectStorage:
    def __init__(self, manifest: bytes, archive: bytes) -> None:
        self.manifest = manifest
        self.archive = archive

    def read_bytes_with_etag(self, uri: str):  # noqa: ANN201
        return self.manifest, '"1"'

    def download_file(self, uri: str, local: str) -> str:
        Path(local).write_bytes(self.archive)
        return local


def test_project_staging_is_immutable_and_deterministic(tmp_path: Path) -> None:
    manifest, archive = _project_bundle(tmp_path)
    root, _source, digest = stage_project(
        ProjectStorage(manifest, archive),
        "s3://safe/input",
        tmp_path / "stage",
        project_id="npa-safe-id",
    )
    assert digest == sha256_bytes(archive)
    assert (root / "antioch.yaml").read_text().startswith("id: npa-safe-id")
    assert deterministic_project_id("run", "state") == deterministic_project_id(
        "run", "state"
    )


def test_project_staging_rejects_traversal(tmp_path: Path) -> None:
    manifest, archive = _project_bundle(tmp_path, "../antioch.yaml")
    with pytest.raises(AntiochProjectError, match="unsafe"):
        stage_project(
            ProjectStorage(manifest, archive),
            "s3://safe/input",
            tmp_path / "stage",
            project_id="npa-safe",
        )


def _episode(path: Path, **replacements: Any) -> None:
    length = 4
    provenance = EpisodeProvenance(
        scenario="cartpole",
        case="balance",
        seed=7,
        parameters={"mass": 1.0},
        engine_version="1",
        sdk_version="0.3.47",
        source_sha256="a" * 64,
        assets_sha256={"cart": "b" * 64},
        observation_schema=["position", "velocity"],
        action_schema=["force"],
        fps=20,
    )
    arrays: dict[str, Any] = {
        "observation_state": np.zeros((length, 2), dtype=np.float32),
        "observation_image_workspace": np.zeros((length, 8, 8, 3), dtype=np.uint8),
        "observation_image_wrist": np.zeros((length, 8, 8, 3), dtype=np.uint8),
        "action": np.zeros((length, 1), dtype=np.float32),
        "reward": np.ones(length, dtype=np.float32),
        "terminated": np.array([False, False, False, True]),
        "truncated": np.zeros(length, dtype=bool),
        "timestamp": np.arange(length, dtype=np.float64) / 20,
        "provenance": np.array(provenance.model_dump_json()),
    }
    arrays.update(replacements)
    np.savez(path, **arrays)


def test_episode_contract_accepts_complete_data(tmp_path: Path) -> None:
    path = tmp_path / "episode.npz"
    _episode(path)
    arrays, provenance = validate_episode(path)
    assert arrays["action"].shape == (4, 1)
    assert provenance.seed == 7


@pytest.mark.parametrize(
    "replacement",
    [
        {"timestamp": np.array([0.0, 0.1, 0.1, 0.2])},
        {"terminated": np.zeros(4, dtype=bool)},
        {"action": np.zeros((3, 1))},
    ],
)
def test_episode_contract_fails_closed_on_incompatible_data(
    tmp_path: Path, replacement: dict[str, Any]
) -> None:
    path = tmp_path / "episode.npz"
    _episode(path, **replacement)
    with pytest.raises(AntiochDatasetError):
        validate_episode(path)


def test_episode_contract_rejects_partial_bundle(tmp_path: Path) -> None:
    path = tmp_path / "partial.npz"
    np.savez(path, action=np.zeros((2, 1)))
    with pytest.raises(AntiochDatasetError, match="missing required"):
        validate_episode(path)


class FakeCli:
    def __init__(self) -> None:
        self.submissions = 0
        self.cancellations = 0

    def list_for_project(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return []

    def submit_suite(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        self.submissions += 1
        return {"suite_run_id": "suite-safe", "invocation_id": "invoke-safe"}

    def show(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return {"suite_run_id": "suite-safe", "phase": "running"}

    def cancel(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        self.cancellations += 1
        return {"suite_run_id": "suite-safe", "phase": "cancelled"}


def test_idempotent_retry_restart_reconcile_and_cancel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    memory = MemoryStorage()
    cli = FakeCli()
    monkeypatch.setattr(
        "npa.workbench.antioch.manager.stage_project",
        lambda *a, **k: (tmp_path, object(), "c" * 64),
    )
    first = AntiochManager.__new__(AntiochManager)
    first.storage = memory
    first.states = StateStore(memory)
    first._cli = lambda *a, **k: cli  # type: ignore[method-assign]
    record = first.submit(_submit())
    assert record.remote_id == "suite-safe" and cli.submissions == 1
    assert first.submit(_submit()).remote_id == "suite-safe" and cli.submissions == 1

    restarted = AntiochManager.__new__(AntiochManager)
    restarted.storage = memory
    restarted.states = StateStore(memory)
    restarted._cli = lambda *a, **k: cli  # type: ignore[method-assign]
    resume = ResumeRequest(
        output_path="s3://safe/run", workflow_run="run-1", state_id="simulate"
    )
    assert restarted.reconcile(resume).status == "running"
    assert restarted.cancel(resume).status == "cancelled"
    assert restarted.cancel(resume).status == "cancelled"
    assert cli.cancellations == 1


def test_atomic_immutable_completion_conflict() -> None:
    memory = MemoryStorage()
    states = StateStore(memory)
    uri = join_uri("s3://safe/run", "_SUCCESS.json")
    states.put_immutable_json(uri, {"schema": "v1", "ok": True})
    states.put_immutable_json(uri, {"schema": "v1", "ok": True})
    with pytest.raises(Exception, match="different content"):
        states.put_immutable_json(uri, {"schema": "v1", "ok": False})


def test_service_auth_fails_closed_without_exposing_token() -> None:
    client = TestClient(
        create_app(manager=object(), auth_mode="token", token="service-secret")
    )
    denied = client.get("/system-info")
    assert denied.status_code == 401
    assert "service-secret" not in denied.text
    allowed = client.get(
        "/system-info", headers={"Authorization": "Bearer service-secret"}
    )
    assert allowed.status_code == 200
    assert allowed.json()["cpu_only"] is True
