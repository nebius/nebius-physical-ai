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
from pydantic import ValidationError
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.antioch.dataset import AntiochDatasetError, validate_episode
from npa.workbench.antioch.manager import (
    AntiochManager,
    AntiochOperationError,
    _dataset_metadata,
    _manifest_artifact_size,
    _validate_downloaded_artifact,
    operation_key,
)
from npa.workbench.antioch.project import (
    AntiochProjectError,
    deterministic_project_id,
    package_project,
    stage_project,
)
from npa.workbench.antioch.redaction import REDACTED, redact_payload, redact_text
from npa.workbench.antioch.schemas import (
    ArtifactRecord,
    CollectRequest,
    EpisodeProvenance,
    OperationRecord,
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
from npa.workbench.antioch.storage_config import (
    DEFAULT_NEBIUS_STORAGE_ENDPOINT,
    resolve_storage_client,
)
from npa.workbench.antioch.vendor_cli import AntiochCli, AntiochCliError
from npa.workbench.antioch.runtime import AntiochRuntimeError


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


@pytest.fixture(autouse=True)
def _accept_antioch_terms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_ANTIOCH_ACCEPT_TERMS", "YES")


def _submit() -> SubmitRequest:
    return SubmitRequest(
        input_path="s3://safe/input",
        output_path="s3://safe/run",
        workflow_run="run-1",
        state_id="simulate",
        robot_type="dual-camera-cart",
        task="Move the cart to the requested target",
        suite="smoke",
    )


def test_submit_metadata_is_required_and_non_cartpole_values_are_preserved() -> None:
    with pytest.raises(ValidationError):
        SubmitRequest(
            input_path="s3://safe/input",
            output_path="s3://safe/run",
            workflow_run="run-1",
            state_id="simulate",
            suite="smoke",
        )
    request = _submit()
    assert request.robot_type == "dual-camera-cart"
    assert request.task == "Move the cart to the requested target"


def test_collection_fails_closed_for_legacy_state_without_dataset_metadata() -> None:
    request = _submit()
    record = OperationRecord(
        idempotency_key="a" * 64,
        request_sha256="b" * 64,
        workflow_run=request.workflow_run,
        state_id=request.state_id,
        input_path=request.input_path,
        output_path=request.output_path,
        derived_project_id="npa-legacy",
        remote_kind="suite",
        selection=request.suite,
    )
    with pytest.raises(AntiochOperationError) as raised:
        _dataset_metadata(record)
    assert raised.value.error_type == "dataset_metadata_missing"


def test_workload_identity_storage_resolver_needs_no_static_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def fake_from_environment(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        return object()

    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("NEBIUS_S3_ENDPOINT", raising=False)
    monkeypatch.delenv("NPA_STORAGE_ENDPOINT", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setattr(
        "npa.workbench.antioch.storage_config.StorageClient.from_environment",
        fake_from_environment,
    )
    assert resolve_storage_client(host_resolver=lambda: None) is not None
    assert captured == {"endpoint_url": DEFAULT_NEBIUS_STORAGE_ENDPOINT}


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
    assert "a" * 32 not in redact_text(f"scenario run {'a' * 32} was not found")
    assert "b" * 64 not in redact_text(f"container {'b' * 64} is stopping")
    assert (
        "01234567-89ab-cdef-0123-456789abcdef"
        not in redact_text("run 01234567-89ab-cdef-0123-456789abcdef")
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


@pytest.mark.parametrize(
    "invoke,payload",
    [
        (lambda cli: cli.submit_suite(Path("."), "suite"), []),
        (lambda cli: cli.submit_scenario(Path("."), "scenario"), {}),
        (lambda cli: cli.list_for_project(Path("."), kind="suite", project_id="p"), {"items": {}}),
        (lambda cli: cli.download(Path("."), scenario_run_id="r", output=Path(".")), {"files": {}}),
        (lambda cli: cli.logs(Path("."), scenario_run_id="r"), []),
        (lambda cli: cli.services_up(Path(".")), []),
        (lambda cli: cli.services_down(Path(".")), []),
        (lambda cli: cli.machine_status(Path("."), project_id="p"), []),
        (lambda cli: cli.machine_release(Path("."), project_id="p"), []),
    ],
)
def test_vendor_cli_rejects_malformed_response_shapes_directly(
    invoke, payload: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **_kwargs: subprocess.CompletedProcess(
            args, 0, json.dumps(payload), ""
        ),
    )
    with pytest.raises(AntiochCliError) as raised:
        invoke(AntiochCli("antioch"))
    assert raised.value.error_type == "malformed_cli_output"


def test_supported_live_service_commands_never_inline_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def run(args, **_kwargs):  # noqa: ANN001, ANN202
        calls.append(list(args))
        payload = "{}" if "--json" in args else "ok"
        return subprocess.CompletedProcess(args, 0, payload, "")

    monkeypatch.setattr(subprocess, "run", run)
    cli = AntiochCli("antioch")
    cli.services_build(tmp_path, service="sim")
    cli.services_up(tmp_path)
    cli.services_exec(
        tmp_path, "sim", ["install", "-d", "-m", "0700", "/workspace/client"]
    )
    cli.services_copy(tmp_path, tmp_path / "api-key", "sim:/workspace/client/api-key")
    cli.services_down(tmp_path)
    cli.machine_release(tmp_path, project_id="assigned-project-for-test")

    assert calls == [
        ["antioch", "services", "build", "--service", "sim", "--json"],
        ["antioch", "services", "up", "--json"],
        [
            "antioch",
            "services",
            "exec",
            "sim",
            "install",
            "-d",
            "-m",
            "0700",
            "/workspace/client",
        ],
        [
            "antioch",
            "services",
            "cp",
            str(tmp_path / "api-key"),
            "sim:/workspace/client/api-key",
            "--json",
        ],
        ["antioch", "services", "down", "--json"],
        [
            "antioch",
            "machine",
            "release",
            "--project",
            "assigned-project-for-test",
            "--yes",
            "--json",
        ],
    ]


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


def test_project_packaging_is_reproducible_and_excludes_caches(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "antioch.yaml").write_text("id: example\nservices: {}\n")
    (source / "scenario.py").write_text("VALUE = 1\n")
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "scenario.pyc").write_bytes(b"cache")
    kwargs = {
        "source_name": "synthetic",
        "source_revision": "1",
        "source_license": "CC0-1.0",
        "source_sha256": "a" * 64,
    }
    first = package_project(source, tmp_path / "first", **kwargs)
    second = package_project(source, tmp_path / "second", **kwargs)
    assert first.archive.sha256 == second.archive.sha256
    with tarfile.open(tmp_path / "first" / "project.tar.gz") as bundle:
        assert "scenario.py" in bundle.getnames()
        assert not any("__pycache__" in name for name in bundle.getnames())


def _episode(path: Path, **replacements: Any) -> None:
    length = 4
    provenance = EpisodeProvenance(
        scenario="cartpole",
        case="balance",
        seed=7,
        parameters={"mass": 1.0},
        engine_version="1",
        sdk_version="0.3.63",
        source_sha256="a" * 64,
        assets_sha256={"cart": "b" * 64},
        observation_schema=["position", "velocity"],
        action_schema=["force_positive", "force_negative"],
        fps=20,
    )
    arrays: dict[str, Any] = {
        "observation_state": np.zeros((length, 2), dtype=np.float32),
        "observation_image_workspace": np.zeros((length, 8, 8, 3), dtype=np.uint8),
        "observation_image_wrist": np.zeros((length, 8, 8, 3), dtype=np.uint8),
        "action": np.zeros((length, 2), dtype=np.float32),
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
    assert arrays["action"].shape == (4, 2)
    assert provenance.seed == 7


@pytest.mark.parametrize(
    "replacement",
    [
        {"timestamp": np.array([0.0, 0.1, 0.1, 0.2])},
        {"terminated": np.zeros(4, dtype=bool)},
        {"action": np.zeros((3, 2))},
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


def test_episode_contract_rejects_single_channel_act_data(tmp_path: Path) -> None:
    path = tmp_path / "single-action.npz"
    provenance = EpisodeProvenance(
        scenario="cartpole",
        case="balance",
        seed=7,
        parameters={},
        engine_version="1",
        sdk_version="0.3.63",
        source_sha256="a" * 64,
        assets_sha256={"cart": "b" * 64},
        observation_schema=["position", "velocity"],
        action_schema=["force"],
        fps=20,
    )
    _episode(
        path,
        action=np.zeros((4, 1), dtype=np.float32),
        provenance=np.array(provenance.model_dump_json()),
    )
    with pytest.raises(AntiochDatasetError, match="at least two action channels"):
        validate_episode(path)


class FakeCli:
    def __init__(self) -> None:
        self.submissions = 0
        self.cancellations = 0
        self.existing: list[dict[str, str]] = []

    def list_for_project(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return self.existing

    def submit_suite(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        self.submissions += 1
        return {"suite_run_id": "suite-safe", "invocation_id": "invoke-safe"}

    def show(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return {"suite_run_id": "suite-safe", "phase": "running"}

    def cancel(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        self.cancellations += 1
        return {"suite_run_id": "suite-safe", "phase": "cancelled"}

    def rerun(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return {"suite_run_id": "suite-rerun", "invocation_id": "invoke-rerun"}


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
    assert record.robot_type == "dual-camera-cart"
    assert record.task == "Move the cart to the requested target"
    assert record.terms_accepted is True
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


@pytest.mark.parametrize("retryable", [False, True])
def test_cancel_persists_typed_cli_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    retryable: bool,
) -> None:
    memory = MemoryStorage()
    manager = AntiochManager.__new__(AntiochManager)
    manager.storage = memory
    manager.states = StateStore(memory)
    monkeypatch.setattr(
        "npa.workbench.antioch.manager.stage_project",
        lambda *a, **k: (tmp_path, object(), "c" * 64),
    )
    cli = FakeCli()
    manager._cli = lambda *a, **k: cli  # type: ignore[method-assign]
    record = manager.submit(_submit())

    def fail(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AntiochCliError(
            "classified failure",
            error_type="capacity" if retryable else "authentication",
            retryable=retryable,
        )

    cli.cancel = fail  # type: ignore[method-assign]
    request = ResumeRequest(
        output_path=record.output_path,
        workflow_run=record.workflow_run,
        state_id=record.state_id,
    )
    with pytest.raises(AntiochOperationError) as raised:
        manager.cancel(request)
    assert raised.value.retryable is retryable
    durable = manager._record_for(request)
    assert durable.retryable is retryable
    assert durable.error_type == ("capacity" if retryable else "authentication")
    assert durable.status == "running"


@pytest.mark.parametrize("terminal", ["completed", "failed", "cancelled"])
def test_cancel_does_not_clobber_terminal_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, terminal: str
) -> None:
    memory = MemoryStorage()
    manager = AntiochManager.__new__(AntiochManager)
    manager.storage = memory
    manager.states = StateStore(memory)
    monkeypatch.setattr(
        "npa.workbench.antioch.manager.stage_project",
        lambda *a, **k: (tmp_path, object(), "c" * 64),
    )
    cli = FakeCli()
    manager._cli = lambda *a, **k: cli  # type: ignore[method-assign]
    record = manager.submit(_submit())
    immutable = {
        "artifact_manifest_uri": "s3://safe/run/manifests/v1.json",
        "dataset_uri": "s3://safe/run/dataset",
        "completion_uri": "s3://safe/run/_SUCCESS.json",
    }
    record = manager.states.update(record, status=terminal, **immutable)
    request = ResumeRequest(
        output_path=record.output_path,
        workflow_run=record.workflow_run,
        state_id=record.state_id,
    )
    result = manager.cancel(request)
    assert result.status == terminal
    assert {key: getattr(result, key) for key in immutable} == immutable
    assert cli.cancellations == 0


def test_cli_failure_envelope_exposes_retry_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AntiochOperationError("try later", retryable=True, error_type="capacity")

    monkeypatch.setattr("npa.sdk.workbench.antioch.cancel", fail)
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "antioch",
            "cancel",
            "--output-path",
            "s3://safe/run",
            "--workflow-run",
            "run-1",
            "--state-id",
            "simulate",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 1
    envelope = json.loads(result.stderr)
    assert envelope["error"] == {
        "type": "capacity",
        "message": "try later",
        "retryable": True,
        "terminal": False,
    }


def test_concurrent_submitter_is_fenced_and_expired_restart_reconciles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    memory = MemoryStorage()
    manager = AntiochManager.__new__(AntiochManager)
    manager.storage = memory
    manager.states = StateStore(memory)
    request = _submit()
    key = operation_key(request.workflow_run, request.state_id)
    claimed = manager.states.claim(
        OperationRecord(
            idempotency_key=key,
            request_sha256=sha256_bytes(
                canonical_json(request.model_dump(mode="json"))
            ),
            workflow_run=request.workflow_run,
            state_id=request.state_id,
            robot_type=request.robot_type,
            task=request.task,
            input_path=request.input_path,
            output_path=request.output_path,
            derived_project_id=deterministic_project_id(
                request.workflow_run, request.state_id
            ),
            remote_kind="suite",
            selection="smoke",
        )
    )
    leased, acquired = manager.states.acquire_submission(claimed, "first-owner")
    assert acquired
    cli = FakeCli()
    manager._cli = lambda *a, **k: cli  # type: ignore[method-assign]
    assert manager.submit(request).remote_id == ""
    assert cli.submissions == 0

    manager.states.update(
        leased,
        submission_lease_expires_at="2000-01-01T00:00:00Z",
    )
    cli.existing = [{"suite_run_id": "suite-recovered"}]
    monkeypatch.setattr(
        "npa.workbench.antioch.manager.stage_project",
        lambda *a, **k: (tmp_path, object(), "c" * 64),
    )
    recovered = manager.submit(request)
    assert recovered.remote_id == "suite-recovered"
    assert cli.submissions == 0


def test_atomic_immutable_completion_conflict() -> None:
    memory = MemoryStorage()
    states = StateStore(memory)
    uri = join_uri("s3://safe/run", "_SUCCESS.json")
    states.put_immutable_json(uri, {"schema": "v1", "ok": True})
    states.put_immutable_json(uri, {"schema": "v1", "ok": True})
    with pytest.raises(Exception, match="different content"):
        states.put_immutable_json(uri, {"schema": "v1", "ok": False})


def test_stale_cancel_update_cannot_overwrite_concurrent_completion() -> None:
    memory = MemoryStorage()
    states = StateStore(memory)
    request = _submit()
    stale = states.claim(
        OperationRecord(
            idempotency_key=operation_key(request.workflow_run, request.state_id),
            request_sha256=sha256_bytes(
                canonical_json(request.model_dump(mode="json"))
            ),
            workflow_run=request.workflow_run,
            state_id=request.state_id,
            robot_type=request.robot_type,
            task=request.task,
            input_path=request.input_path,
            output_path=request.output_path,
            derived_project_id="npa-race-test",
            remote_kind="suite",
            selection=request.suite,
            status="running",
        )
    )
    immutable = {
        "artifact_manifest_uri": "s3://safe/run/manifests/v1.json",
        "dataset_uri": "s3://safe/run/dataset",
        "completion_uri": "s3://safe/run/_SUCCESS.json",
    }
    states.update(stale, status="completed", **immutable)
    result = states.update(stale, status="cancelled", remote_phase="cancelled")
    assert result.status == "completed"
    assert {key: getattr(result, key) for key in immutable} == immutable


@pytest.mark.parametrize("terminal", ["failed", "cancelled"])
@pytest.mark.parametrize("requested", ["submitted", "queued", "running", "completed"])
def test_failed_or_cancelled_state_rejects_nonterminal_and_terminal_revival(
    terminal: str, requested: str
) -> None:
    memory = MemoryStorage()
    states = StateStore(memory)
    request = _submit()
    record = states.claim(
        OperationRecord(
            idempotency_key=operation_key(request.workflow_run, request.state_id),
            request_sha256="a" * 64,
            workflow_run=request.workflow_run,
            state_id=request.state_id,
            robot_type=request.robot_type,
            task=request.task,
            input_path=request.input_path,
            output_path=request.output_path,
            derived_project_id="npa-terminal-test",
            remote_kind="suite",
            selection=request.suite,
            status=terminal,
        )
    )

    result = states.update(record, status=requested, remote_phase=requested)

    assert result.status == terminal
    assert result.remote_phase == ""


@pytest.mark.parametrize(
    "terminal,other_terminal", [("failed", "cancelled"), ("cancelled", "failed")]
)
def test_distinct_failed_and_cancelled_terminal_transition_is_rejected(
    terminal: str, other_terminal: str
) -> None:
    memory = MemoryStorage()
    states = StateStore(memory)
    request = _submit()
    record = states.claim(
        OperationRecord(
            idempotency_key=operation_key(request.workflow_run, request.state_id),
            request_sha256="a" * 64,
            workflow_run=request.workflow_run,
            state_id=request.state_id,
            robot_type=request.robot_type,
            task=request.task,
            input_path=request.input_path,
            output_path=request.output_path,
            derived_project_id="npa-distinct-terminal-test",
            remote_kind="suite",
            selection=request.suite,
            status=terminal,
        )
    )
    assert states.update(record, status=other_terminal).status == terminal


def test_completed_state_allows_only_atomic_collection_round_trip() -> None:
    memory = MemoryStorage()
    states = StateStore(memory)
    request = _submit()
    completed = states.claim(
        OperationRecord(
            idempotency_key=operation_key(request.workflow_run, request.state_id),
            request_sha256="a" * 64,
            workflow_run=request.workflow_run,
            state_id=request.state_id,
            robot_type=request.robot_type,
            task=request.task,
            input_path=request.input_path,
            output_path=request.output_path,
            derived_project_id="npa-collection-test",
            remote_kind="suite",
            selection=request.suite,
            status="completed",
        )
    )
    assert states.update(completed, status="running").status == "completed"
    collecting, acquired = states.begin_collection(completed)
    assert acquired is True
    assert collecting.status == "collecting"
    duplicate, acquired = states.begin_collection(completed)
    assert acquired is False
    assert duplicate.status == "collecting"
    returned = states.update(collecting, status="completed")
    assert returned.status == "completed"
    same = states.update(returned, status="completed", remote_phase="complete")
    assert same.status == "completed"
    assert same.remote_phase == "complete"
    assert states.update(same, status="cancelled").status == "completed"


def test_collection_lease_excludes_active_owner_and_recovers_expired_owner() -> None:
    memory = MemoryStorage()
    states = StateStore(memory)
    request = _submit()
    record = states.claim(
        OperationRecord(
            idempotency_key=operation_key(request.workflow_run, request.state_id),
            request_sha256="a" * 64,
            workflow_run=request.workflow_run,
            state_id=request.state_id,
            robot_type=request.robot_type,
            task=request.task,
            input_path=request.input_path,
            output_path=request.output_path,
            derived_project_id="npa-collection-lease",
            remote_kind="suite",
            selection=request.suite,
            status="completed",
        )
    )
    active, acquired = states.acquire_collection(record, "owner-a")
    assert acquired is True
    excluded, acquired = states.acquire_collection(active, "owner-b")
    assert acquired is False
    assert excluded.collection_owner == "owner-a"
    expired = states.update(active, collection_lease_expires_at="2000-01-01T00:00:00Z")
    recovered, acquired = states.acquire_collection(expired, "owner-b")
    assert acquired is True
    assert recovered.collection_owner == "owner-b"
    assert recovered.collection_phase == "claimed"


def test_collect_excludes_active_owner_and_executes_after_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = MemoryStorage()
    manager = AntiochManager.__new__(AntiochManager)
    manager.storage = memory
    manager.states = StateStore(memory)
    submitted = _submit()
    completed = manager.states.claim(
        OperationRecord(
            idempotency_key=operation_key(submitted.workflow_run, submitted.state_id),
            request_sha256="a" * 64,
            workflow_run=submitted.workflow_run,
            state_id=submitted.state_id,
            robot_type=submitted.robot_type,
            task=submitted.task,
            input_path=submitted.input_path,
            output_path=submitted.output_path,
            derived_project_id="npa-collect-expiry",
            remote_kind="suite",
            selection=submitted.suite,
            remote_id="suite-safe",
            status="completed",
        )
    )
    request = CollectRequest(
        output_path=completed.output_path,
        workflow_run=completed.workflow_run,
        state_id=completed.state_id,
    )
    active, acquired = manager.states.acquire_collection(completed, "active-owner")
    assert acquired is True
    monkeypatch.setattr(
        "npa.workbench.antioch.manager.stage_project",
        lambda *_args, **_kwargs: pytest.fail("active collector was duplicated"),
    )
    with pytest.raises(AntiochOperationError) as excluded:
        manager.collect(request)
    assert excluded.value.error_type == "collection_in_progress"
    assert excluded.value.retryable is True

    manager.states.update(active, collection_lease_expires_at="2000-01-01T00:00:00Z")

    def recovered_body(*_args, **_kwargs):  # noqa: ANN202
        raise RuntimeError("entered recovered collection body")

    monkeypatch.setattr(
        "npa.workbench.antioch.manager.stage_project", recovered_body
    )
    with pytest.raises(RuntimeError, match="entered recovered collection body"):
        manager.collect(request)
    durable = manager._record_for(request)
    assert durable.status == "completed"
    assert durable.collection_owner == ""
    assert durable.retryable is True


def test_late_submit_after_cancel_never_calls_vendor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = MemoryStorage()
    manager = AntiochManager.__new__(AntiochManager)
    manager.storage = memory
    manager.states = StateStore(memory)
    request = _submit()
    key = operation_key(request.workflow_run, request.state_id)
    record = manager.states.claim(
        OperationRecord(
            idempotency_key=key,
            request_sha256=sha256_bytes(
                canonical_json(request.model_dump(mode="json"))
            ),
            workflow_run=request.workflow_run,
            state_id=request.state_id,
            robot_type=request.robot_type,
            task=request.task,
            input_path=request.input_path,
            output_path=request.output_path,
            derived_project_id="npa-late-submit",
            remote_kind="suite",
            selection=request.suite,
        )
    )
    manager.states.update(record, status="cancelled")
    manager._cli = lambda *a, **k: pytest.fail("late submit invoked vendor CLI")  # type: ignore[method-assign]

    result = manager.submit(request)

    assert result.status == "cancelled"
    assert result.remote_id == ""


def test_terminal_resume_rejects_in_place_rerun_without_vendor_call() -> None:
    memory = MemoryStorage()
    manager = AntiochManager.__new__(AntiochManager)
    manager.storage = memory
    manager.states = StateStore(memory)
    request = _submit()
    record = manager.states.claim(
        OperationRecord(
            idempotency_key=operation_key(request.workflow_run, request.state_id),
            request_sha256="a" * 64,
            workflow_run=request.workflow_run,
            state_id=request.state_id,
            robot_type=request.robot_type,
            task=request.task,
            input_path=request.input_path,
            output_path=request.output_path,
            derived_project_id="npa-terminal-resume",
            remote_kind="suite",
            selection=request.suite,
            remote_id="remote-safe",
            status="failed",
        )
    )
    manager._cli = lambda *a, **k: pytest.fail("terminal resume invoked vendor CLI")  # type: ignore[method-assign]
    resume = ResumeRequest(
        output_path=record.output_path,
        workflow_run=record.workflow_run,
        state_id=record.state_id,
        rerun_terminal=True,
    )

    with pytest.raises(AntiochOperationError) as raised:
        manager.resume(resume)

    assert raised.value.error_type == "invalid_transition"
    assert manager._record_for(resume).status == "failed"


def test_collect_marks_state_before_conversion_and_blocks_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = MemoryStorage()
    manager = AntiochManager.__new__(AntiochManager)
    manager.storage = memory
    manager.states = StateStore(memory)
    request = _submit()
    completed = manager.states.claim(
        OperationRecord(
            idempotency_key=operation_key(request.workflow_run, request.state_id),
            request_sha256="a" * 64,
            workflow_run=request.workflow_run,
            state_id=request.state_id,
            robot_type=request.robot_type,
            task=request.task,
            input_path=request.input_path,
            output_path=request.output_path,
            derived_project_id="npa-manager-collection",
            remote_kind="suite",
            selection=request.suite,
            remote_id="remote-safe",
            status="completed",
        )
    )
    collect = ResumeRequest(
        output_path=completed.output_path,
        workflow_run=completed.workflow_run,
        state_id=completed.state_id,
    )

    def observe_marker(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        del args, kwargs
        assert manager._record_for(collect).status == "collecting"
        raise RuntimeError("stop after observing marker")

    monkeypatch.setattr("npa.workbench.antioch.manager.stage_project", observe_marker)
    with pytest.raises(RuntimeError, match="observing marker"):
        manager.collect(collect)
    recovered = manager._record_for(collect)
    assert recovered.status == "completed"
    assert recovered.collection_owner == ""
    assert recovered.retryable is True
    with pytest.raises(RuntimeError, match="observing marker"):
        manager.collect(collect)


def test_collect_preserves_terminal_failure_and_rejects_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = MemoryStorage()
    manager = AntiochManager.__new__(AntiochManager)
    manager.storage = memory
    manager.states = StateStore(memory)
    submitted = _submit()
    record = manager.states.claim(
        OperationRecord(
            idempotency_key=operation_key(submitted.workflow_run, submitted.state_id),
            request_sha256="a" * 64,
            workflow_run=submitted.workflow_run,
            state_id=submitted.state_id,
            robot_type=submitted.robot_type,
            task=submitted.task,
            input_path=submitted.input_path,
            output_path=submitted.output_path,
            derived_project_id="npa-terminal-collection",
            remote_kind="suite",
            selection=submitted.suite,
            remote_id="suite-safe",
            status="completed",
        )
    )
    request = CollectRequest(
        output_path=record.output_path,
        workflow_run=record.workflow_run,
        state_id=record.state_id,
    )

    def terminal_download(*_args, **_kwargs):  # noqa: ANN202
        raise AntiochOperationError(
            "source artifact is invalid",
            retryable=False,
            error_type="checksum_mismatch",
        )

    monkeypatch.setattr(
        "npa.workbench.antioch.manager.stage_project", terminal_download
    )
    with pytest.raises(AntiochOperationError) as first:
        manager.collect(request)
    assert first.value.error_type == "checksum_mismatch"
    durable = manager._record_for(request)
    assert durable.status == "completed"
    assert durable.collection_owner == ""
    assert durable.collection_phase == "download"
    assert durable.retryable is False

    monkeypatch.setattr(
        "npa.workbench.antioch.manager.stage_project",
        lambda *_args, **_kwargs: pytest.fail("terminal collection was retried"),
    )
    with pytest.raises(AntiochOperationError) as repeated:
        manager.collect(request)
    assert repeated.value.error_type == "checksum_mismatch"
    assert repeated.value.retryable is False


@pytest.mark.parametrize(
    "boundary",
    ["checksum", "conversion", "upload", "manifest", "state_persistence"],
)
def test_collect_real_body_recovers_every_persistence_boundary(
    boundary: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    memory = MemoryStorage()
    manager = AntiochManager.__new__(AntiochManager)
    manager.storage = memory
    manager.states = StateStore(memory)
    submitted = _submit()
    record = manager.states.claim(
        OperationRecord(
            idempotency_key=operation_key(submitted.workflow_run, submitted.state_id),
            request_sha256="a" * 64,
            workflow_run=submitted.workflow_run,
            state_id=submitted.state_id,
            robot_type=submitted.robot_type,
            task=submitted.task,
            input_path=submitted.input_path,
            output_path=submitted.output_path,
            derived_project_id="npa-collect-boundaries",
            remote_kind="suite",
            selection=submitted.suite,
            remote_id="suite-safe",
            status="completed",
        )
    )
    request = CollectRequest(
        output_path=record.output_path,
        workflow_run=record.workflow_run,
        state_id=record.state_id,
    )
    manifest = ProjectManifest(
        archive=ProjectArchive(name="project.tar.gz", size_bytes=1, sha256="b" * 64),
        source_name="fixture",
        source_revision="v1",
        source_license="Apache-2.0",
        source_sha256="c" * 64,
    )
    monkeypatch.setattr(
        "npa.workbench.antioch.manager.stage_project",
        lambda *_args, **_kwargs: (tmp_path, manifest, "c" * 64),
    )

    class CollectCli:
        def show(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return {
                "suite_run_id": "suite-safe",
                "phase": "completed",
                "scenario_runs": [{"scenario_run_id": "scenario-safe"}],
            }

        def download(self, *_args, output: Path, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            episode = output / "episode.npz"
            episode.write_bytes(b"episode")
            return {"files": [{"path": "episode.npz"}]}

        def logs(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return {"events": []}

    manager._cli = lambda *_args, **_kwargs: CollectCli()  # type: ignore[method-assign]

    def convert(_trajectories, output: Path, **_kwargs):  # noqa: ANN001, ANN202
        output.mkdir()
        (output / "data.json").write_text("{}", encoding="utf-8")
        return {"status": "converted"}

    monkeypatch.setattr("npa.workbench.antioch.manager.convert_episodes", convert)

    def upload(
        path: Path, uri: str, *, name: str, scenario_run_id: str = ""
    ) -> ArtifactRecord:
        return ArtifactRecord(
            name=name,
            uri=uri,
            size_bytes=path.stat().st_size,
            sha256=sha256_bytes(path.read_bytes()),
            scenario_run_id=scenario_run_id,
        )

    monkeypatch.setattr(manager.states, "upload_artifact", upload)
    original_put = manager.states.put_immutable_json
    original_update = manager.states.update
    failed = False

    def fail_once(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError(f"injected {boundary} failure")
        return args[0] if args else None

    if boundary == "checksum":
        original = _validate_downloaded_artifact

        def checksum(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            if not failed:
                return fail_once(*args, **kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr("npa.workbench.antioch.manager._validate_downloaded_artifact", checksum)
    elif boundary == "conversion":
        original = convert

        def conversion(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            if not failed:
                return fail_once(*args, **kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr("npa.workbench.antioch.manager.convert_episodes", conversion)
    elif boundary == "upload":
        def upload_boundary(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            if not failed:
                return fail_once(*args, **kwargs)
            return upload(*args, **kwargs)

        monkeypatch.setattr(manager.states, "upload_artifact", upload_boundary)
    elif boundary == "manifest":

        def put(uri, payload):  # noqa: ANN001, ANN202
            if not failed:
                return fail_once(uri, payload)
            return original_put(uri, payload)

        monkeypatch.setattr(manager.states, "put_immutable_json", put)
    else:

        def update(current, **changes):  # noqa: ANN001, ANN202
            if changes.get("completion_uri") and not failed:
                return fail_once(current, **changes)
            return original_update(current, **changes)

        monkeypatch.setattr(manager.states, "update", update)

    with pytest.raises(RuntimeError, match=f"injected {boundary}"):
        manager.collect(request)
    after_failure = manager._record_for(request)
    assert after_failure.status == "completed"
    assert after_failure.collection_owner == ""
    assert after_failure.retryable is True
    completed = manager.collect(request)
    assert completed.completion_uri.endswith("/_SUCCESS.json")
    assert manager.collect(request) == completed


@pytest.mark.parametrize(
    "manifest,expected",
    [
        ({"size_bytes": 0}, 0),
        ({"size": 7}, 7),
        ({"size_bytes": 0, "size": 7}, 0),
    ],
)
def test_manifest_artifact_size_zero_and_legacy_precedence(
    manifest: dict[str, int], expected: int
) -> None:
    assert _manifest_artifact_size(manifest) == expected


def test_manifest_zero_size_validation_detects_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"not-empty")
    with pytest.raises(AntiochOperationError, match="size verification"):
        _validate_downloaded_artifact(
            artifact, {"size_bytes": 0, "size": len(b"not-empty")}
        )


def test_manifest_zero_size_validation_accepts_empty_file(tmp_path: Path) -> None:
    artifact = tmp_path / "empty.bin"
    artifact.write_bytes(b"")
    _validate_downloaded_artifact(
        artifact,
        {"size_bytes": 0, "size": 7, "sha256": sha256_bytes(b"")},
    )
    with pytest.raises(AntiochOperationError, match="checksum verification"):
        _validate_downloaded_artifact(artifact, {"size_bytes": 0, "sha256": "f" * 64})


def test_run_poll_loop_does_not_read_state_before_resubmitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = AntiochManager.__new__(AntiochManager)
    request = _submit()
    claimed_record = OperationRecord(
        idempotency_key="a" * 64,
        request_sha256="b" * 64,
        workflow_run=request.workflow_run,
        state_id=request.state_id,
        robot_type=request.robot_type,
        task=request.task,
        input_path=request.input_path,
        output_path=request.output_path,
        derived_project_id="npa-poll-test",
        remote_kind="suite",
        selection=request.suite,
        status="claimed",
    )
    completed_record = OperationRecord(
        idempotency_key="a" * 64,
        request_sha256="b" * 64,
        workflow_run=request.workflow_run,
        state_id=request.state_id,
        robot_type=request.robot_type,
        task=request.task,
        input_path=request.input_path,
        output_path=request.output_path,
        derived_project_id="npa-poll-test",
        remote_kind="suite",
        selection=request.suite,
        remote_id="suite-safe",
        status="completed",
    )
    records = [claimed_record, completed_record]
    manager.submit = lambda _request: records.pop(0)  # type: ignore[method-assign]
    manager.collect = lambda _request: completed_record  # type: ignore[method-assign]
    manager.states = type(
        "NoReadState",
        (),
        {"read": lambda *a, **k: pytest.fail("poll loop performed a wasted S3 read")},
    )()
    monkeypatch.setattr("npa.workbench.antioch.manager.time.sleep", lambda _delay: None)
    completed = manager.run(request, poll_seconds=0)
    assert completed.remote_id == "suite-safe"


def test_artifact_verification_accepts_s3_metadata_header_casing(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "result.bin"
    artifact.write_bytes(b"verified")

    class S3:
        def upload_file(
            self, path: str, bucket: str, key: str, ExtraArgs: dict[str, Any]
        ) -> None:  # noqa: N803
            self.path = Path(path)
            self.metadata = ExtraArgs["Metadata"]

        def head_object(self, **kwargs):  # noqa: ANN003, ANN201
            return {
                "ContentLength": self.path.stat().st_size,
                "Metadata": {
                    "Sha256": self.metadata["sha256"],
                    "Npa-Role": "antioch-artifact",
                },
            }

    storage = type("Storage", (), {"s3": S3()})()
    record = StateStore(storage).upload_artifact(
        artifact, "s3://safe/result.bin", name="result.bin"
    )
    assert record.sha256 == sha256_bytes(b"verified")


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


def test_service_preserves_submit_dataset_metadata() -> None:
    captured: dict[str, SubmitRequest] = {}

    class Manager:
        def run(self, body: SubmitRequest) -> OperationRecord:
            captured["body"] = body
            return OperationRecord(
                idempotency_key="a" * 64,
                request_sha256="b" * 64,
                workflow_run=body.workflow_run,
                state_id=body.state_id,
                robot_type=body.robot_type,
                task=body.task,
                input_path=body.input_path,
                output_path=body.output_path,
                derived_project_id="npa-service-test",
                remote_kind="suite",
                selection=body.suite,
                status="completed",
            )

    client = TestClient(create_app(manager=Manager(), auth_mode="none"))
    body = _submit().model_dump(mode="json")
    body["robot_type"] = "inspection-arm"
    body["task"] = "Inspect the valve seal"
    response = client.post("/run", json=body)
    assert response.status_code == 200
    assert captured["body"].robot_type == "inspection-arm"
    assert captured["body"].task == "Inspect the valve seal"


@pytest.mark.parametrize("endpoint", ["submit", "reconcile", "cancel", "resume"])
@pytest.mark.parametrize(
    "failure,status,envelope",
    [
        (
            AntiochOperationError(
                "operation conflict", error_type="invalid_transition"
            ),
            409,
            {
                "type": "invalid_transition",
                "message": "operation conflict",
                "retryable": False,
            },
        ),
        (
            AntiochRuntimeError("scoped terms are missing"),
            503,
            {
                "type": "runtime_error",
                "message": "scoped terms are missing",
                "retryable": False,
            },
        ),
        (
            AntiochRuntimeError("runtime cache is cold in offline mode"),
            503,
            {
                "type": "runtime_error",
                "message": "runtime cache is cold in offline mode",
                "retryable": False,
            },
        ),
        (
            AntiochRuntimeError("Antioch CLI version mismatch"),
            503,
            {
                "type": "runtime_error",
                "message": "Antioch CLI version mismatch",
                "retryable": False,
            },
        ),
        (
            AntiochCliError("vendor capacity", error_type="capacity", retryable=True),
            503,
            {
                "type": "capacity",
                "message": "vendor capacity",
                "retryable": True,
            },
        ),
        (
            AntiochCliError("bad output", error_type="malformed_cli_output"),
            502,
            {
                "type": "malformed_cli_output",
                "message": "bad output",
                "retryable": False,
            },
        ),
    ],
)
def test_service_endpoints_return_typed_error_envelopes(
    endpoint: str,
    failure: Exception,
    status: int,
    envelope: dict[str, object],
) -> None:
    class Manager:
        def __getattr__(self, name: str):  # noqa: ANN204
            del name

            def fail(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
                del args, kwargs
                raise failure

            return fail

    body = (
        _submit().model_dump(mode="json")
        if endpoint == "submit"
        else {
            "output_path": "s3://safe/run",
            "workflow_run": "run-1",
            "state_id": "simulate",
        }
    )
    response = TestClient(create_app(manager=Manager(), auth_mode="none")).post(
        f"/{endpoint}", json=body
    )
    assert response.status_code == status
    assert response.json() == {"detail": envelope}


def test_health_reports_runtime_failure_as_degraded_typed_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail() -> None:
        raise AntiochRuntimeError("runtime cache is unavailable")

    monkeypatch.setattr("npa.workbench.antioch.service.ensure_runtime", fail)
    response = TestClient(create_app(manager=object(), auth_mode="none")).get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "cli_installed": False,
        "authenticated": False,
        "cli_version": "",
        "environment": "",
        "detail": "runtime cache is unavailable",
    }
