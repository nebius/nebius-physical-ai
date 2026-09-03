"""Idempotent Antioch operations composed over the structured vendor CLI."""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from npa.clients.storage import StorageClient

from .dataset import convert_episodes
from .project import deterministic_project_id, stage_project
from .redaction import redact_text
from .runtime import ensure_runtime, terms_preflight
from .schemas import (
    ARTIFACT_MANIFEST_SCHEMA,
    COMPLETION_SCHEMA,
    CollectRequest,
    OperationRecord,
    ResumeRequest,
    SubmitRequest,
)
from .storage import StateStore, canonical_json, join_uri, sha256_bytes, sha256_file
from .storage_config import resolve_storage_client
from .vendor_cli import (
    AntiochCli,
    AntiochCliError,
    invocation_id,
    public_snapshot,
    remote_id,
)


class AntiochOperationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        error_type: str = "operation_error",
    ) -> None:
        super().__init__(redact_text(message))
        self.retryable = retryable
        self.error_type = error_type


def operation_key(workflow_run: str, state_id: str) -> str:
    return hashlib.sha256(f"{workflow_run}\n{state_id}".encode()).hexdigest()


def _request_digest(request: SubmitRequest) -> str:
    return sha256_bytes(canonical_json(request.model_dump(mode="json")))


def _phase(payload: dict[str, Any]) -> tuple[str, str]:
    phase = str(payload.get("phase") or payload.get("status") or "").lower()
    outcome = str(payload.get("outcome") or payload.get("result") or "").lower()
    return phase, outcome


def _local_status(phase: str, outcome: str) -> str:
    if phase in {"cancelled", "canceled"} or outcome in {"cancelled", "canceled"}:
        return "cancelled"
    if phase in {"failed", "error"} or outcome in {"failed", "error", "failure"}:
        return "failed"
    if phase in {"complete", "completed", "finished", "succeeded", "passed"}:
        return "failed" if outcome in {"failed", "failure", "error"} else "completed"
    if outcome in {"passed", "success", "succeeded"}:
        return "completed"
    if phase in {"running", "executing"}:
        return "running"
    return "queued"


def _scenario_ids(kind: str, remote: dict[str, Any], own_id: str) -> list[str]:
    if kind == "scenario":
        return [own_id]
    runs = remote.get("scenario_runs") or remote.get("runs") or []
    ids: list[str] = []
    if isinstance(runs, list):
        for item in runs:
            if isinstance(item, dict):
                try:
                    ids.append(remote_id(item, kind="scenario"))
                except AntiochCliError:
                    continue
    if not ids:
        raise AntiochOperationError(
            "completed suite exposed no scenario run identifiers",
            error_type="malformed_cli_output",
        )
    return sorted(set(ids))


def _dataset_metadata(record: OperationRecord) -> tuple[str, str]:
    robot_type = record.robot_type.strip()
    task = record.task.strip()
    if not robot_type or not task:
        raise AntiochOperationError(
            "durable operation is missing required robot_type/task dataset metadata",
            error_type="dataset_metadata_missing",
        )
    return robot_type, task


def _manifest_artifact_size(item: dict[str, Any]) -> Any:
    """Prefer the current field even when its explicit value is zero."""

    if "size_bytes" in item:
        return item["size_bytes"]
    return item.get("size")


def _validate_downloaded_artifact(path: Path, item: dict[str, Any]) -> None:
    expected_size = _manifest_artifact_size(item)
    if expected_size is not None and int(expected_size) != path.stat().st_size:
        raise AntiochOperationError("downloaded artifact failed size verification")
    expected_sha = str(item.get("sha256") or "")
    if expected_sha and expected_sha != sha256_file(path):
        raise AntiochOperationError("downloaded artifact failed checksum verification")


@contextmanager
def _submission_heartbeat(states: StateStore, record: OperationRecord, owner: str):
    """Renew the S3 fencing lease while a vendor queue command is staging."""

    stop = threading.Event()
    errors: list[Exception] = []

    def renew() -> None:
        while not stop.wait(20):
            try:
                states.refresh_submission(record, owner)
            except Exception as exc:  # state fencing must fail closed
                errors.append(exc)
                stop.set()

    worker = threading.Thread(
        target=renew, name="antioch-submission-lease", daemon=True
    )
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()
    if errors:
        raise AntiochOperationError(
            "durable submission lease renewal failed",
            retryable=True,
            error_type="lease_lost",
        ) from errors[0]


@contextmanager
def _collection_heartbeat(states: StateStore, record: OperationRecord, owner: str):
    stop = threading.Event()
    errors: list[Exception] = []

    def renew() -> None:
        while not stop.wait(20):
            try:
                states.refresh_collection(record, owner)
            except Exception as exc:
                errors.append(exc)
                stop.set()

    worker = threading.Thread(target=renew, name="antioch-collection-lease", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()
    if errors:
        raise AntiochOperationError(
            "durable collection lease renewal failed",
            retryable=True,
            error_type="collection_lease_lost",
        ) from errors[0]


class AntiochManager:
    def __init__(self, storage: StorageClient | None = None) -> None:
        if storage is None:
            storage = resolve_storage_client()
        self.storage = storage
        self.states = StateStore(self.storage)

    def _cli(self, expected_version: str = "0.3.63") -> AntiochCli:
        return AntiochCli(ensure_runtime(expected_version=expected_version))

    def _record_for(self, request: ResumeRequest) -> OperationRecord:
        key = operation_key(request.workflow_run, request.state_id)
        current = self.states.read(request.output_path, key)
        if current is None:
            raise AntiochOperationError(
                "no durable Antioch state exists for this workflow state",
                error_type="not_found",
            )
        return current[0]

    def submit(self, request: SubmitRequest) -> OperationRecord:
        acceptance = terms_preflight()
        key = operation_key(request.workflow_run, request.state_id)
        kind = "suite" if request.suite else "scenario"
        record = self.states.claim(
            OperationRecord(
                idempotency_key=key,
                request_sha256=_request_digest(request),
                workflow_run=request.workflow_run,
                state_id=request.state_id,
                robot_type=request.robot_type,
                task=request.task,
                input_path=request.input_path,
                output_path=request.output_path,
                derived_project_id=deterministic_project_id(
                    request.workflow_run, request.state_id
                ),
                remote_kind=kind,
                selection=request.suite or request.scenario,
                terms_name=str(acceptance["name"]),
                terms_url=str(acceptance["url"]),
                terms_version=str(acceptance["version"]),
                terms_scope=str(acceptance["scope"]),
                terms_accepted=bool(acceptance["accepted"]),
            )
        )
        if record.remote_id:
            return record
        owner = str(uuid.uuid4())
        record, acquired = self.states.acquire_submission(record, owner)
        if not acquired:
            return record
        cli = self._cli(request.expected_cli_version)
        try:
            with _submission_heartbeat(self.states, record, owner):
                with tempfile.TemporaryDirectory(
                    prefix="npa-antioch-submit-"
                ) as temp_name:
                    project, _manifest, digest = stage_project(
                        self.storage,
                        request.input_path,
                        Path(temp_name),
                        project_id=record.derived_project_id,
                    )
                    # The deterministic project id closes the crash window between a successful
                    # vendor submission and publishing the remote id to durable S3 state.
                    existing = cli.list_for_project(
                        project, kind=kind, project_id=record.derived_project_id
                    )
                    if len(existing) > 1:
                        raise AntiochOperationError(
                            "multiple remote runs exist for the deterministic project identity; refusing ambiguity",
                            error_type="reconciliation_conflict",
                        )
                    payload = (
                        existing[0]
                        if existing
                        else (
                            cli.submit_suite(project, request.suite)
                            if kind == "suite"
                            else cli.submit_scenario(
                                project,
                                request.scenario,
                                scenario_case=request.scenario_case,
                                parameters=request.parameters,
                            )
                        )
                    )
                return self.states.update(
                    record,
                    input_sha256=digest,
                    remote_id=remote_id(payload, kind=kind),
                    invocation_id=invocation_id(payload),
                    status="submitted",
                    retryable=False,
                    error_type="",
                    error_message="",
                    submission_owner="",
                    submission_lease_expires_at="",
                )
        except AntiochCliError as exc:
            self.states.update(
                record,
                retryable=exc.retryable,
                error_type=exc.error_type,
                error_message=str(exc),
                submission_owner="",
                submission_lease_expires_at="",
            )
            raise AntiochOperationError(
                str(exc), retryable=exc.retryable, error_type=exc.error_type
            ) from exc

    def reconcile(self, request: ResumeRequest) -> OperationRecord:
        record = self._record_for(request)
        if record.status in {"completed", "collecting", "failed", "cancelled"}:
            return record
        if not record.remote_id:
            replay = SubmitRequest(
                input_path=record.input_path,
                output_path=record.output_path,
                workflow_run=record.workflow_run,
                state_id=record.state_id,
                robot_type=record.robot_type,
                task=record.task,
                **(
                    {"suite": record.selection}
                    if record.remote_kind == "suite"
                    else {"scenario": record.selection}
                ),
            )
            return self.submit(replay)
        try:
            with tempfile.TemporaryDirectory(prefix="npa-antioch-status-") as temp_name:
                project, _manifest, _digest = stage_project(
                    self.storage,
                    record.input_path,
                    Path(temp_name),
                    project_id=record.derived_project_id,
                )
                payload = self._cli().show(
                    project, kind=record.remote_kind, remote_id=record.remote_id
                )
            phase, outcome = _phase(payload)
            return self.states.update(
                record,
                remote_phase=phase,
                remote_outcome=outcome,
                status=_local_status(phase, outcome),
                retryable=False,
                error_type="",
                error_message="",
            )
        except AntiochCliError as exc:
            self.states.update(
                record,
                retryable=exc.retryable,
                error_type=exc.error_type,
                error_message=str(exc),
            )
            raise AntiochOperationError(
                str(exc), retryable=exc.retryable, error_type=exc.error_type
            ) from exc

    def cancel(self, request: ResumeRequest) -> OperationRecord:
        record = self._record_for(request)
        if record.status in {"completed", "failed", "cancelled"}:
            return record
        if not record.remote_id:
            return self.states.update(record, status="cancelled")
        try:
            with tempfile.TemporaryDirectory(prefix="npa-antioch-cancel-") as temp_name:
                project, _manifest, _digest = stage_project(
                    self.storage,
                    record.input_path,
                    Path(temp_name),
                    project_id=record.derived_project_id,
                )
                cli = self._cli()
                current = cli.show(
                    project, kind=record.remote_kind, remote_id=record.remote_id
                )
                phase, outcome = _phase(current)
                status = _local_status(phase, outcome)
                record = self.states.update(
                    record,
                    remote_phase=phase,
                    remote_outcome=outcome,
                    status=status,
                    retryable=False,
                    error_type="",
                    error_message="",
                )
                if status in {"completed", "failed", "cancelled"}:
                    return record
                cancelled = cli.cancel(
                    project, kind=record.remote_kind, remote_id=record.remote_id
                )
                cancel_phase, cancel_outcome = _phase(cancelled)
        except AntiochCliError as exc:
            self.states.update(
                record,
                retryable=exc.retryable,
                error_type=exc.error_type,
                error_message=str(exc),
            )
            raise AntiochOperationError(
                str(exc), retryable=exc.retryable, error_type=exc.error_type
            ) from exc
        return self.states.update(
            record,
            status="cancelled",
            remote_phase=cancel_phase or "cancelled",
            remote_outcome=cancel_outcome,
            retryable=False,
            error_type="",
            error_message="",
        )

    def resume(self, request: ResumeRequest) -> OperationRecord:
        record = self.reconcile(request)
        if not request.rerun_terminal or record.status not in {"failed", "cancelled"}:
            return record
        raise AntiochOperationError(
            "terminal Antioch state cannot be rerun in place; use a new state_id",
            error_type="invalid_transition",
        )

    def collect(self, request: CollectRequest) -> OperationRecord:
        record = self._record_for(request)
        if record.completion_uri:
            return record
        if (
            record.status == "completed"
            and record.collection_phase
            and record.error_type
            and not record.retryable
        ):
            raise AntiochOperationError(
                "Antioch collection previously failed terminally",
                retryable=False,
                error_type=record.error_type,
            )
        if record.status not in {"completed", "collecting"}:
            record = self.reconcile(request)
        if record.status not in {"completed", "collecting"}:
            raise AntiochOperationError(
                f"remote operation is not collectable: {record.status}"
            )
        if record.completion_uri:
            return record
        owner = str(uuid.uuid4())
        record, acquired = self.states.acquire_collection(record, owner)
        if not acquired:
            if record.completion_uri:
                return record
            raise AntiochOperationError(
                "Antioch collection is already in progress",
                retryable=True,
                error_type="collection_in_progress",
            )
        try:
            heartbeat = _collection_heartbeat(self.states, record, owner)
            heartbeat.__enter__()
            temporary = tempfile.TemporaryDirectory(prefix="npa-antioch-collect-")
            temp = Path(temporary.__enter__())
            self.states.refresh_collection(record, owner, phase="download")
            project, source_manifest, _digest = stage_project(
                self.storage,
                record.input_path,
                temp,
                project_id=record.derived_project_id,
            )
            cli = self._cli()
            remote = cli.show(
                project, kind=record.remote_kind, remote_id=record.remote_id
            )
            artifacts = []
            trajectories: list[Path] = []
            for scenario_id in _scenario_ids(
                record.remote_kind, remote, record.remote_id
            ):
                self.states.refresh_collection(record, owner, phase="upload")
                downloaded = temp / "downloads" / scenario_id
                downloaded.mkdir(parents=True)
                transfer = cli.download(
                    project, scenario_run_id=scenario_id, output=downloaded
                )
                for item in transfer["files"]:
                    if not isinstance(item, dict):
                        raise AntiochOperationError(
                            "transfer manifest contained a malformed file record"
                        )
                    raw_path = str(
                        item.get("path")
                        or item.get("destination")
                        or item.get("name")
                        or ""
                    )
                    path = Path(raw_path)
                    if not path.is_absolute():
                        path = downloaded / path
                    try:
                        path.resolve().relative_to(downloaded.resolve())
                    except ValueError as exc:
                        raise AntiochOperationError(
                            "transfer manifest referenced a path outside its destination"
                        ) from exc
                    if not path.is_file():
                        raise AntiochOperationError(
                            "transfer manifest referenced a missing artifact"
                        )
                    _validate_downloaded_artifact(path, item)
                    relative_name = path.relative_to(downloaded).as_posix()
                    artifact = self.states.upload_artifact(
                        path,
                        join_uri(
                            record.output_path, "artifacts", scenario_id, relative_name
                        ),
                        name=relative_name,
                        scenario_run_id=scenario_id,
                    )
                    artifacts.append(artifact)
                    if path.suffix == ".npz":
                        trajectories.append(path)
                logs = temp / f"{scenario_id}.logs.json"
                logs.write_text(
                    json.dumps(
                        cli.logs(project, scenario_run_id=scenario_id), sort_keys=True
                    ),
                    encoding="utf-8",
                )
                artifacts.append(
                    self.states.upload_artifact(
                        logs,
                        join_uri(
                            record.output_path, "artifacts", scenario_id, "logs.json"
                        ),
                        name="logs.json",
                        scenario_run_id=scenario_id,
                    )
                )
            dataset_uri = ""
            dataset_files = []
            if request.require_policy_dataset:
                self.states.refresh_collection(record, owner, phase="convert")
                robot_type, task = _dataset_metadata(record)
                dataset = temp / "lerobot-dataset"
                provenance = convert_episodes(
                    trajectories,
                    dataset,
                    robot_type=robot_type,
                    task=task,
                    source_sha256=source_manifest.source_sha256,
                    asset_hashes=source_manifest.asset_hashes,
                )
                self.states.refresh_collection(record, owner, phase="upload")
                for path in sorted(
                    item for item in dataset.rglob("*") if item.is_file()
                ):
                    relative = path.relative_to(dataset).as_posix()
                    dataset_files.append(
                        self.states.upload_artifact(
                            path,
                            join_uri(record.output_path, "dataset", relative),
                            name=relative,
                        ).model_dump(mode="json")
                    )
                dataset_uri = join_uri(record.output_path, "dataset")
            else:
                provenance = None
            manifest_uri = join_uri(record.output_path, "manifests", "v1.json")
            self.states.refresh_collection(record, owner, phase="manifest")
            manifest_record = record.model_copy(
                update={
                    "status": "completed",
                    "collection_owner": "",
                    "collection_lease_expires_at": "",
                    "collection_phase": "",
                    "retryable": False,
                    "error_type": "",
                    "error_message": "",
                    "updated_at": record.created_at,
                    "revision": 1,
                }
            )
            manifest = {
                "schema_name": ARTIFACT_MANIFEST_SCHEMA,
                "created_at": record.created_at,
                "operation": manifest_record.model_dump(mode="json"),
                "remote": public_snapshot(remote),
                "source": source_manifest.model_dump(mode="json"),
                "artifacts": [item.model_dump(mode="json") for item in artifacts],
                "dataset_uri": dataset_uri,
                "dataset_files": dataset_files,
                "dataset_provenance": provenance,
            }
            self.states.put_immutable_json(manifest_uri, manifest)
            completion_uri = join_uri(record.output_path, "_SUCCESS.json")
            # Publishing this immutable marker is deliberately the final data-bus write.
            self.states.put_immutable_json(
                completion_uri,
                {
                    "schema_name": COMPLETION_SCHEMA,
                    "created_at": record.created_at,
                    "manifest_uri": manifest_uri,
                    "manifest_sha256": sha256_bytes(canonical_json(manifest)),
                    "dataset_uri": dataset_uri,
                    "remote_id": record.remote_id,
                },
            )
            self.states.refresh_collection(record, owner, phase="state")
            result = self.states.update(
                record,
                status="completed",
                artifact_manifest_uri=manifest_uri,
                dataset_uri=dataset_uri,
                completion_uri=completion_uri,
                collection_owner="",
                collection_lease_expires_at="",
                collection_phase="",
                retryable=False,
                error_type="",
                error_message="",
            )
            heartbeat.__exit__(None, None, None)
            temporary.__exit__(None, None, None)
            return result
        except Exception as exc:
            self.states.fail_collection(
                record,
                owner,
                error_type=getattr(exc, "error_type", type(exc).__name__),
                retryable=(
                    exc.retryable if isinstance(exc, AntiochOperationError) else True
                ),
            )
            if "temporary" in locals():
                temporary.__exit__(type(exc), exc, exc.__traceback__)
            if "heartbeat" in locals():
                heartbeat.__exit__(type(exc), exc, exc.__traceback__)
            if isinstance(exc, AntiochOperationError):
                raise
            raise

    def run(
        self, request: SubmitRequest, *, poll_seconds: float = 10.0
    ) -> OperationRecord:
        record = self.submit(request)
        resume = ResumeRequest(
            output_path=request.output_path,
            workflow_run=request.workflow_run,
            state_id=request.state_id,
        )
        while not record.remote_id:
            time.sleep(poll_seconds)
            # ``submit`` is a compare-and-swap operation. Calling it on every
            # pass lets this runner take over an expired fencing lease after a
            # submitting process dies; an active owner remains untouched.
            record = self.submit(request)
        while record.status not in {"completed", "failed", "cancelled"}:
            time.sleep(poll_seconds)
            record = self.reconcile(resume)
        if record.status != "completed":
            raise AntiochOperationError(f"Antioch operation ended in {record.status}")
        return self.collect(CollectRequest(**resume.model_dump()))
