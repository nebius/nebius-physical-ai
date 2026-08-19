"""Idempotent Antioch operations composed over the structured vendor CLI."""

from __future__ import annotations

import hashlib
import json
import os
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
from .runtime import ensure_runtime
from .schemas import (
    ARTIFACT_MANIFEST_SCHEMA,
    COMPLETION_SCHEMA,
    CollectRequest,
    OperationRecord,
    ResumeRequest,
    SubmitRequest,
    utc_now,
)
from .storage import StateStore, canonical_json, join_uri, sha256_bytes, sha256_file
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


class AntiochManager:
    def __init__(self, storage: StorageClient | None = None) -> None:
        if storage is None:
            endpoint = os.environ.get("AWS_ENDPOINT_URL", "") or os.environ.get(
                "NEBIUS_S3_ENDPOINT", ""
            )
            if endpoint:
                storage = StorageClient.from_environment(endpoint_url=endpoint)
            else:
                from npa.clients.config import resolve_project_storage

                configured = resolve_project_storage()
                storage = StorageClient.from_environment(
                    endpoint_url=configured.endpoint_url,
                    aws_access_key_id=configured.aws_access_key_id,
                    aws_secret_access_key=configured.aws_secret_access_key,
                )
        self.storage = storage
        self.states = StateStore(self.storage)

    def _cli(self, expected_version: str = "0.3.47") -> AntiochCli:
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
        key = operation_key(request.workflow_run, request.state_id)
        kind = "suite" if request.suite else "scenario"
        record = self.states.claim(
            OperationRecord(
                idempotency_key=key,
                request_sha256=_request_digest(request),
                workflow_run=request.workflow_run,
                state_id=request.state_id,
                input_path=request.input_path,
                output_path=request.output_path,
                derived_project_id=deterministic_project_id(
                    request.workflow_run, request.state_id
                ),
                remote_kind=kind,
                selection=request.suite or request.scenario,
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
        if not record.remote_id:
            replay = SubmitRequest(
                input_path=record.input_path,
                output_path=record.output_path,
                workflow_run=record.workflow_run,
                state_id=record.state_id,
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
        if record.status == "cancelled":
            return record
        if not record.remote_id:
            return self.states.update(record, status="cancelled")
        with tempfile.TemporaryDirectory(prefix="npa-antioch-cancel-") as temp_name:
            project, _manifest, _digest = stage_project(
                self.storage,
                record.input_path,
                Path(temp_name),
                project_id=record.derived_project_id,
            )
            self._cli().cancel(
                project, kind=record.remote_kind, remote_id=record.remote_id
            )
        return self.states.update(record, status="cancelled", remote_phase="cancelled")

    def resume(self, request: ResumeRequest) -> OperationRecord:
        record = self.reconcile(request)
        if not request.rerun_terminal or record.status not in {"failed", "cancelled"}:
            return record
        with tempfile.TemporaryDirectory(prefix="npa-antioch-rerun-") as temp_name:
            project, _manifest, _digest = stage_project(
                self.storage,
                record.input_path,
                Path(temp_name),
                project_id=record.derived_project_id,
            )
            payload = self._cli().rerun(
                project, kind=record.remote_kind, remote_id=record.remote_id
            )
        return self.states.update(
            record,
            remote_id=remote_id(payload, kind=record.remote_kind),
            invocation_id=invocation_id(payload),
            status="submitted",
            remote_phase="",
            remote_outcome="",
        )

    def collect(self, request: CollectRequest) -> OperationRecord:
        record = self.reconcile(request)
        if record.status != "completed":
            raise AntiochOperationError(
                f"remote operation is not collectable: {record.status}"
            )
        if record.completion_uri:
            return record
        record = self.states.update(record, status="collecting")
        with tempfile.TemporaryDirectory(prefix="npa-antioch-collect-") as temp_name:
            temp = Path(temp_name)
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
                    expected_size = item.get("size_bytes") or item.get("size")
                    expected_sha = str(item.get("sha256") or "")
                    if (
                        expected_size is not None
                        and int(expected_size) != path.stat().st_size
                    ):
                        raise AntiochOperationError(
                            "downloaded artifact failed size verification"
                        )
                    if expected_sha and expected_sha != sha256_file(path):
                        raise AntiochOperationError(
                            "downloaded artifact failed checksum verification"
                        )
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
                dataset = temp / "lerobot-dataset"
                provenance = convert_episodes(
                    trajectories,
                    dataset,
                    robot_type=request.robot_type,
                    task=request.task,
                    source_sha256=source_manifest.source_sha256,
                    asset_hashes=source_manifest.asset_hashes,
                )
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
            manifest = {
                "schema_name": ARTIFACT_MANIFEST_SCHEMA,
                "created_at": utc_now(),
                "operation": record.model_dump(mode="json"),
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
                    "created_at": utc_now(),
                    "manifest_uri": manifest_uri,
                    "manifest_sha256": sha256_bytes(canonical_json(manifest)),
                    "dataset_uri": dataset_uri,
                    "remote_id": record.remote_id,
                },
            )
        return self.states.update(
            record,
            status="completed",
            artifact_manifest_uri=manifest_uri,
            dataset_uri=dataset_uri,
            completion_uri=completion_uri,
        )

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
            current = self.states.read(record.output_path, record.idempotency_key)
            if current is None:
                raise AntiochOperationError(
                    "durable Antioch state disappeared during submission"
                )
            record = current[0]
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
