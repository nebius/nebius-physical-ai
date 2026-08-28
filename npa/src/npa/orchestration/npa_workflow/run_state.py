"""Durable run manifest for ``npa.workflow`` executions on object storage."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Any, Mapping, Sequence

RUN_SCHEMA_VERSION = "npa.workflow.run.v1"
RUNTIME_SCHEMA_VERSION = "npa.workflow.runtime.v1"
PAIDF_WORKFLOW_NAME = "physical-ai-data-factory"
PAIDF_COSMOS3_WORKFLOW_NAME = "paidf-cosmos3"
PAIDF_INPUT_WORKFLOW_NAMES = frozenset(
    {PAIDF_WORKFLOW_NAME, PAIDF_COSMOS3_WORKFLOW_NAME}
)


def is_paidf_input_workflow_name(name: object) -> bool:
    """Whether submit owns real-video/LeRobot preparation for this workflow."""

    return str(name or "").strip() in PAIDF_INPUT_WORKFLOW_NAMES


def paidf_artifact_prefix(run_id: str) -> str:
    """Canonical PAIDF run prefix shared by submit/status/artifact consumers."""

    return f"{PAIDF_WORKFLOW_NAME}/{str(run_id or '').strip()}".strip("/")


def paidf_workflow_prefix(run_id: str) -> str:
    """Canonical durable ``npa.workflow`` ledger prefix for one PAIDF run."""

    return f"{paidf_artifact_prefix(run_id)}/npa-workflow"


def is_paidf_run_id(run_id: str) -> bool:
    """Whether an NPA-reserved run ID belongs to the PAIDF happy path."""

    return str(run_id or "").strip().lower().startswith("paidf-")


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


@dataclass
class RunManifest:
    workflow: str
    run_id: str
    api_version: str
    run_prefix_uri: str = ""
    status: str = "planned"
    sky_job_id: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    input_source: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=utc_now)
    schema_version: str = RUN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workflow": self.workflow,
            "run_id": self.run_id,
            "api_version": self.api_version,
            "run_prefix_uri": self.run_prefix_uri,
            "status": self.status,
            "sky_job_id": self.sky_job_id,
            "updated_at": self.updated_at,
            "steps": list(self.steps),
            "input_source": dict(self.input_source),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunManifest:
        return cls(
            workflow=str(payload.get("workflow") or ""),
            run_id=str(payload.get("run_id") or ""),
            api_version=str(payload.get("api_version") or ""),
            run_prefix_uri=str(payload.get("run_prefix_uri") or ""),
            status=str(payload.get("status") or "planned"),
            sky_job_id=str(payload.get("sky_job_id") or ""),
            steps=[
                dict(item)
                for item in payload.get("steps") or []
                if isinstance(item, dict)
            ],
            input_source=dict(payload.get("input_source") or {}),
            updated_at=str(payload.get("updated_at") or utc_now()),
            schema_version=str(payload.get("schema_version") or RUN_SCHEMA_VERSION),
        )


@dataclass
class RuntimeRunState:
    """Durable ledger for a runtime-orchestrated run (waves, jobs, decisions).

    This is written *in addition to* :class:`RunManifest` (whose schema is
    unchanged) and is what makes the runtime tier resumable: a wave whose key is
    already recorded as ``succeeded`` is replayed from the ledger instead of being
    resubmitted, so re-running the same ``run_id`` is idempotent.
    """

    workflow: str
    run_id: str
    api_version: str = ""
    status: str = "running"
    run_prefix_uri: str = ""
    #: Fingerprint of the plan this ledger was recorded for. ``--resume`` replays waves
    #: by key, and keys only line up when the traversal is identical, so a resumed run
    #: whose spec/config changed must not silently reuse them.
    plan_fingerprint: str = ""
    waves: list[dict[str, Any]] = field(default_factory=list)
    # Additive, per-stage projection of the wave ledger.  This is deliberately
    # kept in runtime.json so status/logs/cancel all consume one state store.
    stages: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    watermarks: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=utc_now)
    schema_version: str = RUNTIME_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workflow": self.workflow,
            "run_id": self.run_id,
            "api_version": self.api_version,
            "status": self.status,
            "run_prefix_uri": self.run_prefix_uri,
            "plan_fingerprint": self.plan_fingerprint,
            "updated_at": self.updated_at,
            "waves": list(self.waves),
            "stages": list(self.stages),
            "decisions": list(self.decisions),
            "watermarks": dict(self.watermarks),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeRunState:
        return cls(
            workflow=str(payload.get("workflow") or ""),
            run_id=str(payload.get("run_id") or ""),
            api_version=str(payload.get("api_version") or ""),
            status=str(payload.get("status") or "running"),
            run_prefix_uri=str(payload.get("run_prefix_uri") or ""),
            plan_fingerprint=str(payload.get("plan_fingerprint") or ""),
            waves=[
                dict(item)
                for item in payload.get("waves") or []
                if isinstance(item, dict)
            ],
            stages=[
                dict(item)
                for item in payload.get("stages") or []
                if isinstance(item, dict)
            ],
            decisions=[
                dict(item)
                for item in payload.get("decisions") or []
                if isinstance(item, dict)
            ],
            watermarks=dict(payload.get("watermarks") or {}),
            updated_at=str(payload.get("updated_at") or utc_now()),
            schema_version=str(payload.get("schema_version") or RUNTIME_SCHEMA_VERSION),
        )

    def completed_wave(self, key: str) -> dict[str, Any] | None:
        """Return the recorded outcome of a wave that already succeeded."""

        for record in reversed(self.waves):
            if record.get("key") == key and record.get("status") == "succeeded":
                return record
        return None

    def in_flight_wave(self, key: str) -> dict[str, Any] | None:
        """Return a wave recorded as started but never finished.

        A ``running`` record means a driver submitted the wave and then stopped
        watching it (crash, kill, lost connection). The managed job may still be
        alive, so a resumed run must reconcile it instead of submitting a second
        copy of the same work.
        """

        for record in reversed(self.waves):
            if record.get("key") != key:
                continue
            status = str(record.get("status") or "")
            if status == "succeeded":
                return None
            recovery = str(record.get("recovery_decision") or "")
            unresolved = recovery in {
                "block_indeterminate",
                "block_after_uncertain_success",
                "recovery_deadline_exhausted_verified_absent",
                "interrupted_verified_absent",
                "resume_block_terminal_or_legacy_absence",
                "resume_block_output_present",
                "resume_block_output_indeterminate",
            }
            return dict(record) if status == "running" or unresolved else None
        return None

    def latest_wave(self, key: str) -> dict[str, Any] | None:
        for record in reversed(self.waves):
            if record.get("key") == key:
                return dict(record)
        return None

    def record_wave(self, record: Mapping[str, Any]) -> None:
        key = str(record.get("key") or "")
        attempt = int(record.get("attempt") or 1)
        for index, existing in enumerate(self.waves):
            if (
                existing.get("key") == key
                and int(existing.get("attempt") or 1) == attempt
            ):
                self.waves[index] = dict(record)
                self._record_stage_attempts(record)
                return
        self.waves.append(dict(record))
        self._record_stage_attempts(record)

    def _record_stage_attempts(self, record: Mapping[str, Any]) -> None:
        """Idempotently project one exact wave attempt into stage records."""

        members = _wave_members(record)
        tasks = [
            item for item in record.get("tasks") or [] if isinstance(item, Mapping)
        ]
        observations = [
            item
            for item in record.get("observations") or []
            if isinstance(item, Mapping)
        ]
        attempt = int(record.get("attempt") or 1)
        wave_status = str(record.get("status") or "unknown")
        terminal = wave_status.lower() in {"succeeded", "failed", "cancelled"}
        for state_name, iteration in members:
            key = f"{state_name}#{iteration}" if iteration is not None else state_name
            matching_tasks = [
                item for item in tasks if str(item.get("task_name") or "") == state_name
            ]
            task = matching_tasks[-1] if matching_tasks else {}
            progress_times = [
                str(item.get("last_progress_at") or "")
                for item in matching_tasks
                if str(item.get("last_progress_at") or "")
            ]
            observed_times = [
                str(item.get("observed_at") or "")
                for item in observations
                if str(item.get("observed_at") or "")
            ]
            scheduler_state = str(
                task.get("status") or record.get("sky_status") or "UNKNOWN"
            )
            stage_record = {
                "key": key,
                "stage": state_name,
                "iteration": iteration,
                "logical_state": _normalized_stage_state(
                    task.get("status") or record.get("sky_status") or wave_status
                ),
                "attempt": attempt,
                "managed_job_id": str(
                    record.get("job_id") or record.get("sky_job_id") or ""
                ),
                "job_group_id": str(
                    record.get("group") or record.get("job_name") or ""
                ),
                "job_name": str(record.get("job_name") or ""),
                "wave_key": str(record.get("key") or ""),
                "submitted_at": str(record.get("started_at") or ""),
                "started_at": str(task.get("start_at") or ""),
                "finished_at": str(task.get("end_at") or record.get("ended_at") or ""),
                "last_observed_at": max(observed_times, default=""),
                "last_scheduler_state": scheduler_state,
                # Heartbeats are real task progress only. Submission, polling,
                # manifest writes, and transitions must not fabricate liveness.
                "last_heartbeat_at": max(progress_times, default=""),
                "heartbeat_source": "scheduler_task_progress" if progress_times else "",
                "pending_reason": dict(record.get("pending_reason") or {}),
                "log_location": str(task.get("log_path") or task.get("log_uri") or ""),
                "artifact_locations": list(record.get("outputs") or []),
                "terminal_outcome": wave_status if terminal else "",
                "provenance": "runtime_wave_projection",
                "attribution_ambiguous": len(matching_tasks) > 1,
            }
            for index, existing in enumerate(self.stages):
                if (
                    str(existing.get("key") or "") == key
                    and int(existing.get("attempt") or 1) == attempt
                ):
                    # Preserve the last real heartbeat if a later status poll has
                    # no progress timestamp.
                    if not stage_record["last_heartbeat_at"]:
                        stage_record["last_heartbeat_at"] = str(
                            existing.get("last_heartbeat_at") or ""
                        )
                        stage_record["heartbeat_source"] = str(
                            existing.get("heartbeat_source") or ""
                        )
                    self.stages[index] = stage_record
                    break
            else:
                self.stages.append(stage_record)


def _normalized_stage_state(value: object) -> str:
    state = str(value or "").strip().upper().replace("-", "_")
    if state in {"OK", "SUCCESS", "COMPLETED", "SUCCEEDED"}:
        return "SUCCEEDED"
    if state in {"CANCELED", "CANCELLED", "ABORTED"}:
        return "CANCELLED"
    if state in {"FAILED_STARTUP", "FAILED_SETUP"}:
        return state
    if state.startswith("FAILED") or state in {"FAILURE", "ERROR", "BLOCKED"}:
        return "FAILED"
    return state or "UNKNOWN"


def _wave_members(wave: Mapping[str, Any]) -> list[tuple[str, int | None]]:
    key = str(wave.get("key") or "")
    members: list[tuple[str, int | None]] = []
    if key.count("|") >= 2:
        encoded = key.split("|", 2)[2]
        for item in encoded.split(","):
            parts = item.rsplit(":", 2)
            if len(parts) != 3:
                continue
            _loop, state, raw_iteration = parts
            iteration = int(raw_iteration) if raw_iteration.isdigit() else None
            members.append((state, iteration))
    if members:
        return members
    return [(str(item), None) for item in wave.get("states") or []]


def reconstruct_stage_job_attribution(
    manifest: RunManifest,
    *,
    runtime_waves: Sequence[Mapping[str, Any]] = (),
) -> dict[str, dict[str, Any]]:
    """Map each logical stage to only its own durable wave/attempt identities.

    A root job ID is inherited only for the legacy single-managed-job contract
    (no runtime waves).  Conflicts and missing historical evidence remain
    explicit instead of broadcasting the latest discovered job across stages.
    """

    stages: dict[str, dict[str, Any]] = {}
    by_member: dict[tuple[str, int | None], list[dict[str, Any]]] = {}
    for order, wave in enumerate(runtime_waves):
        if not isinstance(wave, Mapping):
            continue
        record = {
            "attempt": int(wave.get("attempt") or 1),
            "job_id": str(wave.get("job_id") or wave.get("sky_job_id") or ""),
            "job_name": str(wave.get("job_name") or ""),
            "wave_key": str(wave.get("key") or ""),
            "wave_kind": str(wave.get("kind") or ""),
            "group": str(wave.get("group") or ""),
            "state": _normalized_stage_state(
                wave.get("sky_status") or wave.get("status")
            ),
            "started_at": str(wave.get("started_at") or ""),
            "ended_at": str(wave.get("ended_at") or ""),
            "provenance": "runtime_wave",
            "_order": order,
        }
        for member in _wave_members(wave):
            by_member.setdefault(member, []).append(dict(record))

    for index, step in enumerate(manifest.steps):
        name = str(step.get("state") or f"step-{index}")
        raw_iteration = step.get("iteration")
        try:
            iteration = int(raw_iteration) if raw_iteration is not None else None
        except (TypeError, ValueError):
            iteration = None
        key = f"{name}#{iteration}" if iteration is not None else name
        if key in stages:
            key = f"{key}@{index}"
        attempts = list(by_member.get((name, iteration), ()))
        if not attempts and iteration is None:
            # Legacy runtime ledgers did not encode iteration in their key.
            candidates = [
                record
                for (member_name, _member_iteration), records in by_member.items()
                if member_name == name
                for record in records
            ]
            if len({item.get("wave_key") for item in candidates}) == 1:
                attempts = candidates
        step_job = str(step.get("job_id") or step.get("sky_job_id") or "").strip()
        step_job_is_unproven_root = (
            bool(runtime_waves)
            and step_job == manifest.sky_job_id
            and not any(item.get("job_id") == step_job for item in attempts)
        )
        if (
            step_job
            and not step_job_is_unproven_root
            and not any(item.get("job_id") == step_job for item in attempts)
        ):
            attempts.append(
                {
                    "attempt": int(step.get("attempt") or 1),
                    "job_id": step_job,
                    "job_name": str(step.get("job_name") or ""),
                    "wave_key": str(step.get("wave_key") or ""),
                    "wave_kind": "",
                    "group": str(step.get("group") or ""),
                    "state": _normalized_stage_state(
                        step.get("sky_status") or step.get("status")
                    ),
                    "started_at": str(step.get("start_at") or ""),
                    "ended_at": str(step.get("end_at") or ""),
                    "provenance": "manifest_step",
                    "_order": len(runtime_waves) + index,
                }
            )
        legacy_single = bool(manifest.sky_job_id) and not runtime_waves
        if not attempts and legacy_single:
            attempts.append(
                {
                    "attempt": 1,
                    "job_id": manifest.sky_job_id,
                    "job_name": "",
                    "wave_key": "",
                    "wave_kind": "legacy_single_managed_job",
                    "group": "",
                    "state": _normalized_stage_state(step.get("status")),
                    "started_at": "",
                    "ended_at": "",
                    "provenance": "legacy_single_managed_job",
                    "_order": index,
                }
            )
        attempts.sort(
            key=lambda item: (
                int(item.get("attempt") or 1),
                int(item.get("_order") or 0),
            )
        )
        final = attempts[-1] if attempts else {}
        final_attempt = int(final.get("attempt") or 0)
        final_ids = {
            str(item.get("job_id") or "")
            for item in attempts
            if int(item.get("attempt") or 0) == final_attempt
            and str(item.get("job_id") or "")
        }
        ambiguous = len(final_ids) > 1
        public_attempts = [
            {field: value for field, value in item.items() if field != "_order"}
            for item in attempts
        ]
        stages[key] = {
            "workflow_state": name,
            "iteration": iteration,
            "managed_job_id": "" if ambiguous else next(iter(final_ids), ""),
            "job_name": str(final.get("job_name") or ""),
            "attempts": public_attempts,
            "active_attempt": final_attempt or None,
            "attribution": (
                "ambiguous" if ambiguous else str(final.get("provenance") or "unknown")
            ),
            "attribution_provenance": sorted(
                {str(item.get("provenance") or "unknown") for item in attempts}
            ),
        }
    return stages


def manifest_key(prefix: str) -> str:
    base = prefix.rstrip("/")
    return f"{base}/npa-workflow/manifest.json"


def runtime_key(prefix: str) -> str:
    base = prefix.rstrip("/")
    return f"{base}/npa-workflow/runtime.json"


def status_key(prefix: str) -> str:
    base = prefix.rstrip("/")
    return f"{base}/npa-workflow/status.json"


class RunStateStore:
    """Persist workflow run manifests (mock ``reader``/``writer`` in unit tests)."""

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str,
        reader: Any | None = None,
        writer: Any | None = None,
        endpoint_url: str = "",
        aws_access_key_id: str = "",
        aws_secret_access_key: str = "",
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self._reader = reader
        self._writer = writer
        self._endpoint_url = endpoint_url
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key

    @property
    def run_prefix_uri(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}"

    def read_manifest(self) -> RunManifest | None:
        key = manifest_key(self.prefix)
        try:
            body = self._read(key)
        except FileNotFoundError:
            return None
        payload = json.loads(body)
        if not isinstance(payload, dict):
            return None
        return RunManifest.from_dict(payload)

    def write_manifest(self, manifest: RunManifest) -> dict[str, Any]:
        manifest.updated_at = utc_now()
        manifest.run_prefix_uri = self.run_prefix_uri
        payload = manifest.to_dict()
        self._write(manifest_key(self.prefix), payload)
        self._write(
            status_key(self.prefix),
            {
                "schema_version": RUN_SCHEMA_VERSION,
                "run_id": manifest.run_id,
                "workflow": manifest.workflow,
                "status": manifest.status,
                "updated_at": manifest.updated_at,
                "step_count": len(manifest.steps),
            },
        )
        return payload

    def read_runtime_state(self) -> RuntimeRunState | None:
        try:
            body = self._read(runtime_key(self.prefix))
        except FileNotFoundError:
            return None
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return RuntimeRunState.from_dict(payload)

    def write_runtime_state(self, state: RuntimeRunState) -> dict[str, Any]:
        state.updated_at = utc_now()
        state.run_prefix_uri = self.run_prefix_uri
        payload = state.to_dict()
        self._write(runtime_key(self.prefix), payload)
        return payload

    def append_step(
        self, manifest: RunManifest, step_record: Mapping[str, Any]
    ) -> dict[str, Any]:
        manifest.steps.append(dict(step_record))
        return self.write_manifest(manifest)

    def write_artifact(
        self,
        relative_key: str,
        body: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Write one non-empty run artifact beneath this run's exact prefix."""

        key = str(relative_key or "").strip().lstrip("/")
        if not key or ".." in key.split("/"):
            raise ValueError("run artifact key must be a safe relative path")
        if not body:
            raise ValueError("run artifact body must be non-empty")
        target = f"{self.prefix}/{key}"
        if self._writer is not None:
            self._writer(self.bucket, target, body)
        else:
            from npa.clients.storage import StorageClient

            client = StorageClient.from_environment(
                endpoint_url=self._endpoint_url,
                aws_access_key_id=self._aws_access_key_id,
                aws_secret_access_key=self._aws_secret_access_key,
            )
            client._s3.put_object(
                Bucket=self.bucket,
                Key=target,
                Body=body,
                ContentType=content_type,
            )
        return f"s3://{self.bucket}/{target}"

    def _read(self, key: str) -> str:
        if self._reader is not None:
            return str(self._reader(self.bucket, key))
        from npa.clients.storage import StorageClient

        client = StorageClient.from_environment(
            endpoint_url=self._endpoint_url,
            aws_access_key_id=self._aws_access_key_id,
            aws_secret_access_key=self._aws_secret_access_key,
        )
        try:
            response = client._s3.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            raise FileNotFoundError(f"s3://{self.bucket}/{key}") from exc
        return response["Body"].read().decode("utf-8")

    def _write(self, key: str, payload: Mapping[str, Any]) -> None:
        body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        if self._writer is not None:
            self._writer(self.bucket, key, body)
            return
        from npa.clients.storage import StorageClient

        client = StorageClient.from_environment(
            endpoint_url=self._endpoint_url,
            aws_access_key_id=self._aws_access_key_id,
            aws_secret_access_key=self._aws_secret_access_key,
        )
        client._s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )


SUBMITTED_STATUS = "submitted"

_TASK_STATUS_MAP = {
    "SUCCEEDED": "SUCCEEDED",
    "FAILED": "FAILED",
    "FAILED_SETUP": "FAILED_SETUP",
    "CANCELLED": "CANCELLED",
    "RUNNING": "RUNNING",
    "STARTING": "STARTING",
    "PENDING": "PENDING",
}

TERMINAL_STEP_STATES = frozenset(
    {"SUCCEEDED", "CANCELLED", "FAILED", "FAILED_SETUP", "FAILED_STARTUP", "BLOCKED"}
)
_DETERMINISTIC_STARTUP_PATTERNS = (
    re.compile(
        r"container\s+not found.*ray-node|ray-node.*container\s+not found",
        re.IGNORECASE,
    ),
    re.compile(r"ray-node.*(?:deleted|not found)", re.IGNORECASE),
    re.compile(r"cannot exec in a deleted state", re.IGNORECASE),
)
NORMALIZED_DELETED_RAY_NODE = (
    "ray-node container deleted before SkyPilot initialization"
)


def reconcile_submitted_manifest(
    manifest: RunManifest,
    *,
    live_status: str = "",
    task_rows: Sequence[Mapping[str, Any]] = (),
) -> RunManifest:
    """Merge SkyPilot outcomes into a submitted manifest without losing lineage."""

    startup_terminal = str(manifest.status or "").upper() == "FAILED_STARTUP"
    rows_by_id = {
        int(row["task_id"]): row
        for row in task_rows
        if str(row.get("task_id", "")).isdigit()
    }
    reconciled: list[dict[str, Any]] = []
    for index, original in enumerate(manifest.steps):
        step = dict(original)
        row = rows_by_id.get(index)
        if row is not None:
            sky_status = str(row.get("status") or "").upper()
            if sky_status:
                reconciled_status = _TASK_STATUS_MAP.get(sky_status, sky_status)
                if not (
                    str(original.get("status") or "").upper() == "FAILED_STARTUP"
                    and reconciled_status not in {"SUCCEEDED", "CANCELLED"}
                ):
                    step["status"] = reconciled_status
                step["sky_status"] = sky_status
            for field in (
                "task_id",
                "task_name",
                "submitted_at",
                "start_at",
                "end_at",
                "retry_count",
                "last_progress_at",
                "last_updated_at",
                "failure_reason",
            ):
                if row.get(field) not in (None, ""):
                    step[field] = row[field]
        reconciled.append(step)

    live = str(live_status or "").upper()
    if live == "SUCCEEDED":
        # The managed pipeline cannot report overall success until every task has
        # completed. Preserve richer task-row fields where available, but do not
        # leave individual stages permanently "submitted" merely because a queue
        # detail query was temporarily unavailable after terminal success.
        for step in reconciled:
            step["status"] = "SUCCEEDED"
    elif (live.startswith("FAILED") or live == "CANCELLED") and not any(
        str(step.get("status") or "").upper().startswith("FAILED")
        or str(step.get("status") or "").upper() == "CANCELLED"
        for step in reconciled
    ):
        first_incomplete = next(
            (
                step
                for step in reconciled
                if str(step.get("status") or "").upper() != "SUCCEEDED"
            ),
            None,
        )
        if first_incomplete is not None:
            first_incomplete["status"] = live
    if startup_terminal and live not in {"SUCCEEDED", "CANCELLED"}:
        manifest.status = "FAILED_STARTUP"
    elif live:
        manifest.status = live
    elif reconciled:
        statuses = [str(step.get("status") or "").upper() for step in reconciled]
        if any(
            status.startswith("FAILED") or status == "CANCELLED" for status in statuses
        ):
            manifest.status = "FAILED"
        elif all(status == "SUCCEEDED" for status in statuses):
            manifest.status = "SUCCEEDED"
        elif any(status in {"RUNNING", "STARTING"} for status in statuses):
            manifest.status = "RUNNING"
    manifest.steps = reconciled
    return manifest


def _timestamp(value: object) -> datetime | None:
    if value in (None, "", 0, 0.0):
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    try:
        if text.replace(".", "", 1).isdigit():
            return datetime.fromtimestamp(float(text), tz=timezone.utc)
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except (OverflowError, OSError, ValueError):
        return None


def _iso(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_startup_failure(controller_output: str) -> tuple[str, int]:
    """Return a stable startup-failure label and its evidence count."""

    matches = sum(
        1
        for line in str(controller_output or "").splitlines()
        if any(pattern.search(line) for pattern in _DETERMINISTIC_STARTUP_PATTERNS)
    )
    return (NORMALIZED_DELETED_RAY_NODE, matches) if matches else ("", 0)


def build_actionable_run_status(
    manifest: RunManifest,
    *,
    live_status: str = "",
    task_rows: Sequence[Mapping[str, Any]] = (),
    runtime_waves: Sequence[Mapping[str, Any]] = (),
    job_observations: Mapping[str, Mapping[str, Any]] | None = None,
    controller_output: str = "",
    project: str = "",
    failure_threshold: int = 3,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Project manifest + scheduler evidence into actionable stage status.

    The S3 manifest remains the only durable workflow status.  This projection
    enriches those same steps with scheduler fields and may terminalize a
    repeated deterministic startup failure while retaining raw controller state.
    """

    if failure_threshold <= 0:
        raise ValueError("startup failure threshold must be positive")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    rows = {
        int(row["task_id"]): row
        for row in task_rows
        if str(row.get("task_id", "")).isdigit()
    }
    attributions = reconstruct_stage_job_attribution(
        manifest, runtime_waves=runtime_waves
    )
    observations = dict(job_observations or {})
    normalized_failure, failure_evidence = normalize_startup_failure(controller_output)
    stages: dict[str, dict[str, Any]] = {}
    active_key = ""
    active_index: int | None = None
    newest_progress: datetime | None = None
    newest_observed: datetime | None = _timestamp(manifest.updated_at)
    for index, step in enumerate(manifest.steps):
        name = str(step.get("state") or f"step-{index}")
        iteration = step.get("iteration")
        key = f"{name}#{iteration}" if iteration is not None else name
        if key in stages:
            key = f"{key}@{index}"
        attribution = attributions.get(key, {})
        managed_job_id = str(attribution.get("managed_job_id") or "")
        observation = observations.get(managed_job_id, {})
        observed_rows = [
            item
            for item in observation.get("task_rows") or []
            if isinstance(item, Mapping)
        ]
        named_rows = [
            item for item in observed_rows if str(item.get("task_name") or "") == name
        ]
        row = (
            named_rows[-1]
            if named_rows
            else observed_rows[-1]
            if len(observed_rows) == 1
            else rows.get(index, {})
            if not observations
            else {}
        )
        scheduler_job_state = str(observation.get("status") or "").upper()
        raw_scheduler = str(
            row.get("status") or scheduler_job_state or step.get("sky_status") or ""
        ).upper()
        step_state = _normalized_stage_state(step.get("status") or "SUBMITTED")
        final_attempt = next(
            (
                item
                for item in reversed(list(attribution.get("attempts") or []))
                if isinstance(item, Mapping)
            ),
            {},
        )
        attempt_state = _normalized_stage_state(final_attempt.get("state"))
        scheduler_state = _normalized_stage_state(raw_scheduler)
        step_terminal = step_state in TERMINAL_STEP_STATES | {
            "SUCCEEDED",
            "FAILED",
        }
        attempt_terminal = attempt_state in TERMINAL_STEP_STATES | {
            "SUCCEEDED",
            "FAILED",
        }
        scheduler_terminal = scheduler_state in TERMINAL_STEP_STATES | {
            "SUCCEEDED",
            "FAILED",
        }
        outcome_conflict = scheduler_terminal and (
            (step_terminal and step_state != scheduler_state)
            or (
                not step_terminal
                and attempt_terminal
                and attempt_state != scheduler_state
            )
        )
        if outcome_conflict:
            state = "UNKNOWN"
            outcome_provenance = "conflicting_durable_and_scheduler_evidence"
        elif step_terminal:
            state = step_state
            outcome_provenance = "authoritative_stage_record"
        elif raw_scheduler:
            state = _TASK_STATUS_MAP.get(scheduler_state, scheduler_state)
            outcome_provenance = "scheduler_final_attempt"
        elif attempt_state != "UNKNOWN":
            state = attempt_state
            outcome_provenance = "durable_runtime_attempt"
        else:
            state = step_state
            outcome_provenance = "durable_stage_record"
        retry_count = 0
        for field_name in ("retry_count", "recovery_count", "num_restarts", "attempt"):
            raw = row.get(field_name, step.get(field_name))
            try:
                retry_count = max(retry_count, int(raw or 0))
            except (TypeError, ValueError):
                continue
        retry_count = max(
            retry_count,
            max(
                (
                    int(item.get("attempt") or 1) - 1
                    for item in attribution.get("attempts") or []
                    if isinstance(item, Mapping)
                ),
                default=0,
            ),
        )
        heartbeat_timestamps = [
            value
            for value in (
                _timestamp(row.get("last_progress_at")),
                _timestamp(row.get("last_heartbeat_at")),
                _timestamp(step.get("last_progress_at")),
                _timestamp(step.get("last_heartbeat_at")),
            )
            if value is not None
        ]
        last_progress = max(heartbeat_timestamps) if heartbeat_timestamps else None
        observed_timestamps = [
            value
            for value in (
                _timestamp(observation.get("observed_at")),
                _timestamp(row.get("last_updated_at", step.get("last_updated_at"))),
                _timestamp(row.get("end_at", step.get("end_at"))),
                _timestamp(row.get("start_at", step.get("start_at"))),
                _timestamp(row.get("submitted_at", step.get("submitted_at"))),
                _timestamp(final_attempt.get("ended_at")),
                _timestamp(final_attempt.get("started_at")),
                _timestamp(manifest.updated_at),
            )
            if value is not None
        ]
        last_observed = max(observed_timestamps) if observed_timestamps else None
        if last_progress and (
            newest_progress is None or last_progress > newest_progress
        ):
            newest_progress = last_progress
        if last_observed and (
            newest_observed is None or last_observed > newest_observed
        ):
            newest_observed = last_observed
        start = (
            _timestamp(row.get("start_at", step.get("start_at")))
            or _timestamp(row.get("submitted_at", step.get("submitted_at")))
            or _timestamp(final_attempt.get("started_at"))
        )
        end = (
            _timestamp(row.get("end_at", step.get("end_at")))
            or _timestamp(final_attempt.get("ended_at"))
            or current
        )
        log_command = (
            f"npa workbench workflow logs {manifest.run_id} --stage {name}"
            + (f" --project {project}" if project else "")
        )
        profile = step.get("resources_profile") or {}
        stage_payload: dict[str, Any] = {
            "index": index + 1,
            "state": state,
            "workflow_state": name,
            "managed_job_id": managed_job_id,
            "managed_job_attempts": list(attribution.get("attempts") or []),
            "active_attempt": attribution.get("active_attempt"),
            "job_attribution": attribution.get("attribution", "unknown"),
            "job_attribution_provenance": list(
                attribution.get("attribution_provenance") or []
            ),
            "task_id": row.get("task_id", index),
            "scheduler_state": raw_scheduler or state,
            "raw_scheduler_state": raw_scheduler,
            "outcome_provenance": outcome_provenance,
            "outcome_conflict": outcome_conflict,
            "retry_count": retry_count,
            "last_progress_at": _iso(last_progress),
            "last_heartbeat_at": _iso(last_progress),
            "heartbeat_source": "scheduler_task_progress" if last_progress else "",
            "last_observed_at": _iso(last_observed),
            "elapsed_seconds": max(0, int((end - start).total_seconds()))
            if start
            else None,
            "staleness_seconds": (
                max(0, int((current - last_progress).total_seconds()))
                if last_progress
                else None
            ),
            "last_normalized_startup_failure": str(
                step.get("last_normalized_startup_failure") or ""
            ),
            "startup_failure_evidence": int(step.get("startup_failure_evidence") or 0),
            "log_command": log_command,
            "requested_accelerators": (
                str(profile.get("accelerators") or "")
                if isinstance(profile, dict)
                else ""
            ),
            "resources_profile": profile,
        }
        if active_index is None and state == "FAILED_STARTUP":
            active_index = index + 1
            active_key = key
        elif active_index is None and state not in TERMINAL_STEP_STATES:
            active_index = index + 1
            active_key = key
            if normalized_failure:
                stage_payload["last_normalized_startup_failure"] = normalized_failure
                stage_payload["startup_failure_evidence"] = failure_evidence
                stage_payload["retry_count"] = max(
                    retry_count, max(0, failure_evidence - 1)
                )
                if failure_evidence >= failure_threshold:
                    state = "FAILED_STARTUP"
                    stage_payload["state"] = state
                    step["status"] = state
                    step["last_normalized_startup_failure"] = normalized_failure
                    step["startup_failure_evidence"] = failure_evidence
                    manifest.status = "FAILED_STARTUP"
        stages[key] = stage_payload

    live = str(live_status or "").upper()
    status = str(manifest.status or "SUBMITTED").upper()
    if status != "FAILED_STARTUP":
        active = stages.get(active_key, {})
        scheduler = str(active.get("scheduler_state") or "").upper()
        retries = int(active.get("retry_count") or 0)
        stage_states = [str(item.get("state") or "UNKNOWN") for item in stages.values()]
        all_terminal = bool(stage_states) and all(
            state in TERMINAL_STEP_STATES or state == "UNKNOWN"
            for state in stage_states
        )
        if all_terminal and "UNKNOWN" in stage_states:
            status = "UNKNOWN"
        elif all_terminal and any(state.startswith("FAILED") for state in stage_states):
            status = "FAILED"
        elif all_terminal and "CANCELLED" in stage_states:
            status = "CANCELLED"
        elif all_terminal and all(state == "SUCCEEDED" for state in stage_states):
            status = "SUCCEEDED"
        elif scheduler in {"PENDING", "STARTING"}:
            status = "RETRYING" if retries else scheduler
        elif live:
            status = live
    manifest.status = status
    stage_job_ids = sorted(
        {
            str(item.get("managed_job_id") or "")
            for item in stages.values()
            if str(item.get("managed_job_id") or "")
        }
    )
    return {
        "run_id": manifest.run_id,
        "workflow_name": manifest.workflow,
        "status": status,
        "live_status": live,
        "raw_controller_state": live,
        "sky_job_id": manifest.sky_job_id,
        "sky_job_ids": stage_job_ids,
        "active_stage_name": str(
            stages.get(active_key, {}).get("workflow_state") or ""
        ),
        "active_stage_index": active_index,
        "last_heartbeat_at": _iso(newest_progress),
        "last_observed_at": _iso(newest_observed),
        "heartbeat_age_seconds": (
            max(0, int((current - newest_progress).total_seconds()))
            if newest_progress
            else None
        ),
        "heartbeat_stale": (
            newest_progress is None
            or max(0, int((current - newest_progress).total_seconds())) > 300
        ),
        "stages": stages,
    }


def plan_step_records(
    steps: Sequence[Any],
    *,
    accelerator_overrides: Mapping[str, str] | None = None,
    accelerator_override: str = "",
) -> list[dict[str, Any]]:
    """Build manifest step records for a *submitted* (not locally executed) run.

    Mirrors the interpreter's local-execution records so one manifest schema covers
    both paths. Carrying ``resources_profile`` is the point: it is what makes a
    submitted run legible to downstream consumers — the insights backbone derives a
    run's GPU count from ``resources_profile.accelerators``, so a manifest without it
    describes a run that looks CPU-only no matter how many accelerators it requested.
    """
    overrides = dict(accelerator_overrides or {})
    records: list[dict[str, Any]] = []
    for step in steps:
        resources_profile = dict(getattr(step, "resources_profile", {}) or {})
        requested = str(resources_profile.get("accelerators") or "").strip()
        if requested:
            if accelerator_override:
                resources_profile["accelerators"] = accelerator_override
            elif requested in overrides:
                resources_profile["accelerators"] = overrides[requested]
        record: dict[str, Any] = {
            "state": getattr(step, "state", ""),
            "iteration": getattr(step, "iteration", None),
            "status": SUBMITTED_STATUS,
            "resources": getattr(step, "resources", "") or "",
            "resources_profile": resources_profile,
        }
        for optional in ("tool_ref", "group", "loop_label"):
            value = getattr(step, optional, "")
            if value:
                record[optional] = value
        record["inputs"] = [dict(item) for item in getattr(step, "inputs", ()) or ()]
        record["outputs"] = [dict(item) for item in getattr(step, "outputs", ()) or ()]
        records.append(record)
    return records


def persist_submitted_manifest(
    config: Mapping[str, Any],
    *,
    run_id: str,
    workflow: str,
    api_version: str = "",
    steps: Sequence[Any] = (),
    status: str = SUBMITTED_STATUS,
    sky_job_id: str = "",
    endpoint_url: str = "",
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    accelerator_overrides: Mapping[str, str] | None = None,
    accelerator_override: str = "",
) -> str:
    """Write the run manifest for a cluster-submitted run; return the run prefix URI.

    Returns ``""`` when the spec declares no ``config.bucket`` (there is nowhere to
    write). Raises on a genuine write failure so callers can surface it — a submit
    that was already accepted must not be reported as failed, but a silently missing
    manifest would leave the run invisible to every manifest consumer.
    """
    store = store_for_config(
        config,
        run_id=run_id,
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    if store is None:
        return ""
    manifest = RunManifest(
        workflow=workflow,
        run_id=run_id,
        api_version=api_version,
        status=status,
        sky_job_id=sky_job_id,
        input_source=input_source_from_config(config),
    )
    manifest.steps = plan_step_records(
        steps,
        accelerator_overrides=accelerator_overrides,
        accelerator_override=accelerator_override,
    )
    store.write_manifest(manifest)
    return store.run_prefix_uri


def input_source_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Project PAIDF's non-secret input contract into the durable run manifest."""

    source_kind = str(config.get("input_source_kind") or "").strip()
    if not source_kind:
        return {}
    return {
        "source_kind": source_kind,
        "input_origin": str(config.get("input_origin") or ""),
        "input_origin_label": str(config.get("input_origin_label") or ""),
        "authoritative_upstream_url": str(config.get("input_authoritative_url") or ""),
        "immutable_revision": str(config.get("input_immutable_revision") or ""),
        "asset_license": str(config.get("input_license") or ""),
        "asset_attribution": str(config.get("input_attribution") or ""),
        "sha256": str(config.get("input_sha256") or ""),
        "staged_canonical_s3_uri": str(config.get("input_staged_uri") or ""),
        "provenance_uri": str(config.get("input_provenance_uri") or ""),
    }


def store_for_config(
    config: Mapping[str, Any],
    *,
    run_id: str,
    endpoint_url: str = "",
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
) -> RunStateStore | None:
    bucket = str(config.get("bucket") or "").strip()
    prefix = str(config.get("prefix") or run_id).strip()
    if not bucket:
        return None
    return RunStateStore(
        bucket=bucket,
        prefix=prefix,
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
