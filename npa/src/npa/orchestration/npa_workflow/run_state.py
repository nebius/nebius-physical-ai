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
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class RunManifest:
    workflow: str
    run_id: str
    api_version: str
    run_prefix_uri: str = ""
    status: str = "planned"
    sky_job_id: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
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
            steps=[dict(item) for item in payload.get("steps") or [] if isinstance(item, dict)],
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
            waves=[dict(item) for item in payload.get("waves") or [] if isinstance(item, dict)],
            decisions=[
                dict(item) for item in payload.get("decisions") or [] if isinstance(item, dict)
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
            return dict(record) if status == "running" else None
        return None

    def record_wave(self, record: Mapping[str, Any]) -> None:
        key = str(record.get("key") or "")
        for index, existing in enumerate(self.waves):
            if existing.get("key") == key:
                self.waves[index] = dict(record)
                return
        self.waves.append(dict(record))


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

    def append_step(self, manifest: RunManifest, step_record: Mapping[str, Any]) -> dict[str, Any]:
        manifest.steps.append(dict(step_record))
        return self.write_manifest(manifest)

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
    re.compile(r"container\s+not found.*ray-node|ray-node.*container\s+not found", re.IGNORECASE),
    re.compile(r"ray-node.*(?:deleted|not found)", re.IGNORECASE),
    re.compile(r"cannot exec in a deleted state", re.IGNORECASE),
)
NORMALIZED_DELETED_RAY_NODE = "ray-node container deleted before SkyPilot initialization"


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
        if any(status.startswith("FAILED") or status == "CANCELLED" for status in statuses):
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
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
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
    normalized_failure, failure_evidence = normalize_startup_failure(controller_output)
    stages: dict[str, dict[str, Any]] = {}
    active_key = ""
    active_index: int | None = None
    newest_progress: datetime | None = _timestamp(manifest.updated_at)
    for index, step in enumerate(manifest.steps):
        name = str(step.get("state") or f"step-{index}")
        iteration = step.get("iteration")
        key = f"{name}#{iteration}" if iteration is not None else name
        if key in stages:
            key = f"{key}@{index}"
        row = rows.get(index, {})
        raw_scheduler = str(row.get("status") or step.get("sky_status") or "").upper()
        state = str(step.get("status") or "SUBMITTED").upper()
        if raw_scheduler:
            state = _TASK_STATUS_MAP.get(raw_scheduler, raw_scheduler)
        retry_count = 0
        for field_name in ("retry_count", "recovery_count", "num_restarts", "attempt"):
            raw = row.get(field_name, step.get(field_name))
            try:
                retry_count = max(retry_count, int(raw or 0))
            except (TypeError, ValueError):
                continue
        timestamps = [
            value
            for value in (
                _timestamp(row.get("last_progress_at", step.get("last_progress_at"))),
                _timestamp(row.get("last_updated_at", step.get("last_updated_at"))),
                _timestamp(row.get("end_at", step.get("end_at"))),
                _timestamp(row.get("start_at", step.get("start_at"))),
                _timestamp(row.get("submitted_at", step.get("submitted_at"))),
                _timestamp(manifest.updated_at),
            )
            if value is not None
        ]
        last_progress = max(timestamps) if timestamps else None
        if last_progress and (newest_progress is None or last_progress > newest_progress):
            newest_progress = last_progress
        start = _timestamp(row.get("start_at", step.get("start_at"))) or _timestamp(
            row.get("submitted_at", step.get("submitted_at"))
        )
        end = _timestamp(row.get("end_at", step.get("end_at"))) or current
        log_command = (
            f"npa workbench workflow logs {manifest.run_id} --stage {name}"
            + (f" --project {project}" if project else "")
        )
        profile = step.get("resources_profile") or {}
        stage_payload: dict[str, Any] = {
            "index": index + 1,
            "state": state,
            "workflow_state": name,
            "managed_job_id": manifest.sky_job_id,
            "task_id": row.get("task_id", index),
            "scheduler_state": raw_scheduler or state,
            "raw_scheduler_state": raw_scheduler,
            "retry_count": retry_count,
            "last_progress_at": _iso(last_progress),
            "elapsed_seconds": max(0, int((end - start).total_seconds())) if start else None,
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
                str(profile.get("accelerators") or "") if isinstance(profile, dict) else ""
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
                stage_payload["retry_count"] = max(retry_count, max(0, failure_evidence - 1))
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
        if scheduler in {"PENDING", "STARTING"}:
            status = "RETRYING" if retries else scheduler
        elif live:
            status = live
        elif stages and all(str(item.get("state")) == "SUCCEEDED" for item in stages.values()):
            status = "SUCCEEDED"
    manifest.status = status
    return {
        "run_id": manifest.run_id,
        "workflow_name": manifest.workflow,
        "status": status,
        "live_status": live,
        "raw_controller_state": live,
        "sky_job_id": manifest.sky_job_id,
        "active_stage_name": str(stages.get(active_key, {}).get("workflow_state") or ""),
        "active_stage_index": active_index,
        "last_heartbeat_at": _iso(newest_progress),
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
    )
    manifest.steps = plan_step_records(
        steps,
        accelerator_overrides=accelerator_overrides,
        accelerator_override=accelerator_override,
    )
    store.write_manifest(manifest)
    return store.run_prefix_uri


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
