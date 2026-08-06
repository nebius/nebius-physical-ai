"""Authoritative, repeat-safe cancellation planning for durable workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from npa.orchestration.npa_workflow.run_resolution import RunResolution
from npa.orchestration.skypilot.workflow import ManagedJobEvidence, lookup_managed_job
from npa.orchestration.skypilot.workflow_state import (
    WorkflowStateError,
    read_stage_status,
    workflow_state_error_is_missing,
)


_TERMINAL_OK = {"SUCCEEDED"}
_TERMINAL_FAILURE = {"FAILED", "CANCELLED"}
_NONTERMINAL = {
    "PENDING",
    "STARTING",
    "SUBMITTED",
    "RUNNING",
    "RECOVERING",
    "RETRYING",
    "CANCELLING",
    "MANIFEST_PENDING",
}


def normalize_workflow_state(value: object) -> str:
    """Normalize persisted/runtime/SkyPilot spelling into one state vocabulary."""

    state = str(value or "").strip().upper().replace("-", "_")
    if not state:
        return ""
    if state in {"OK", "SUCCESS", "COMPLETED"}:
        return "SUCCEEDED"
    if state.startswith("FAILED") or state in {"FAILURE", "ERROR", "BLOCKED"}:
        return "FAILED"
    if state in {"CANCELED", "ABORTED"}:
        return "CANCELLED"
    return state


def is_terminal_workflow_state(value: object) -> bool:
    state = normalize_workflow_state(value)
    return state in _TERMINAL_OK | _TERMINAL_FAILURE


@dataclass
class WorkflowJobRecord:
    """One immutable managed-job identity recovered from durable run state."""

    job_id: str
    job_name: str = ""
    persisted_states: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    live_outcome: str = ""
    live_status: str = ""
    live_error: str = ""

    @property
    def terminal_in_durable_state(self) -> bool:
        states = {normalize_workflow_state(item) for item in self.persisted_states}
        return bool(states) and all(is_terminal_workflow_state(item) for item in states)

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "job_name": self.job_name,
            "persisted_states": sorted(self.persisted_states),
            "sources": sorted(self.sources),
            "live_outcome": self.live_outcome,
            "live_status": self.live_status,
            "live_error": self.live_error,
        }


@dataclass
class CancellationAssessment:
    """Cancellation decision made before any mutating SkyPilot call."""

    detected_state: str
    jobs: list[WorkflowJobRecord] = field(default_factory=list)
    active_jobs: list[WorkflowJobRecord] = field(default_factory=list)
    terminal_jobs: list[WorkflowJobRecord] = field(default_factory=list)
    absent_jobs: list[WorkflowJobRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def no_cancellation_needed(self) -> bool:
        return not self.active_jobs and not self.errors


LookupFn = Callable[..., ManagedJobEvidence]


def assess_run_cancellation(
    resolution: RunResolution,
    *,
    sky_bin: str = "",
    lookup: LookupFn | None = None,
) -> CancellationAssessment:
    """Inspect every durable job/stage record and identify only active jobs.

    Root manifests predate runtime waves and may legitimately have no singular
    ``sky_job_id``.  The runtime ledger, per-step/per-stage records, and exact
    SkyPilot queue evidence are therefore considered together.  Provider/auth
    unavailability remains an error; a successful exact lookup returning absence
    is authoritative convergence.
    """

    records: dict[str, WorkflowJobRecord] = {}
    errors: list[str] = []
    if resolution.runtime_state_error:
        errors.append(
            "runtime ledger verification failed: " + resolution.runtime_state_error
        )
    terminal_candidates: list[str] = []
    nonterminal_records_without_id: list[str] = []
    runtime = resolution.runtime_state
    raw_runtime_waves = list(runtime.get("waves") or [])
    for index, wave in enumerate(raw_runtime_waves):
        if not isinstance(wave, dict):
            errors.append(f"runtime wave {index} is malformed")
    runtime_waves = [item for item in raw_runtime_waves if isinstance(item, dict)]

    def add_job(
        job_id: object,
        *,
        job_name: object = "",
        state: object = "",
        source: str,
    ) -> None:
        cleaned_id = str(job_id or "").strip()
        cleaned_state = normalize_workflow_state(state)
        if not cleaned_id:
            child_uses_root_job = (
                source.startswith(("manifest step ", "stage "))
                and bool(str(manifest.get("sky_job_id") or "").strip())
                and not runtime_waves
            )
            if (
                cleaned_state in _NONTERMINAL
                and source != "root manifest"
                and not child_uses_root_job
            ):
                nonterminal_records_without_id.append(
                    f"{source} is {cleaned_state} but has no managed-job ID"
                )
            return
        record = records.setdefault(cleaned_id, WorkflowJobRecord(job_id=cleaned_id))
        cleaned_name = str(job_name or "").strip()
        if cleaned_name and (
            not record.job_name or record.job_name == resolution.run_id
        ):
            record.job_name = cleaned_name
        if cleaned_state:
            record.persisted_states.add(cleaned_state)
        record.sources.add(source)

    manifest = resolution.manifest if isinstance(resolution.manifest, dict) else {}
    manifest_state = normalize_workflow_state(manifest.get("status"))
    root_job_id = str(manifest.get("sky_job_id") or "").strip()
    runtime_job_ids = {
        str(wave.get("job_id") or wave.get("sky_job_id") or "").strip()
        for wave in runtime_waves
        if isinstance(wave, dict)
    }
    if is_terminal_workflow_state(manifest_state):
        terminal_candidates.append(manifest_state)
    if not runtime_waves:
        add_job(
            manifest.get("sky_job_id"),
            job_name=manifest.get("job_name") or resolution.run_id,
            state=manifest_state,
            source="root manifest",
        )

    for index, step in enumerate(manifest.get("steps") or []):
        if not isinstance(step, dict):
            continue
        step_state = normalize_workflow_state(
            step.get("sky_status") or step.get("status") or step.get("state")
        )
        step_job_id = str(step.get("job_id") or step.get("sky_job_id") or "").strip()
        if not runtime_waves or (
            step_job_id
            and (step_job_id != root_job_id or step_job_id in runtime_job_ids)
        ):
            add_job(
                step_job_id,
                job_name=step.get("job_name"),
                state=step_state,
                source=f"manifest step {step.get('state') or index}",
            )

    legacy_stages = manifest.get("stages")
    if isinstance(legacy_stages, dict):
        stage_states: list[str] = []
        for stage_name, raw in legacy_stages.items():
            info = dict(raw) if isinstance(raw, dict) else {}
            if resolution.state is not None:
                try:
                    persisted = read_stage_status(resolution.state, str(stage_name))
                except WorkflowStateError as exc:
                    if not workflow_state_error_is_missing(exc):
                        errors.append(
                            f"stage {stage_name} status verification failed: {exc}"
                        )
                else:
                    if isinstance(persisted, dict):
                        info.update(persisted)
            stage_state = normalize_workflow_state(
                info.get("state") or info.get("status") or info.get("sky_status")
            )
            if stage_state:
                stage_states.append(stage_state)
            stage_job_id = str(
                info.get("sky_job_id") or info.get("job_id") or ""
            ).strip()
            if not runtime_waves or (
                stage_job_id
                and (stage_job_id != root_job_id or stage_job_id in runtime_job_ids)
            ):
                add_job(
                    stage_job_id,
                    job_name=info.get("job_name"),
                    state=stage_state,
                    source=f"stage {stage_name}",
                )
        if stage_states and all(
            is_terminal_workflow_state(item) for item in stage_states
        ):
            terminal_candidates.append(_aggregate_terminal_states(stage_states))

    runtime_state = normalize_workflow_state(runtime.get("status"))
    if is_terminal_workflow_state(runtime_state):
        terminal_candidates.append(runtime_state)
    waves = runtime_waves
    if isinstance(waves, list):
        wave_states: list[str] = []
        for index, wave in enumerate(waves):
            wave_state = normalize_workflow_state(
                wave.get("sky_status") or wave.get("status")
            )
            if wave_state:
                wave_states.append(wave_state)
            add_job(
                wave.get("job_id") or wave.get("sky_job_id"),
                job_name=wave.get("job_name"),
                state=wave_state,
                source=f"runtime wave {wave.get('key') or index}",
            )
        if wave_states and all(
            is_terminal_workflow_state(item) for item in wave_states
        ):
            terminal_candidates.append(_aggregate_terminal_states(wave_states))

    launch = resolution.receipt.get("launch")
    if isinstance(launch, dict) and not runtime_waves:
        add_job(
            launch.get("sky_job_id") or launch.get("job_id"),
            job_name=launch.get("job_name") or resolution.run_id,
            state=launch.get("status"),
            source="submission receipt",
        )
    resolved_job_id = str(resolution.job_id or "").strip()
    if not runtime_waves or resolved_job_id in runtime_job_ids:
        add_job(
            resolved_job_id,
            job_name=resolution.job_name or resolution.run_id,
            state=(
                resolution.managed_job.status
                if resolution.managed_job is not None
                and resolution.managed_job.outcome == "found"
                else ""
            ),
            source=f"run resolution ({resolution.source or 'exact sources'})",
        )

    # A terminal authoritative run state is sufficient when it carries no job
    # identity. If identities do exist, verify any record not itself terminal so
    # a stale root status cannot hide a still-running retry/wave.
    detected_terminal = _aggregate_terminal_states(terminal_candidates)
    lookup_fn = lookup or lookup_managed_job
    active: list[WorkflowJobRecord] = []
    terminal: list[WorkflowJobRecord] = []
    absent: list[WorkflowJobRecord] = []
    for record in records.values():
        if record.terminal_in_durable_state:
            record.live_outcome = "durable_terminal"
            record.live_status = _aggregate_terminal_states(record.persisted_states)
            terminal.append(record)
            continue
        evidence = _cached_evidence(resolution, record.job_id)
        if evidence is None:
            evidence = lookup_fn(
                record.job_name or resolution.run_id,
                job_id=record.job_id,
                sky_bin=sky_bin or None,
            )
        record.live_outcome = evidence.outcome
        record.live_status = normalize_workflow_state(evidence.status)
        record.live_error = evidence.error
        if evidence.outcome == "absent":
            absent.append(record)
        elif evidence.outcome == "unavailable":
            errors.append(
                f"managed job {record.job_id} ({record.job_name or resolution.run_id}) "
                f"could not be verified: {evidence.error or 'provider unavailable'}"
            )
        elif is_terminal_workflow_state(record.live_status):
            terminal.append(record)
        elif evidence.outcome == "found":
            active.append(record)
        else:
            errors.append(
                f"managed job {record.job_id} returned unsupported verification "
                f"outcome {evidence.outcome!r}"
            )

    if nonterminal_records_without_id:
        errors.extend(nonterminal_records_without_id)
    if not records and not detected_terminal:
        managed = resolution.managed_job
        if managed is not None and managed.outcome == "unavailable":
            errors.append(
                "exact managed-job verification is unavailable: "
                + (managed.error or "provider unavailable")
            )
        elif resolution.verification_unavailable:
            errors.append(
                "run state is non-terminal and at least one authoritative source "
                "could not be verified"
            )

    if not active and terminal and len(terminal) + len(absent) == len(records):
        detected_terminal = detected_terminal or _aggregate_terminal_states(
            record.live_status or next(iter(record.persisted_states), "")
            for record in terminal
        )
    detected_state = (
        "VERIFICATION_UNAVAILABLE"
        if errors
        else "ACTIVE"
        if active
        else detected_terminal or "NO_ACTIVE_JOB"
    )
    return CancellationAssessment(
        detected_state=detected_state,
        jobs=sorted(records.values(), key=lambda item: _job_sort_key(item.job_id)),
        active_jobs=sorted(active, key=lambda item: _job_sort_key(item.job_id)),
        terminal_jobs=sorted(terminal, key=lambda item: _job_sort_key(item.job_id)),
        absent_jobs=sorted(absent, key=lambda item: _job_sort_key(item.job_id)),
        errors=errors,
    )


def _cached_evidence(
    resolution: RunResolution, job_id: str
) -> ManagedJobEvidence | None:
    evidence = resolution.managed_job
    if evidence is None:
        return None
    resolved_id = str(evidence.job_id or resolution.job_id or "").strip()
    return evidence if resolved_id == str(job_id).strip() else None


def _aggregate_terminal_states(states: Any) -> str:
    normalized = [normalize_workflow_state(item) for item in states if item]
    if any(item == "FAILED" for item in normalized):
        return "FAILED"
    if any(item == "CANCELLED" for item in normalized):
        return "CANCELLED"
    if normalized and all(item == "SUCCEEDED" for item in normalized):
        return "SUCCEEDED"
    return ""


def _job_sort_key(job_id: str) -> tuple[int, int | str]:
    return (0, int(job_id)) if str(job_id).isdigit() else (1, str(job_id))
