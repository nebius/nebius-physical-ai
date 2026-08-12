from __future__ import annotations

from npa.orchestration.npa_workflow.run_state import (
    RunManifest,
    RuntimeRunState,
    build_actionable_run_status,
    reconstruct_stage_job_attribution,
)


def _manifest(*, root_job: str = "8") -> RunManifest:
    return RunManifest(
        workflow="physical-ai-data-factory",
        run_id="paidf-stage-jobs",
        api_version="npa.workflow/v0.0.1",
        status="running",
        sky_job_id=root_job,
        steps=[
            {"state": "annotate", "status": "submitted"},
            {"state": "augment", "status": "submitted"},
            {"state": "curate", "status": "submitted"},
        ],
    )


def _wave(
    sequence: int,
    state: str,
    job_id: str,
    *,
    attempt: int = 1,
    status: str = "running",
    group: str = "serial",
) -> dict[str, object]:
    return {
        "key": f"{sequence:03d}|{group}|:{state}:-",
        "states": [state],
        "job_id": job_id,
        "job_name": f"paidf-{state}-attempt-{attempt}",
        "attempt": attempt,
        "status": status,
    }


def test_runtime_waves_prevent_job_eight_from_being_broadcast_to_every_stage() -> None:
    attribution = reconstruct_stage_job_attribution(
        _manifest(),
        runtime_waves=[
            _wave(1, "annotate", "11"),
            _wave(2, "augment", "12"),
            _wave(3, "curate", "13"),
        ],
    )

    assert {stage: info["managed_job_id"] for stage, info in attribution.items()} == {
        "annotate": "11",
        "augment": "12",
        "curate": "13",
    }
    assert all(
        "8" not in [attempt["job_id"] for attempt in info["attempts"]]
        for info in attribution.values()
    )


def test_mixed_stage_outcomes_use_each_final_job_observation() -> None:
    waves = [
        _wave(1, "annotate", "11", status="succeeded"),
        _wave(2, "augment", "12", status="failed"),
        _wave(3, "curate", "13", status="cancelled"),
    ]
    result = build_actionable_run_status(
        _manifest(),
        runtime_waves=waves,
        job_observations={
            "11": {"status": "SUCCEEDED", "task_rows": []},
            "12": {"status": "FAILED", "task_rows": []},
            "13": {"status": "CANCELLED", "task_rows": []},
        },
    )

    assert result["stages"]["annotate"]["state"] == "SUCCEEDED"
    assert result["stages"]["augment"]["state"] == "FAILED"
    assert result["stages"]["curate"]["state"] == "CANCELLED"
    assert result["status"] == "FAILED"
    assert result["stages"]["augment"]["outcome_provenance"] == (
        "scheduler_final_attempt"
    )


def test_retries_are_preserved_and_latest_attempt_is_deterministic() -> None:
    waves = [
        _wave(2, "augment", "21", attempt=1, status="failed"),
        _wave(2, "augment", "22", attempt=2, status="running"),
    ]
    manifest = RunManifest(
        workflow="physical-ai-data-factory",
        run_id="paidf-retry",
        api_version="npa.workflow/v0.0.1",
        steps=[{"state": "augment", "status": "submitted"}],
    )

    [info] = reconstruct_stage_job_attribution(manifest, runtime_waves=waves).values()

    assert [item["job_id"] for item in info["attempts"]] == ["21", "22"]
    assert info["active_attempt"] == 2
    assert info["managed_job_id"] == "22"


def test_record_wave_keeps_historical_attempts() -> None:
    runtime = RuntimeRunState(workflow="paidf", run_id="paidf-retry")
    first = _wave(2, "augment", "21", attempt=1, status="failed")
    second = _wave(2, "augment", "22", attempt=2, status="running")
    runtime.record_wave(first)
    runtime.record_wave(second)
    runtime.record_wave({**second, "status": "succeeded"})

    assert [
        (item["attempt"], item["job_id"], item["status"]) for item in runtime.waves
    ] == [
        (1, "21", "failed"),
        (2, "22", "succeeded"),
    ]


def test_parallel_jobgroup_is_shared_only_by_proven_wave_members() -> None:
    manifest = RunManifest(
        workflow="paidf",
        run_id="paidf-parallel",
        api_version="npa.workflow/v0.0.1",
        steps=[
            {"state": "evaluate-a", "status": "submitted", "group": "eval"},
            {"state": "evaluate-b", "status": "submitted", "group": "eval"},
            {"state": "curate", "status": "submitted"},
        ],
    )
    wave = {
        "key": "001|eval|eval:evaluate-a:-,eval:evaluate-b:-",
        "states": ["evaluate-a", "evaluate-b"],
        "kind": "parallel",
        "group": "eval",
        "job_id": "30",
        "attempt": 1,
        "status": "running",
    }

    attribution = reconstruct_stage_job_attribution(manifest, runtime_waves=[wave])

    assert attribution["evaluate-a"]["managed_job_id"] == "30"
    assert attribution["evaluate-b"]["managed_job_id"] == "30"
    assert attribution["curate"]["managed_job_id"] == ""
    assert attribution["curate"]["attribution"] == "unknown"


def test_missing_root_and_missing_stage_evidence_remain_unknown() -> None:
    attribution = reconstruct_stage_job_attribution(_manifest(root_job=""))

    assert all(item["managed_job_id"] == "" for item in attribution.values())
    assert all(item["attribution"] == "unknown" for item in attribution.values())


def test_partial_runtime_ledger_uses_explicit_stage_identity_not_copied_root() -> None:
    manifest = _manifest()
    manifest.steps[1]["job_id"] = "22"
    manifest.steps[2]["job_id"] = "8"

    attribution = reconstruct_stage_job_attribution(
        manifest,
        runtime_waves=[_wave(1, "annotate", "21")],
    )

    assert attribution["annotate"]["managed_job_id"] == "21"
    assert attribution["augment"]["managed_job_id"] == "22"
    assert attribution["augment"]["attribution"] == "manifest_step"
    assert attribution["curate"]["managed_job_id"] == ""
    assert attribution["curate"]["attribution"] == "unknown"


def test_runtime_terminal_state_is_used_when_scheduler_observation_is_missing() -> None:
    result = build_actionable_run_status(
        _manifest(root_job="8"),
        runtime_waves=[
            _wave(1, "annotate", "11", status="succeeded"),
            _wave(2, "augment", "12", status="failed"),
            _wave(3, "curate", "13", status="cancelled"),
        ],
    )

    assert result["stages"]["annotate"]["state"] == "SUCCEEDED"
    assert result["stages"]["augment"]["state"] == "FAILED"
    assert result["stages"]["curate"]["state"] == "CANCELLED"
    assert result["status"] == "FAILED"


def test_legacy_single_job_manifest_remains_compatible() -> None:
    attribution = reconstruct_stage_job_attribution(_manifest(root_job="8"))

    assert {item["managed_job_id"] for item in attribution.values()} == {"8"}
    assert {item["attribution"] for item in attribution.values()} == {
        "legacy_single_managed_job"
    }


def test_conflicting_final_attempt_ids_are_exposed_as_ambiguous() -> None:
    manifest = RunManifest(
        workflow="paidf",
        run_id="paidf-conflict",
        api_version="npa.workflow/v0.0.1",
        steps=[{"state": "curate", "status": "submitted"}],
    )
    waves = [
        _wave(1, "curate", "40", attempt=2),
        _wave(1, "curate", "41", attempt=2),
    ]

    [info] = reconstruct_stage_job_attribution(manifest, runtime_waves=waves).values()

    assert info["managed_job_id"] == ""
    assert info["attribution"] == "ambiguous"
