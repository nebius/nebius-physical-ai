"""Repeat-safe cancellation across runtime-orchestrated workflow waves."""

from __future__ import annotations

from npa.orchestration.npa_workflow.cancellation import assess_run_cancellation
from npa.orchestration.npa_workflow.run_resolution import RunResolution
from npa.orchestration.skypilot import cleanup as cleanup_module
from npa.orchestration.skypilot.cleanup import CleanupResult
from npa.orchestration.skypilot.workflow import ManagedJobEvidence


def _resolution(runtime: dict, *, manifest: dict | None = None) -> RunResolution:
    return RunResolution(
        run_id="paidf-runtime",
        project="prod",
        found=True,
        source="canonical_paidf_s3_prefix",
        manifest=manifest
        or {
            "schema_version": "npa.workflow.run.v1",
            "run_id": "paidf-runtime",
            "status": "submitted",
        },
        runtime_state=runtime,
    )


def test_terminal_multistage_run_without_root_job_id_is_an_explicit_noop() -> None:
    resolution = _resolution(
        {
            "status": "succeeded",
            "waves": [
                {"key": "annotate", "job_id": "101", "status": "succeeded"},
                {"key": "curate", "job_id": "102", "status": "succeeded"},
            ],
        }
    )

    assessment = assess_run_cancellation(
        resolution,
        lookup=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError(
                "terminal durable waves need no provider cancellation lookup"
            )
        ),
    )

    assert assessment.detected_state == "SUCCEEDED"
    assert assessment.no_cancellation_needed
    assert assessment.active_jobs == []
    assert [job.job_id for job in assessment.terminal_jobs] == ["101", "102"]


def test_active_multistage_assessment_targets_every_nonterminal_job_once() -> None:
    resolution = _resolution(
        {
            "status": "running",
            "waves": [
                {
                    "key": "augment",
                    "job_id": "201",
                    "job_name": "paidf-runtime-wave-1",
                    "status": "running",
                },
                {
                    "key": "augment-retry",
                    "job_id": "201",
                    "job_name": "paidf-runtime-wave-1",
                    "status": "retrying",
                },
                {"key": "evaluate", "job_id": "202", "status": "failed"},
                {
                    "key": "curate",
                    "job_id": "203",
                    "job_name": "paidf-runtime-wave-3",
                    "status": "submitted",
                },
            ],
        }
    )
    looked_up: list[tuple[str, str]] = []

    def lookup(job_name: str, *, job_id: str, **_kwargs) -> ManagedJobEvidence:
        looked_up.append((job_name, job_id))
        return ManagedJobEvidence("found", job_id=job_id, status="RUNNING")

    assessment = assess_run_cancellation(resolution, lookup=lookup)

    assert assessment.detected_state == "ACTIVE"
    assert [job.job_id for job in assessment.active_jobs] == ["201", "203"]
    assert [job.job_id for job in assessment.terminal_jobs] == ["202"]
    assert looked_up == [
        ("paidf-runtime-wave-1", "201"),
        ("paidf-runtime-wave-3", "203"),
    ]


def test_legacy_single_job_manifest_remains_compatible() -> None:
    resolution = _resolution(
        {},
        manifest={
            "run_id": "paidf-runtime",
            "workflow_name": "legacy",
            "sky_job_id": "250",
            "status": "running",
            "stages": {
                "prepare": {"status": "succeeded"},
                "train": {"status": "running"},
            },
        },
    )

    assessment = assess_run_cancellation(
        resolution,
        lookup=lambda *args, **kwargs: ManagedJobEvidence(
            "found", job_id="250", status="RUNNING"
        ),
    )

    assert assessment.detected_state == "ACTIVE"
    assert assessment.errors == []
    assert [job.job_id for job in assessment.active_jobs] == ["250"]


def test_active_stage_outweighs_a_stale_terminal_root_state() -> None:
    resolution = _resolution(
        {
            "status": "succeeded",
            "waves": [
                {
                    "key": "late-retry",
                    "job_id": "251",
                    "status": "running",
                }
            ],
        },
        manifest={"run_id": "paidf-runtime", "status": "succeeded"},
    )

    assessment = assess_run_cancellation(
        resolution,
        lookup=lambda *args, **kwargs: ManagedJobEvidence(
            "found", job_id="251", status="RUNNING"
        ),
    )

    assert assessment.detected_state == "ACTIVE"
    assert [job.job_id for job in assessment.active_jobs] == ["251"]


def test_runtime_wave_identity_ignores_a_bogus_discovered_root_job_eight() -> None:
    resolution = _resolution(
        {
            "status": "running",
            "waves": [
                {
                    "key": "001|serial|:annotate:-",
                    "states": ["annotate"],
                    "job_id": "601",
                    "job_name": "paidf-runtime-annotate",
                    "status": "running",
                },
                {
                    "key": "002|serial|:curate:-",
                    "states": ["curate"],
                    "job_id": "602",
                    "job_name": "paidf-runtime-curate",
                    "status": "succeeded",
                },
            ],
        },
        manifest={
            "schema_version": "npa.workflow.run.v1",
            "run_id": "paidf-runtime",
            "sky_job_id": "8",
            "status": "running",
            "steps": [
                {"state": "annotate", "status": "running"},
                {"state": "curate", "status": "succeeded"},
            ],
            # The historical bug copied the latest discovered root ID into every
            # stage. Runtime waves are the durable attempt identities and must win.
            "stages": {
                "annotate": {"status": "running", "sky_job_id": "8"},
                "curate": {"status": "succeeded", "sky_job_id": "8"},
            },
        },
    )
    resolution.job_id = "8"
    resolution.job_name = "bogus-latest-discovery"
    looked_up: list[str] = []

    def lookup(job_name: str, *, job_id: str, **_kwargs) -> ManagedJobEvidence:
        looked_up.append(job_id)
        return ManagedJobEvidence("found", job_id=job_id, status="RUNNING")

    assessment = assess_run_cancellation(resolution, lookup=lookup)

    assert [job.job_id for job in assessment.jobs] == ["601", "602"]
    assert [job.job_id for job in assessment.active_jobs] == ["601"]
    assert looked_up == ["601"]


def test_partial_runtime_ledger_keeps_a_distinct_explicit_stage_job() -> None:
    resolution = _resolution(
        {
            "status": "running",
            "waves": [
                {
                    "key": "001|serial|:annotate:-",
                    "states": ["annotate"],
                    "job_id": "701",
                    "status": "succeeded",
                }
            ],
        },
        manifest={
            "schema_version": "npa.workflow.run.v1",
            "run_id": "paidf-partial-ledger",
            "sky_job_id": "8",
            "status": "running",
            "steps": [
                {"state": "annotate", "status": "succeeded", "job_id": "8"},
                {"state": "augment", "status": "running", "job_id": "702"},
            ],
        },
    )
    looked_up: list[str] = []

    def lookup(job_name: str, *, job_id: str, **_kwargs) -> ManagedJobEvidence:
        looked_up.append(job_id)
        return ManagedJobEvidence("found", job_id=job_id, status="RUNNING")

    assessment = assess_run_cancellation(resolution, lookup=lookup)

    assert [job.job_id for job in assessment.jobs] == ["701", "702"]
    assert [job.job_id for job in assessment.active_jobs] == ["702"]
    assert looked_up == ["702"]


def test_terminal_not_found_cancel_race_is_successful_convergence(monkeypatch) -> None:
    monkeypatch.setattr(
        cleanup_module,
        "_cancel_job",
        lambda *args, **kwargs: CleanupResult(errors=["job became terminal"]),
    )
    outcomes = iter(["absent", "absent"])
    monkeypatch.setattr(
        cleanup_module,
        "_verify_managed_job_convergence",
        lambda *args, **kwargs: next(outcomes),
    )
    monkeypatch.setattr(
        cleanup_module,
        "wait_for_jobs_terminal",
        lambda *args, **kwargs: (True, []),
    )
    monkeypatch.setattr(
        cleanup_module,
        "sky_down",
        lambda *args, **kwargs: CleanupResult(
            resources_removed=["paidf-runtime-wave-1:already-absent"]
        ),
    )

    result = cleanup_module.cleanup_launched_workflow(
        "301", "paidf-runtime", job_name="paidf-runtime-wave-1"
    )

    assert result.ok
    assert "job:301:already-absent" in result.resources_removed
    assert not result.errors


def test_auth_malformed_and_ambiguous_records_remain_failures() -> None:
    malformed = _resolution(
        {
            "status": "running",
            "waves": [
                "not-a-record",
                {"key": "annotate", "job_id": "401", "status": "running"},
            ],
        }
    )

    assessment = assess_run_cancellation(
        malformed,
        lookup=lambda *args, **kwargs: ManagedJobEvidence(
            "unavailable",
            error="Unauthorized while resolving ambiguous exact managed-job records",
        ),
    )

    assert assessment.detected_state == "VERIFICATION_UNAVAILABLE"
    assert assessment.active_jobs == []
    assert any("runtime wave 0 is malformed" in error for error in assessment.errors)
    assert any("Unauthorized" in error for error in assessment.errors)


def test_partial_multijob_cleanup_aggregates_failure_and_preserves_cluster(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def cleanup(job_id: str, *args, **kwargs) -> CleanupResult:
        calls.append(job_id)
        if job_id == "502":
            return CleanupResult(errors=["PermissionDenied cancelling exact job 502"])
        return CleanupResult(resources_removed=[f"job:{job_id}"])

    monkeypatch.setattr(cleanup_module, "cleanup_launched_workflow", cleanup)
    monkeypatch.setattr(
        cleanup_module,
        "sky_down",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError(
                "the shared cluster must remain while any exact job is unverified"
            )
        ),
    )

    result = cleanup_module.cleanup_launched_workflows(
        [("501", "wave-1"), ("502", "wave-2"), ("501", "wave-1")],
        "paidf-runtime",
    )

    assert calls == ["501", "502"]
    assert result.resources_removed == ["job:501"]
    assert result.errors == ["PermissionDenied cancelling exact job 502"]
