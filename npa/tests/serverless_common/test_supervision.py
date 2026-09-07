from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from npa.clients.serverless import EndpointNotFoundError, JobInfo
from npa.orchestration.npa_workflow.run_state import RunStateStore
from npa.orchestration.npa_workflow.supervisor import (
    AttemptIdentity,
    PreflightEvidence,
    ServerlessRecoverySpec,
    ServerlessSupervisorAdapter,
    SupervisorLedger,
)
from npa.serverless_common.supervision import (
    ServerlessSupervisionConfig,
    ServerlessSupervisionError,
    supervise_serverless_job,
)


class MemoryStore(RunStateStore):
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        super().__init__(
            bucket="unit-bucket",
            prefix="runs/serverless",
            reader=self._read,
            writer=self._write,
            artifact_lister=self._list,
        )

    def _read(self, _bucket: str, key: str) -> bytes:
        if key not in self.objects:
            raise FileNotFoundError(key)
        return self.objects[key]

    def _write(self, _bucket: str, key: str, body: bytes) -> None:
        self.objects[key] = body

    def _list(self, _bucket: str, prefix: str) -> list[str]:
        return [key for key in self.objects if key.startswith(prefix)]


class FakeServerlessClient:
    def __init__(self) -> None:
        self.jobs: dict[str, JobInfo] = {}

    def get_job(self, job_id_or_name: str, project_id: str) -> JobInfo:
        for job in self.jobs.values():
            if job_id_or_name in {job.id, job.name}:
                return job
        raise EndpointNotFoundError(
            "not found", project_id=project_id, endpoint_id=job_id_or_name
        )

    def cancel_job(self, job_id: str, project_id: str) -> JobInfo:
        job = self.get_job(job_id, project_id)
        cancelled = replace(job, status="cancelled")
        self.jobs[job.id] = cancelled
        return cancelled

    def create_job(self, **kwargs: Any) -> JobInfo:
        job = JobInfo(
            id=f"provider-{len(self.jobs) + 1}",
            name=str(kwargs["name"]),
            project_id=str(kwargs["project_id"]),
            status="queued",
            pending_reason="capacity unavailable",
        )
        self.jobs[job.id] = job
        return job


class CountingServerlessAdapter(ServerlessSupervisorAdapter):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.observe_calls = 0

    def observe(self, identity: AttemptIdentity):
        self.observe_calls += 1
        return super().observe(identity)


def test_shared_production_loop_observes_through_serverless_adapter() -> None:
    client = FakeServerlessClient()
    client.jobs["provider-1"] = JobInfo(
        id="provider-1",
        name="serverless-run",
        project_id="project-role",
        status="succeeded",
    )
    adapter = CountingServerlessAdapter(
        ServerlessRecoverySpec(
            project_id="project-role",
            image="registry.example/tool@sha256:" + "a" * 64,
            command="run",
            output_path="s3://unit-bucket/result/",
        ),
        client=client,
    )
    identity = AttemptIdentity(
        runtime="serverless",
        run_id="serverless-run",
        attempt=1,
        logical_attempt_id="serverless-run:1",
        provider_job_id="provider-1",
        provider_job_name="serverless-run",
        workflow_sha256="w" * 64,
        source_sha256="s" * 64,
        image_digest="i" * 64,
    )

    recovered, final = supervise_serverless_job(
        adapter=adapter,
        ledger=SupervisorLedger(MemoryStore()),
        identity=identity,
        config=ServerlessSupervisionConfig(
            expected_workflow_sha256="w" * 64,
            expected_source_sha256="s" * 64,
            expected_image_digest="i" * 64,
            declared_outputs=("s3://unit-bucket/result/summary.json",),
            preflight=PreflightEvidence(
                checks={
                    name: "pass" for name in PreflightEvidence.REQUIRED_RELAUNCH_CHECKS
                }
            ),
            poll_interval_seconds=0,
        ),
        output_checker=lambda _uri: True,
        sleeper=lambda _seconds: None,
    )

    assert recovered == identity
    assert final.status == "succeeded"
    assert adapter.observe_calls == 1


def test_shared_serverless_loop_exhausts_persistent_capacity_recovery() -> None:
    client = FakeServerlessClient()
    client.jobs["provider-1"] = JobInfo(
        id="provider-1",
        name="serverless-run",
        project_id="project-role",
        status="queued",
        pending_reason="capacity unavailable",
    )
    store = MemoryStore()
    adapter = CountingServerlessAdapter(
        ServerlessRecoverySpec(
            project_id="project-role",
            image="registry.example/tool@sha256:" + "a" * 64,
            command="run",
            output_path="s3://unit-bucket/result/",
        ),
        client=client,
    )
    identity = AttemptIdentity(
        runtime="serverless",
        run_id="serverless-run",
        attempt=1,
        logical_attempt_id="serverless-run:1",
        provider_job_id="provider-1",
        provider_job_name="serverless-run",
        workflow_sha256="w" * 64,
        source_sha256="s" * 64,
        image_digest="i" * 64,
    )
    config = ServerlessSupervisionConfig(
        expected_workflow_sha256="w" * 64,
        expected_source_sha256="s" * 64,
        expected_image_digest="i" * 64,
        declared_outputs=("s3://unit-bucket/result/summary.json",),
        max_infrastructure_recoveries=1,
        preflight=PreflightEvidence(
            checks={name: "pass" for name in PreflightEvidence.REQUIRED_RELAUNCH_CHECKS}
        ),
        poll_interval_seconds=0,
    )

    with pytest.raises(
        ServerlessSupervisionError, match="INFRASTRUCTURE_RECOVERY_EXHAUSTED"
    ):
        supervise_serverless_job(
            adapter=adapter,
            ledger=SupervisorLedger(store),
            identity=identity,
            config=config,
            output_checker=lambda _uri: False,
            sleeper=lambda _seconds: None,
        )

    assert adapter.observe_calls == 2
    assert sorted(job.name for job in client.jobs.values()) == [
        "serverless-run",
        "serverless-run-a2",
    ]
    assert client.jobs["provider-2"].status == "cancelled"
    assert SupervisorLedger(store).latest()["recovery"]["reason_code"] == (
        "INFRASTRUCTURE_RECOVERY_EXHAUSTED"
    )


def test_shared_serverless_success_requires_declared_s3_outputs() -> None:
    client = FakeServerlessClient()
    client.jobs["provider-1"] = JobInfo(
        id="provider-1",
        name="serverless-run",
        project_id="project-role",
        status="succeeded",
    )
    adapter = CountingServerlessAdapter(
        ServerlessRecoverySpec(project_id="project-role"), client=client
    )
    identity = AttemptIdentity(
        runtime="serverless",
        run_id="serverless-run",
        attempt=1,
        logical_attempt_id="serverless-run:1",
        provider_job_id="provider-1",
        provider_job_name="serverless-run",
        workflow_sha256="w" * 64,
        source_sha256="s" * 64,
        image_digest="i" * 64,
    )

    with pytest.raises(ServerlessSupervisionError, match="DECLARED_OUTPUT_MISSING"):
        supervise_serverless_job(
            adapter=adapter,
            ledger=SupervisorLedger(MemoryStore()),
            identity=identity,
            config=ServerlessSupervisionConfig(
                expected_workflow_sha256="w" * 64,
                expected_source_sha256="s" * 64,
                expected_image_digest="i" * 64,
                declared_outputs=("s3://unit-bucket/result/summary.json",),
            ),
            output_checker=lambda _uri: False,
            sleeper=lambda _seconds: None,
        )
