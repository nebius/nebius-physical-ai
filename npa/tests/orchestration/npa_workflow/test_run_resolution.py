from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.clients.config import StorageConfig
from npa.orchestration.npa_workflow.run_resolution import resolve_run
from npa.orchestration.npa_workflow.submission_state import update_submission_state
from npa.orchestration.skypilot.workflow import ManagedJobEvidence
from npa.orchestration.skypilot.workflow_state import WorkflowStateError


class ExactS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.queries: list[tuple[str, str, dict | None]] = []
        self.failure: Exception | None = None

    def get_object(self, *, Bucket: str, Key: str):  # noqa: ANN201, N803
        if self.failure is not None:
            raise self.failure
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)
        return {"Body": BytesIO(self.objects[(Bucket, Key)])}

    def get_paginator(self, name: str):  # noqa: ANN201
        assert name == "list_objects_v2"
        parent = self

        class Paginator:
            def paginate(
                self,
                *,
                Bucket: str,  # noqa: N803
                Prefix: str,  # noqa: N803
                PaginationConfig: dict | None = None,  # noqa: N803
            ):
                parent.queries.append((Bucket, Prefix, PaginationConfig))
                if parent.failure is not None:
                    raise parent.failure
                keys = sorted(
                    key
                    for bucket, key in parent.objects
                    if bucket == Bucket and key.startswith(Prefix)
                )
                maximum = int((PaginationConfig or {}).get("MaxItems") or len(keys))
                yield {"Contents": [{"Key": key} for key in keys[:maximum]]}

        return Paginator()

    def put_json(self, bucket: str, key: str, payload: dict) -> None:
        self.objects[(bucket, key)] = json.dumps(payload).encode()


@pytest.fixture
def resolver_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ExactS3:
    monkeypatch.setenv("HOME", str(tmp_path))
    storage = ExactS3()
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow_state.boto3.client",
        lambda *args, **kwargs: storage,
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow_state.resolve_project_storage",
        lambda project=None: StorageConfig(
            checkpoint_bucket="s3://alias-bucket/checkpoints/",
            endpoint_url="https://storage.alias.invalid",
            aws_access_key_id="fixture-access",
            aws_secret_access_key="fixture-secret",
        ),
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow_state.load_credentials",
        lambda: SimpleNamespace(
            s3_access_key_id="",
            s3_secret_access_key="",
            s3_endpoint="",
            s3_bucket="",
        ),
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.run_resolution.lookup_managed_job",
        lambda *args, **kwargs: ManagedJobEvidence("absent"),
    )
    return storage


def _manifest(run_id: str, *, workflow: str = "physical-ai-data-factory") -> dict:
    return {
        "schema_version": "npa.workflow.run.v1",
        "workflow": workflow,
        "run_id": run_id,
        "api_version": "npa.workflow/v0.0.1",
        "run_prefix_uri": f"s3://alias-bucket/{workflow}/{run_id}",
        "status": "submitted",
        "sky_job_id": "17",
        "steps": [{"state": "stage-one", "status": "submitted"}],
    }


def test_exact_paidf_run_found_from_durable_receipt(resolver_env: ExactS3) -> None:
    run_id = "receipt-run"
    update_submission_state(
        "paidf",
        run_id,
        {
            "workflow": {
                "name": "physical-ai-data-factory",
                "api_version": "npa.workflow/v0.0.1",
                "run_prefix_uri": f"s3://alias-bucket/physical-ai-data-factory/{run_id}",
                "steps": [{"state": "curate", "status": "submitted"}],
            },
            "launch": {"status": "submitted", "sky_job_id": "17"},
        },
    )

    resolved = resolve_run(run_id, project="paidf")

    assert resolved.found is True
    assert resolved.source == "durable_submission_receipt"
    assert resolved.job_id == "17"
    assert resolved.manifest_pending is True
    assert resolved.manifest_uri.endswith(
        f"physical-ai-data-factory/{run_id}/npa-workflow/manifest.json"
    )


def test_receipt_proven_run_stays_found_when_live_verification_is_unavailable(
    resolver_env: ExactS3, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "receipt-provider-down"
    update_submission_state(
        "paidf",
        run_id,
        {
            "workflow": {
                "name": "physical-ai-data-factory",
                "run_prefix_uri": f"s3://alias-bucket/physical-ai-data-factory/{run_id}",
            },
            "launch": {"status": "submitted", "sky_job_id": "18"},
        },
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.run_resolution.lookup_managed_job",
        lambda *args, **kwargs: ManagedJobEvidence(
            "unavailable", error="fixture provider unavailable"
        ),
    )

    resolved = resolve_run(run_id, project="paidf")

    assert resolved.found is True
    assert resolved.source == "durable_submission_receipt"
    assert resolved.manifest_pending is True
    assert resolved.verification_unavailable is True
    assert resolved.conclusively_absent is False


def test_legacy_partial_canonical_prefix_is_a_found_run(
    resolver_env: ExactS3,
) -> None:
    run_id = "legacy-in-flight"
    key = f"physical-ai-data-factory/{run_id}/cosmos_augmented/part-000.mp4"
    resolver_env.objects[("alias-bucket", key)] = b"partial"

    resolved = resolve_run(run_id, project="paidf")

    assert resolved.found is True
    assert resolved.source == "canonical_paidf_s3_prefix"
    assert resolved.manifest_pending is True
    check = next(
        item for item in resolved.checks if item.source == "canonical_paidf_s3_prefix"
    )
    assert check.outcome == "found"
    assert resolver_env.queries == [
        (
            "alias-bucket",
            f"physical-ai-data-factory/{run_id}/",
            {"MaxItems": 1, "PageSize": 1},
        )
    ]


def test_managed_job_evidence_finds_run_while_manifest_is_pending(
    resolver_env: ExactS3, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.run_resolution.lookup_managed_job",
        lambda *args, **kwargs: ManagedJobEvidence(
            "found",
            job_id="91",
            status="RUNNING",
            task_rows=({"task_id": 0, "task_name": "annotate", "status": "RUNNING"},),
        ),
    )

    resolved = resolve_run("managed-only", project="paidf")

    assert resolved.found is True
    assert resolved.source == "managed_job"
    assert resolved.job_id == "91"
    assert resolved.manifest_pending is True


def test_runtime_ledger_recovers_exact_active_wave_identity(
    resolver_env: ExactS3, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "runtime-in-flight"
    runtime_key = f"physical-ai-data-factory/{run_id}/npa-workflow/runtime.json"
    resolver_env.put_json(
        "alias-bucket",
        runtime_key,
        {
            "schema_version": "npa.workflow.runtime.v1",
            "workflow": "physical-ai-data-factory",
            "run_id": run_id,
            "status": "running",
            "waves": [
                {
                    "key": "wave-01",
                    "states": ["annotate"],
                    "status": "succeeded",
                    "job_id": "40",
                    "job_name": f"{run_id}-01-annotate",
                },
                {
                    "key": "wave-02",
                    "states": ["curate"],
                    "status": "running",
                    "job_id": "41",
                    "job_name": f"{run_id}-02-curate",
                    "tasks": [
                        {"task_id": 0, "task_name": "curate", "status": "PENDING"}
                    ],
                },
            ],
        },
    )
    lookups: list[tuple[str, str]] = []

    def lookup(name: str, *, job_id: str = "", **kwargs):  # noqa: ANN003, ANN202
        lookups.append((name, job_id))
        return ManagedJobEvidence(
            "found",
            job_id="41",
            status="RUNNING",
            task_rows=({"task_id": 0, "task_name": "curate", "status": "PENDING"},),
        )

    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.run_resolution.lookup_managed_job", lookup
    )

    resolved = resolve_run(run_id, project="paidf")

    assert resolved.source == "canonical_paidf_s3_prefix"
    assert resolved.job_id == "41"
    assert resolved.job_name == f"{run_id}-02-curate"
    assert resolved.runtime_state["status"] == "running"
    assert lookups == [(f"{run_id}-02-curate", "41")]


def test_project_alias_selects_bucket_endpoint_and_canonical_nested_prefix(
    resolver_env: ExactS3, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected: list[str | None] = []

    def storage(project=None):  # noqa: ANN001, ANN202
        selected.append(project)
        return StorageConfig(
            checkpoint_bucket="s3://profile-bucket/project-checkpoints/",
            endpoint_url="https://profile.storage.invalid",
            aws_access_key_id="fixture-access",
            aws_secret_access_key="fixture-secret",
        )

    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow_state.resolve_project_storage", storage
    )
    run_id = "alias-run"
    key = f"physical-ai-data-factory/{run_id}/npa-workflow/manifest.json"
    resolver_env.put_json("profile-bucket", key, _manifest(run_id))

    resolved = resolve_run(run_id, project="paidf-profile")

    assert selected and set(selected) == {"paidf-profile"}
    assert resolved.state is not None
    assert resolved.state.bucket == "profile-bucket"
    assert resolved.state.endpoint_url == "https://profile.storage.invalid"
    assert resolved.state.prefix == key.removesuffix("/manifest.json")


def test_explicit_workflow_uri_wins_over_receipt(resolver_env: ExactS3) -> None:
    run_id = "explicit-wins"
    update_submission_state(
        "paidf",
        run_id,
        {
            "workflow": {
                "name": "physical-ai-data-factory",
                "run_prefix_uri": f"s3://alias-bucket/physical-ai-data-factory/{run_id}",
            },
            "launch": {"status": "submitted", "sky_job_id": "17"},
        },
    )
    explicit = (
        f"s3://alias-bucket/archive/physical-ai-data-factory/{run_id}/npa-workflow"
    )
    resolver_env.put_json(
        "alias-bucket",
        f"archive/physical-ai-data-factory/{run_id}/npa-workflow/manifest.json",
        _manifest(run_id),
    )

    resolved = resolve_run(run_id, project="paidf", workflow_s3_uri=explicit)

    assert resolved.source == "explicit_workflow_s3_uri"
    assert resolved.state is not None
    assert resolved.state.uri == explicit
    assert [item.source for item in resolved.checks] == ["explicit_workflow_s3_uri"]


def test_provider_and_skypilot_unavailable_is_not_absence(
    resolver_env: ExactS3, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolver_env.failure = PermissionError("fixture access denied")
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.run_resolution.lookup_managed_job",
        lambda *args, **kwargs: ManagedJobEvidence(
            "unavailable", error="fixture SkyPilot auth unavailable"
        ),
    )

    resolved = resolve_run("unavailable-run", project="paidf")

    assert resolved.found is False
    assert resolved.verification_unavailable is True
    assert resolved.conclusively_absent is False
    assert {item.outcome for item in resolved.checks} >= {"unavailable"}


def test_truly_absent_run_reports_every_checked_source(resolver_env: ExactS3) -> None:
    resolved = resolve_run("truly-absent", project="paidf")

    assert resolved.found is False
    assert resolved.conclusively_absent is True
    assert [item.source for item in resolved.checks] == [
        "explicit_workflow_s3_uri",
        "durable_submission_receipt",
        "canonical_paidf_s3_prefix",
        "managed_job",
        "ordinary_workflow",
    ]
    assert all(item.outcome in {"absent", "not_supplied"} for item in resolved.checks)


@pytest.mark.parametrize("run_id", ["../escape", "nested/run", ".", "..", "bad%2Frun"])
def test_traversal_run_ids_are_rejected_before_provider_access(
    resolver_env: ExactS3, run_id: str
) -> None:
    with pytest.raises(WorkflowStateError, match="path component"):
        resolve_run(run_id, project="paidf")
    assert resolver_env.queries == []


def test_traversal_in_explicit_s3_uri_is_rejected_before_provider_access(
    resolver_env: ExactS3,
) -> None:
    with pytest.raises(WorkflowStateError, match="unsafe"):
        resolve_run(
            "escape",
            project="paidf",
            workflow_s3_uri=(
                "s3://alias-bucket/physical-ai-data-factory/../escape/npa-workflow"
            ),
        )
    assert resolver_env.queries == []


def test_unrelated_nested_keys_never_become_the_requested_run(
    resolver_env: ExactS3,
) -> None:
    resolver_env.objects[
        (
            "alias-bucket",
            "physical-ai-data-factory/wanted-other/nested/wanted/file.json",
        )
    ] = b"{}"

    resolved = resolve_run("wanted", project="paidf")

    assert resolved.conclusively_absent is True
    assert resolver_env.queries[0][1] == "physical-ai-data-factory/wanted/"


def test_ordinary_exact_workflow_contract_remains_supported(
    resolver_env: ExactS3,
) -> None:
    run_id = "ordinary-run"
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "workflow_name": "ordinary",
        "sky_job_id": "4",
        "stages": {"train": {"name": "train"}},
    }
    resolver_env.put_json(
        "alias-bucket", f"checkpoints/{run_id}/manifest.json", manifest
    )

    resolved = resolve_run(run_id, project="paidf")

    assert resolved.found is True
    assert resolved.source == "ordinary_workflow"
    assert resolved.manifest == manifest


def test_stale_planned_receipt_cannot_override_terminal_s3_manifest(
    resolver_env: ExactS3,
) -> None:
    run_id = "completed-ten-wave"
    update_submission_state(
        "paidf",
        run_id,
        {
            "launch_state": "planned",
            "workflow": {"name": "physical-ai-data-factory"},
            "planning": {"state": "durable"},
        },
    )
    manifest = _manifest(run_id)
    manifest["status"] = "SUCCEEDED"
    manifest["sky_job_id"] = ""
    manifest["steps"] = [
        {"state": f"wave-{index:02d}", "status": "SUCCEEDED"}
        for index in range(1, 11)
    ]
    resolver_env.put_json(
        "alias-bucket",
        f"physical-ai-data-factory/{run_id}/npa-workflow/manifest.json",
        manifest,
    )
    for index in range(40):
        resolver_env.objects[
            (
                "alias-bucket",
                f"physical-ai-data-factory/{run_id}/reports/artifact-{index:02d}.json",
            )
        ] = b"{}"

    resolved = resolve_run(
        run_id, project="paidf", allow_local_not_submitted=True
    )
    assert resolved.found is True
    assert resolved.not_submitted is False
    assert resolved.manifest is not None
    assert resolved.manifest["status"] == "SUCCEEDED"


def test_planned_receipt_is_not_not_submitted_when_later_evidence_unavailable(
    resolver_env: ExactS3,
) -> None:
    run_id = "planned-but-storage-unavailable"
    update_submission_state(
        "paidf",
        run_id,
        {
            "launch_state": "planned",
            "workflow": {"name": "physical-ai-data-factory"},
            "planning": {"state": "durable"},
        },
    )
    resolver_env.failure = PermissionError("eventual consistency / auth outage")

    resolved = resolve_run(
        run_id, project="paidf", allow_local_not_submitted=True
    )
    assert resolved.not_submitted is False
    assert resolved.verification_unavailable is True
