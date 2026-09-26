from __future__ import annotations

import json

from botocore.exceptions import ClientError
import pytest

from npa.orchestration.npa_workflow.run_state import (
    RunManifest,
    RunStateStore,
    is_paidf_input_workflow_name,
    reconcile_submitted_manifest,
)


@pytest.mark.parametrize(
    "name",
    [
        "physical-ai-data-factory",
        "paidf-cosmos3",
        "nvidia-paidf-vda-cosmos-transfer25",
    ],
)
def test_submit_recognizes_every_paidf_input_workflow(name: str) -> None:
    assert is_paidf_input_workflow_name(name)


def test_run_state_store_roundtrip() -> None:
    store: dict[tuple[str, str], bytes] = {}

    def writer(bucket: str, key: str, body: bytes) -> None:
        store[(bucket, key)] = body

    def reader(bucket: str, key: str) -> str:
        return store[(bucket, key)].decode("utf-8")

    state_store = RunStateStore(
        bucket="bucket",
        prefix="runs/demo",
        reader=reader,
        writer=writer,
    )
    manifest = RunManifest(
        workflow="demo",
        run_id="demo-1",
        api_version="npa.workflow/v0.0.1",
        status="running",
    )
    state_store.write_manifest(manifest)
    state_store.append_step(manifest, {"state": "augment", "status": "ok"})
    loaded = state_store.read_manifest()
    assert loaded is not None
    assert loaded.run_id == "demo-1"
    assert loaded.steps[0]["state"] == "augment"
    status_payload = json.loads(store[("bucket", "runs/demo/npa-workflow/status.json")])
    assert status_payload["status"] == "running"


def test_submitted_manifest_reconciles_partial_failure_and_keeps_resources() -> None:
    manifest = RunManifest(
        workflow="paidf",
        run_id="run-1",
        api_version="npa.workflow/v0.0.1",
        status="submitted",
        sky_job_id="4",
        steps=[
            {"state": "annotate", "status": "submitted", "resources_profile": {}},
            {
                "state": "augment",
                "status": "submitted",
                "resources_profile": {
                    "accelerators": "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
                },
            },
            {"state": "evaluate", "status": "submitted", "resources_profile": {}},
        ],
    )

    reconcile_submitted_manifest(
        manifest,
        live_status="FAILED",
        task_rows=(
            {"task_id": 0, "task_name": "annotate", "status": "SUCCEEDED"},
            {"task_id": 1, "task_name": "augment", "status": "FAILED"},
            {"task_id": 2, "task_name": "evaluate", "status": "PENDING"},
        ),
    )

    assert manifest.status == "FAILED"
    assert [step["status"] for step in manifest.steps] == [
        "SUCCEEDED",
        "FAILED",
        "PENDING",
    ]
    assert manifest.steps[1]["resources_profile"]["accelerators"].startswith("RTXPRO-")
    assert manifest.sky_job_id == "4"


def test_terminal_success_reconciles_every_stage_without_queue_rows() -> None:
    manifest = RunManifest(
        workflow="paidf",
        run_id="run-success",
        api_version="npa.workflow/v0.0.1",
        status="submitted",
        sky_job_id="7",
        steps=[
            {"state": "annotate", "status": "submitted", "artifact": "s3://b/a"},
            {"state": "augment", "status": "running", "artifact": "s3://b/v"},
        ],
    )

    reconcile_submitted_manifest(manifest, live_status="SUCCEEDED", task_rows=())

    assert manifest.status == "SUCCEEDED"
    assert [step["status"] for step in manifest.steps] == ["SUCCEEDED", "SUCCEEDED"]
    assert [step["artifact"] for step in manifest.steps] == ["s3://b/a", "s3://b/v"]


def test_runtime_run_state_roundtrip_is_separate_from_the_manifest() -> None:
    """The runtime ledger is an additional document; RunManifest is untouched."""

    from npa.orchestration.npa_workflow.run_state import (
        RUNTIME_SCHEMA_VERSION,
        RuntimeRunState,
        runtime_key,
    )

    store: dict[tuple[str, str], bytes] = {}

    def reader(bucket: str, key: str) -> str:
        # Contract of the reader seam (and of the real S3 path): a missing object
        # raises FileNotFoundError. Anything else must propagate, so a transient
        # storage error can never be mistaken for "no ledger" and silently make
        # --resume resubmit every wave.
        try:
            return store[(bucket, key)].decode("utf-8")
        except KeyError as exc:
            raise FileNotFoundError(f"s3://{bucket}/{key}") from exc

    state_store = RunStateStore(
        bucket="bucket",
        prefix="runs/demo",
        reader=reader,
        writer=lambda bucket, key, body: store.__setitem__((bucket, key), body),
    )

    assert state_store.read_runtime_state() is None  # nothing written yet

    runtime_state = RuntimeRunState(
        workflow="demo", run_id="demo-1", api_version="npa.workflow/v0.0.1"
    )
    runtime_state.record_wave(
        {"key": "001|serial|:a:-", "status": "running", "job_id": "7"}
    )
    state_store.write_runtime_state(runtime_state)
    # Same key updated in place, not appended twice.
    runtime_state.record_wave(
        {"key": "001|serial|:a:-", "status": "succeeded", "job_id": "7"}
    )
    runtime_state.decisions.append({"decision": "promote_checkpoint"})
    runtime_state.watermarks["ingest"] = {"objects": 2}
    state_store.write_runtime_state(runtime_state)

    loaded = state_store.read_runtime_state()
    assert loaded is not None
    assert loaded.schema_version == RUNTIME_SCHEMA_VERSION
    assert loaded.run_prefix_uri == "s3://bucket/runs/demo"
    assert [wave["status"] for wave in loaded.waves] == ["succeeded"]
    assert loaded.completed_wave("001|serial|:a:-") is not None
    assert loaded.completed_wave("002|serial|:b:-") is None
    assert loaded.decisions[0]["decision"] == "promote_checkpoint"
    assert loaded.watermarks["ingest"]["objects"] == 2
    # Written next to, not instead of, the run manifest.
    assert ("bucket", runtime_key("runs/demo")) in store
    assert ("bucket", "runs/demo/npa-workflow/manifest.json") not in store


def test_block_relaunch_wave_remains_in_flight_for_exact_reconciliation() -> None:
    from npa.orchestration.npa_workflow.run_state import RuntimeRunState

    state = RuntimeRunState(workflow="demo", run_id="run-1")
    record = {
        "key": "001|serial|:train:-",
        "attempt": 2,
        "status": "failed",
        "sky_status": "PENDING",
        "job_id": "job-2",
        "job_name": "run-1-01-train-a2",
        "recovery_decision": "block_relaunch",
    }
    state.record_wave(record)

    assert state.in_flight_wave(record["key"]) == record


def test_verified_terminal_block_relaunch_wave_is_not_in_flight() -> None:
    from npa.orchestration.npa_workflow.run_state import RuntimeRunState

    state = RuntimeRunState(workflow="demo", run_id="run-1")
    record = {
        "key": "001|serial|:train:-",
        "attempt": 2,
        "status": "failed",
        "sky_status": "CANCELLED",
        "job_id": "job-2",
        "job_name": "run-1-01-train-a2",
        "recovery_decision": "block_relaunch",
        "cancellation": {"state": "verified", "error": ""},
    }
    state.record_wave(record)

    assert state.in_flight_wave(record["key"]) is None


@pytest.mark.parametrize("sky_status", ["PENDING", "RUNNING", "", None])
def test_verified_cancellation_without_terminality_remains_in_flight(
    sky_status: str | None,
) -> None:
    from npa.orchestration.npa_workflow.run_state import RuntimeRunState

    state = RuntimeRunState(workflow="demo", run_id="run-1")
    record = {
        "key": "001|serial|:train:-",
        "status": "failed",
        "sky_status": sky_status,
        "job_id": "job-2",
        "recovery_decision": "block_relaunch",
        "cancellation": {"state": "verified"},
    }
    state.record_wave(record)
    assert state.in_flight_wave(record["key"]) == record


@pytest.mark.parametrize("launch_sequence", [False, 0.0, "0", None, [], {}])
def test_malformed_launch_sequence_does_not_prove_prelaunch_absence(
    launch_sequence: object,
) -> None:
    from npa.orchestration.npa_workflow.run_state import RuntimeRunState

    state = RuntimeRunState(workflow="demo", run_id="run-1")
    record = {
        "key": "001|serial|:train:-",
        "status": "failed",
        "job_id": "",
        "launch_sequence": launch_sequence,
        "recovery_decision": "future_preflight_decision",
    }
    state.record_wave(record)
    assert state.in_flight_wave(record["key"]) == record


def test_missing_launch_sequence_does_not_prove_prelaunch_absence() -> None:
    from npa.orchestration.npa_workflow.run_state import RuntimeRunState

    state = RuntimeRunState(workflow="demo", run_id="run-1")
    record = {
        "key": "001|serial|:train:-",
        "status": "failed",
        "job_id": "",
        "recovery_decision": "future_preflight_decision",
    }
    state.record_wave(record)
    assert state.in_flight_wave(record["key"]) == record


def test_unknown_recovery_decision_remains_in_flight() -> None:
    from npa.orchestration.npa_workflow.run_state import RuntimeRunState

    state = RuntimeRunState(workflow="demo", run_id="run-1")
    record = {
        "key": "001|serial|:train:-",
        "attempt": 2,
        "status": "failed",
        "sky_status": "PENDING",
        "job_id": "job-2",
        "job_name": "run-1-01-train-a2",
        "recovery_decision": "block_awaiting_quota_v2",
    }
    state.record_wave(record)

    assert state.in_flight_wave(record["key"]) == record


@pytest.mark.parametrize("sky_status", ["SUCCEEDED", "FAILED_CONTROLLER", "STOPPED"])
def test_unknown_recovery_decision_respects_terminal_provider_evidence(
    sky_status: str,
) -> None:
    from npa.orchestration.npa_workflow.run_state import RuntimeRunState

    state = RuntimeRunState(workflow="demo", run_id="run-1")
    state.record_wave(
        {
            "key": "001|serial|:train:-",
            "attempt": 2,
            "status": "failed",
            "sky_status": sky_status,
            "job_id": "job-2",
            "job_name": "run-1-01-train-a2",
            "recovery_decision": "future_terminal_decision",
        }
    )

    assert state.in_flight_wave("001|serial|:train:-") is None


def test_unknown_prelaunch_decision_without_launch_identity_is_resolved() -> None:
    from npa.orchestration.npa_workflow.run_state import RuntimeRunState

    state = RuntimeRunState(workflow="demo", run_id="run-1")
    state.record_wave(
        {
            "key": "001|serial|:train:-",
            "attempt": 1,
            "status": "failed",
            "sky_status": "",
            "job_id": "",
            "launch_sequence": 0,
            "recovery_decision": "future_preflight_decision",
        }
    )

    assert state.in_flight_wave("001|serial|:train:-") is None


@pytest.mark.parametrize(
    "recovery", ["reuse_completed_wave", "block_output_reuse_evidence"]
)
@pytest.mark.parametrize("sky_status", ["PENDING", "CANCELLED"])
def test_output_reuse_recovery_remains_in_flight(recovery, sky_status) -> None:
    from npa.orchestration.npa_workflow.run_state import RuntimeRunState

    state = RuntimeRunState(workflow="demo", run_id="run-1")
    state.record_wave(
        {
            "key": "001|serial|:train:-",
            "attempt": 2,
            "status": "failed",
            "sky_status": sky_status,
            "job_id": "job-2",
            "job_name": "run-1-01-train-a2",
            "recovery_decision": recovery,
        }
    )

    record = state.in_flight_wave("001|serial|:train:-")
    assert record is not None
    assert record["recovery_decision"] == recovery


def test_run_state_store_persists_exact_nonempty_workflow_artifact() -> None:
    written: dict[tuple[str, str], bytes] = {}
    state_store = RunStateStore(
        bucket="bucket",
        prefix="runs/groot",
        writer=lambda bucket, key, body: written.__setitem__((bucket, key), body),
    )
    body = b"apiVersion: npa.workflow/v0.0.1\nkind: Workflow\n"

    uri = state_store.write_artifact(
        "workflow.yaml", body, content_type="application/yaml"
    )

    assert uri == "s3://bucket/runs/groot/workflow.yaml"
    assert written[("bucket", "runs/groot/workflow.yaml")] == body

    import pytest

    with pytest.raises(ValueError, match="non-empty"):
        state_store.write_artifact("empty.yaml", b"")
    with pytest.raises(ValueError, match="safe relative"):
        state_store.write_artifact("../escape.yaml", body)


def test_run_state_store_uses_explicit_artifact_listing_capability() -> None:
    listed: list[tuple[str, str]] = []

    def artifact_lister(bucket: str, prefix: str) -> list[str]:
        listed.append((bucket, prefix))
        return [f"{prefix}b.json", f"{prefix}a.json", "other/key.json"]

    state_store = RunStateStore(
        bucket="bucket",
        prefix="runs/demo",
        artifact_lister=artifact_lister,
    )

    assert state_store.list_artifacts("supervisor") == [
        "supervisor/a.json",
        "supervisor/b.json",
    ]
    assert listed == [("bucket", "runs/demo/supervisor/")]


def test_run_state_store_artifact_uses_explicit_storage_credentials(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeS3:
        def put_object(self, **kwargs: object) -> None:
            captured["put"] = kwargs

    class FakeStorage:
        _s3 = FakeS3()

    def fake_from_environment(**kwargs: str) -> FakeStorage:
        captured["credentials"] = kwargs
        return FakeStorage()

    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        fake_from_environment,
    )
    state_store = RunStateStore(
        bucket="bucket",
        prefix="runs/groot",
        endpoint_url="https://storage.example.invalid",
        aws_access_key_id="project-access",
        aws_secret_access_key="project-secret",
    )

    state_store.write_artifact("workflow.yaml", b"kind: Workflow\n")

    assert captured["credentials"] == {
        "endpoint_url": "https://storage.example.invalid",
        "aws_access_key_id": "project-access",
        "aws_secret_access_key": "project-secret",
    }
    assert captured["put"] == {
        "Bucket": "bucket",
        "Key": "runs/groot/workflow.yaml",
        "Body": b"kind: Workflow\n",
        "ContentType": "application/octet-stream",
    }


def test_run_state_store_output_check_uses_explicit_storage_credentials(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeS3:
        def head_object(self, **kwargs: object) -> dict[str, int]:
            captured["head"] = kwargs
            return {"ContentLength": 17}

    class FakeStorage:
        _s3 = FakeS3()

    def fake_from_environment(**kwargs: str) -> FakeStorage:
        captured["credentials"] = kwargs
        return FakeStorage()

    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        fake_from_environment,
    )
    state_store = RunStateStore(
        bucket="project-bucket",
        prefix="runs/demo",
        endpoint_url="https://project-storage.example.invalid",
        aws_access_key_id="project-access",
        aws_secret_access_key="project-secret",
    )

    assert state_store.artifact_exists("s3://project-bucket/runs/demo/result.json")
    assert captured["credentials"] == {
        "endpoint_url": "https://project-storage.example.invalid",
        "aws_access_key_id": "project-access",
        "aws_secret_access_key": "project-secret",
    }
    assert captured["head"] == {
        "Bucket": "project-bucket",
        "Key": "runs/demo/result.json",
    }


def test_run_state_store_prefix_output_looks_past_zero_byte_marker(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []
    captured_credentials: dict[str, str] = {}

    class FakeS3:
        def list_objects_v2(self, **kwargs: object) -> dict[str, object]:
            calls.append(dict(kwargs))
            if kwargs.get("ContinuationToken") == "page-2":
                return {
                    "Contents": [{"Key": "runs/demo/output/result.json", "Size": 17}],
                    "IsTruncated": False,
                }
            return {
                "Contents": [{"Key": "runs/demo/output/", "Size": 0}],
                "IsTruncated": True,
                "NextContinuationToken": "page-2",
            }

    class FakeStorage:
        _s3 = FakeS3()

    def fake_from_environment(**kwargs: str) -> FakeStorage:
        captured_credentials.update(kwargs)
        return FakeStorage()

    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        fake_from_environment,
    )
    state_store = RunStateStore(
        bucket="project-bucket",
        prefix="runs/demo",
        endpoint_url="https://project-storage.example.invalid",
        aws_access_key_id="synthetic-access",
        aws_secret_access_key="synthetic-secret",
    )

    assert state_store.artifact_exists("s3://project-bucket/runs/demo/output/")
    assert captured_credentials == {
        "endpoint_url": "https://project-storage.example.invalid",
        "aws_access_key_id": "synthetic-access",
        "aws_secret_access_key": "synthetic-secret",
    }
    assert calls == [
        {
            "Bucket": "project-bucket",
            "Prefix": "runs/demo/output/",
            "MaxKeys": 1000,
        },
        {
            "Bucket": "project-bucket",
            "Prefix": "runs/demo/output/",
            "MaxKeys": 1000,
            "ContinuationToken": "page-2",
        },
    ]


def test_run_state_store_prefix_output_rejects_stalled_pagination(
    monkeypatch,
) -> None:
    class FakeS3:
        def list_objects_v2(self, **_kwargs: object) -> dict[str, object]:
            return {
                "Contents": [{"Key": "runs/demo/output/", "Size": 0}],
                "IsTruncated": True,
            }

    class FakeStorage:
        _s3 = FakeS3()

    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        lambda **_kwargs: FakeStorage(),
    )
    state_store = RunStateStore(bucket="project-bucket", prefix="runs/demo")

    with pytest.raises(RuntimeError, match="continuation token"):
        state_store.artifact_exists("s3://project-bucket/runs/demo/output/")


@pytest.mark.parametrize(
    "response",
    [
        {"Contents": []},
        {"Contents": [], "NextContinuationToken": "page-2"},
        {
            "Contents": [],
            "IsTruncated": False,
            "NextContinuationToken": "page-2",
        },
        {"Contents": [], "IsTruncated": "false"},
        {"Contents": [], "IsTruncated": True, "NextContinuationToken": 2},
    ],
)
def test_run_state_store_prefix_output_rejects_malformed_pagination(
    monkeypatch,
    response: dict[str, object],
) -> None:
    calls = 0

    class FakeS3:
        def list_objects_v2(self, **_kwargs: object) -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls > 2:
                raise AssertionError("malformed pagination was followed")
            return response

    class FakeStorage:
        _s3 = FakeS3()

    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        lambda **_kwargs: FakeStorage(),
    )
    state_store = RunStateStore(bucket="project-bucket", prefix="runs/demo")

    with pytest.raises(RuntimeError, match="pagination|continuation token"):
        state_store.artifact_exists("s3://project-bucket/runs/demo/output/")
    assert calls == 1


@pytest.mark.parametrize("response", [None, [], "malformed"])
def test_run_state_store_prefix_output_rejects_malformed_response(
    monkeypatch,
    response: object,
) -> None:
    class FakeS3:
        def list_objects_v2(self, **_kwargs: object) -> object:
            return response

    class FakeStorage:
        _s3 = FakeS3()

    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        lambda **_kwargs: FakeStorage(),
    )
    state_store = RunStateStore(bucket="project-bucket", prefix="runs/demo")

    with pytest.raises(RuntimeError, match="malformed response"):
        state_store.artifact_exists("s3://project-bucket/runs/demo/output/")


def test_run_state_store_prefix_output_rejects_repeated_pagination_token(
    monkeypatch,
) -> None:
    calls = 0

    class FakeS3:
        def list_objects_v2(self, **_kwargs: object) -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls > 2:
                raise AssertionError("pagination cycle was not rejected")
            return {
                "Contents": [{"Key": "runs/demo/output/", "Size": 0}],
                "IsTruncated": True,
                "NextContinuationToken": "same-page",
            }

    class FakeStorage:
        _s3 = FakeS3()

    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        lambda **_kwargs: FakeStorage(),
    )
    state_store = RunStateStore(bucket="project-bucket", prefix="runs/demo")

    with pytest.raises(RuntimeError, match="continuation token"):
        state_store.artifact_exists("s3://project-bucket/runs/demo/output/")
    assert calls == 2


@pytest.mark.parametrize(
    "contents",
    [
        [],
        [{"Key": "runs/demo/output/", "Size": 0}],
        [
            {"Key": "runs/demo/output/", "Size": 0},
            {"Key": "runs/demo/output/empty.json", "Size": 0},
        ],
    ],
)
def test_run_state_store_prefix_output_requires_nonempty_content(
    monkeypatch,
    contents: list[dict[str, object]],
) -> None:
    class FakeS3:
        def list_objects_v2(self, **_kwargs: object) -> dict[str, object]:
            return {"Contents": contents, "IsTruncated": False}

    class FakeStorage:
        _s3 = FakeS3()

    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        lambda **_kwargs: FakeStorage(),
    )
    state_store = RunStateStore(bucket="project-bucket", prefix="runs/demo")

    assert not state_store.artifact_exists("s3://project-bucket/runs/demo/output/")


def test_run_state_store_prefix_output_scans_all_zero_byte_pages(
    monkeypatch,
) -> None:
    class FakeS3:
        def list_objects_v2(self, **kwargs: object) -> dict[str, object]:
            if kwargs.get("ContinuationToken") == "page-2":
                return {
                    "Contents": [{"Key": "runs/demo/output/empty.json", "Size": 0}],
                    "IsTruncated": False,
                }
            return {
                "Contents": [{"Key": "runs/demo/output/", "Size": 0}],
                "IsTruncated": True,
                "NextContinuationToken": "page-2",
            }

    class FakeStorage:
        _s3 = FakeS3()

    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        lambda **_kwargs: FakeStorage(),
    )
    state_store = RunStateStore(bucket="project-bucket", prefix="runs/demo")

    assert not state_store.artifact_exists("s3://project-bucket/runs/demo/output/")


@pytest.mark.parametrize(
    "item",
    [
        {"Key": "runs/demo/output/missing-size.json"},
        {"Key": "runs/demo/output/invalid-size.json", "Size": "invalid"},
        {"Key": "runs/demo/output/negative-size.json", "Size": -1},
        *(
            {"Key": "runs/demo/output/invalid.json", "Size": size}
            for size in [False, True, 0.5, 1.5, "0", None]
        ),
    ],
)
def test_run_state_store_prefix_output_rejects_malformed_size(
    monkeypatch,
    item: dict[str, object],
) -> None:
    class FakeS3:
        def list_objects_v2(self, **_kwargs: object) -> dict[str, object]:
            return {"Contents": [item], "IsTruncated": False}

    class FakeStorage:
        _s3 = FakeS3()

    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        lambda **_kwargs: FakeStorage(),
    )
    state_store = RunStateStore(bucket="project-bucket", prefix="runs/demo")

    with pytest.raises(RuntimeError, match="valid Size"):
        state_store.artifact_exists("s3://project-bucket/runs/demo/output/")


@pytest.mark.parametrize(
    "item",
    [
        {"Size": 17},
        {"Key": "runs/another/output.json", "Size": 17},
    ],
)
def test_run_state_store_prefix_output_rejects_malformed_identity(
    monkeypatch,
    item: dict[str, object],
) -> None:
    class FakeS3:
        def list_objects_v2(self, **_kwargs: object) -> dict[str, object]:
            return {"Contents": [item], "IsTruncated": False}

    class FakeStorage:
        _s3 = FakeS3()

    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        lambda **_kwargs: FakeStorage(),
    )
    state_store = RunStateStore(bucket="project-bucket", prefix="runs/demo")

    with pytest.raises(RuntimeError, match="requested prefix"):
        state_store.artifact_exists("s3://project-bucket/runs/demo/output/")


@pytest.mark.parametrize(
    ("code", "expected"),
    [("NoSuchKey", False), ("AccessDenied", "raises"), ("SlowDown", "raises")],
)
def test_run_state_store_prefix_output_preserves_provider_failures(
    monkeypatch,
    code: str,
    expected: bool | str,
) -> None:
    class FakeS3:
        def list_objects_v2(self, **_kwargs: object) -> dict[str, object]:
            raise ClientError({"Error": {"Code": code}}, "ListObjectsV2")

    class FakeStorage:
        _s3 = FakeS3()

    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        lambda **_kwargs: FakeStorage(),
    )
    state_store = RunStateStore(bucket="project-bucket", prefix="runs/demo")

    if expected == "raises":
        with pytest.raises(ClientError):
            state_store.artifact_exists("s3://project-bucket/runs/demo/output/")
    else:
        assert (
            state_store.artifact_exists("s3://project-bucket/runs/demo/output/")
            is expected
        )


def test_completed_wave_ignores_failed_attempts() -> None:
    from npa.orchestration.npa_workflow.run_state import RuntimeRunState

    state = RuntimeRunState(workflow="demo", run_id="demo-1")
    state.record_wave({"key": "001", "status": "failed", "attempt": 1})
    assert state.completed_wave("001") is None
    state.record_wave({"key": "001", "status": "succeeded", "attempt": 2})
    assert state.completed_wave("001")["attempt"] == 2


def test_read_runtime_state_propagates_unexpected_storage_errors() -> None:
    """A transient read error must not look like "no ledger" (resume safety)."""

    from npa.orchestration.npa_workflow.run_state import RunStateStore as Store

    def angry_reader(bucket: str, key: str) -> str:
        raise PermissionError(f"denied s3://{bucket}/{key}")

    store = Store(
        bucket="bucket", prefix="runs/demo", reader=angry_reader, writer=lambda *_: None
    )
    with pytest.raises(PermissionError):
        store.read_runtime_state()


@pytest.mark.parametrize(
    "corrupt_body",
    ['{"schema_version":', "[]", b'\xff\xfe{"schema_version":'],
)
def test_read_runtime_state_rejects_corrupt_ledger(
    corrupt_body: str | bytes,
) -> None:
    """Corrupt durable state must never be mistaken for an absent resume ledger."""

    from npa.orchestration.npa_workflow.errors import NpaWorkflowError
    from npa.orchestration.npa_workflow.run_state import RunStateStore as Store

    corrupt = Store(
        bucket="bucket",
        prefix="runs/demo",
        reader=lambda *_args: corrupt_body,
        writer=lambda *_: pytest.fail("corrupt state must never be overwritten"),
    )

    with pytest.raises(
        NpaWorkflowError,
        match=r"durable runtime state is corrupt.*runtime\.json",
    ) as error:
        corrupt.read_runtime_state()
    assert "runs/demo/npa-workflow/runtime.json" in str(error.value)
    assert "s3://bucket" not in str(error.value)


SEMANTIC_CORRUPTION_CASES = (
    "empty_object",
    "missing_run_id",
    "missing_workflow",
    "mismatched_run_id",
    "mismatched_workflow",
    "missing_schema_version",
    "unsupported_schema_version",
    "waves_string",
    "waves_non_object_entry",
    "stages_string",
    "decisions_object",
    "plan_migrations_string",
    "watermarks_array",
    "api_version_array",
)


def _semantic_runtime_payload(case: str) -> dict[str, object]:
    from npa.orchestration.npa_workflow.run_state import RuntimeRunState

    payload = RuntimeRunState(workflow="demo", run_id="demo-1").to_dict()
    if case == "empty_object":
        return {}
    if case.startswith("missing_"):
        payload.pop(case.removeprefix("missing_"))
    elif case == "mismatched_run_id":
        payload["run_id"] = "other-run"
    elif case == "mismatched_workflow":
        payload["workflow"] = "other-workflow"
    elif case == "unsupported_schema_version":
        payload["schema_version"] = "npa.workflow.runtime.v999"
    elif case == "waves_string":
        payload["waves"] = "corrupt-but-valid-json"
    elif case == "waves_non_object_entry":
        payload["waves"] = [
            {"key": "done", "status": "succeeded"},
            "corrupt-entry",
        ]
    elif case == "stages_string":
        payload["stages"] = "corrupt-but-valid-json"
    elif case == "decisions_object":
        payload["decisions"] = {"decision": "promote"}
    elif case == "plan_migrations_string":
        payload["plan_migrations"] = "corrupt-but-valid-json"
    elif case == "watermarks_array":
        payload["watermarks"] = []
    elif case == "api_version_array":
        payload["api_version"] = ["wrong-type"]
    return payload


@pytest.mark.parametrize("case", SEMANTIC_CORRUPTION_CASES)
def test_read_runtime_state_rejects_semantically_corrupt_or_mismatched_ledger(
    case: str,
) -> None:
    from npa.orchestration.npa_workflow.errors import NpaWorkflowError
    from npa.orchestration.npa_workflow.run_state import (
        RunStateStore as Store,
        runtime_key,
    )

    key = runtime_key("runs/demo")
    original = (
        json.dumps(_semantic_runtime_payload(case), sort_keys=True) + "\n"
    ).encode()
    objects = {key: original}
    writes: list[str] = []

    def writer(_bucket: str, object_key: str, body: bytes) -> None:
        writes.append(object_key)
        objects[object_key] = body

    store = Store(
        bucket="bucket",
        prefix="runs/demo",
        reader=lambda _bucket, object_key: objects[object_key],
        writer=writer,
    )

    with pytest.raises(
        NpaWorkflowError,
        match=r"durable runtime state is corrupt.*runtime\.json",
    ) as error:
        store.read_runtime_state(
            expected_workflow="demo",
            expected_run_id="demo-1",
        )

    assert writes == []
    assert objects[key] == original
    assert key in str(error.value)
    assert "s3://bucket" not in str(error.value)


def test_read_runtime_state_accepts_legacy_v1_without_additive_fields() -> None:
    from npa.orchestration.npa_workflow.run_state import (
        RUNTIME_SCHEMA_VERSION,
        RunStateStore as Store,
    )

    payload = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "workflow": "demo",
        "run_id": "demo-1",
        "waves": [],
    }
    store = Store(
        bucket="bucket",
        prefix="runs/demo",
        reader=lambda *_args: json.dumps(payload),
    )

    state = store.read_runtime_state(
        expected_workflow="demo",
        expected_run_id="demo-1",
    )

    assert state is not None
    assert state.waves == []
    assert state.stages == []
    assert state.decisions == []
    assert state.plan_migrations == []
    assert state.watermarks == {}


@pytest.mark.parametrize("error_code", ["AccessDenied", "SlowDown", "InternalError"])
def test_read_runtime_state_propagates_s3_read_failures(
    error_code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only an absent object may initialize an empty production resume ledger."""

    from botocore.exceptions import ClientError

    from npa.orchestration.npa_workflow.run_state import RunStateStore as Store

    class FailingS3:
        def get_object(self, **_kwargs: object) -> dict[str, object]:
            raise ClientError({"Error": {"Code": error_code}}, "GetObject")

    class FailingStorage:
        _s3 = FailingS3()

    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        lambda **_kwargs: FailingStorage(),
    )

    with pytest.raises(ClientError) as error:
        Store(bucket="bucket", prefix="runs/demo").read_runtime_state()
    assert error.value.response["Error"]["Code"] == error_code


def test_read_runtime_state_propagates_s3_transport_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from botocore.exceptions import EndpointConnectionError

    from npa.orchestration.npa_workflow.run_state import RunStateStore as Store

    failure = EndpointConnectionError(endpoint_url="https://storage.example.invalid")

    class FailingS3:
        def get_object(self, **_kwargs: object) -> dict[str, object]:
            raise failure

    class FailingStorage:
        _s3 = FailingS3()

    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        lambda **_kwargs: FailingStorage(),
    )

    with pytest.raises(EndpointConnectionError) as error:
        Store(bucket="bucket", prefix="runs/demo").read_runtime_state()
    assert error.value is failure


def test_read_runtime_state_returns_none_for_missing_s3_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from botocore.exceptions import ClientError

    from npa.orchestration.npa_workflow.run_state import RunStateStore as Store

    class MissingS3:
        def get_object(self, **_kwargs: object) -> dict[str, object]:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

    class MissingStorage:
        _s3 = MissingS3()

    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        lambda **_kwargs: MissingStorage(),
    )

    assert (
        Store(bucket="bucket", prefix="runs/missing").read_runtime_state(
            expected_workflow="demo",
            expected_run_id="demo-1",
        )
        is None
    )


# ── Resource-honest manifests for submitted runs ─────────────────────────────
# A cluster submit used to leave no `npa.workflow.run.v1` manifest at all, and the
# manifest the local path did write carried no resource profile -- so a run that
# requested N accelerators was indistinguishable from a CPU-only run, and the
# insights `gpus` metric (which reads steps[].resources_profile.accelerators) had
# no producer anywhere in the system.


class _FakeStep:
    def __init__(self, state, resources="", profile=None, tool_ref="", iteration=None):
        self.state = state
        self.resources = resources
        self.resources_profile = profile or {}
        self.tool_ref = tool_ref
        self.iteration = iteration
        self.group = ""
        self.loop_label = ""


def test_plan_step_records_carry_the_resource_profile() -> None:
    from npa.orchestration.npa_workflow.run_state import (
        SUBMITTED_STATUS,
        plan_step_records,
    )

    records = plan_step_records(
        [
            _FakeStep(
                "train",
                "trainer-gpu",
                {"accelerators": "RTXPRO6000:4", "cpus": 16},
                "workbench.rl.policy_train",
            ),
            _FakeStep("aggregate", "control-cpu", {"cpus": 4}),
        ]
    )
    assert records[0]["resources_profile"]["accelerators"] == "RTXPRO6000:4"
    assert records[0]["tool_ref"] == "workbench.rl.policy_train"
    assert records[0]["status"] == SUBMITTED_STATUS
    assert records[1]["resources_profile"] == {"cpus": 4}
    assert "tool_ref" not in records[1]


def test_plan_step_records_persist_exact_submit_accelerator_override() -> None:
    from npa.orchestration.npa_workflow.run_state import plan_step_records

    records = plan_step_records(
        [
            _FakeStep(
                "augment",
                "gpu",
                {"accelerators": "RTXPRO6000:1", "cpus": 16},
            )
        ],
        accelerator_override="RTXPRO-6000-BLACKWELL-SERVER-EDITION:1",
    )

    assert records[0]["resources_profile"]["accelerators"] == (
        "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
    )


def test_persist_submitted_manifest_writes_a_resource_honest_manifest() -> None:
    from npa.orchestration.npa_workflow import run_state as rs

    written: dict[tuple[str, str], bytes] = {}

    class _Store(rs.RunStateStore):
        def _write(self, key, payload):  # type: ignore[override]
            written[(self.bucket, key)] = json.dumps(payload).encode("utf-8")

    original = rs.store_for_config
    rs.store_for_config = lambda config, *, run_id, **kwargs: _Store(  # type: ignore[assignment]
        bucket=str(config.get("bucket")), prefix=str(config.get("prefix") or run_id)
    )
    try:
        uri = rs.persist_submitted_manifest(
            {"bucket": "bkt", "prefix": "runs/demo-1"},
            run_id="demo-1",
            workflow="demo",
            api_version="npa.workflow/v0.0.1",
            steps=[_FakeStep("train", "trainer-gpu", {"accelerators": "H100:2"})],
        )
    finally:
        rs.store_for_config = original

    assert uri == "s3://bkt/runs/demo-1"
    payload = json.loads(written[("bkt", "runs/demo-1/npa-workflow/manifest.json")])
    assert payload["schema_version"] == "npa.workflow.run.v1"
    assert payload["status"] == "submitted"
    assert payload["steps"][0]["resources_profile"]["accelerators"] == "H100:2"


def test_persist_submitted_manifest_without_a_bucket_is_a_no_op() -> None:
    from npa.orchestration.npa_workflow.run_state import persist_submitted_manifest

    assert persist_submitted_manifest({}, run_id="r", workflow="w", steps=[]) == ""


def test_paidf_input_provenance_survives_run_manifest_round_trip() -> None:
    from npa.orchestration.npa_workflow.run_state import input_source_from_config

    source = input_source_from_config(
        {
            "input_source_kind": "upstream_sample",
            "input_origin": "actual_capture",
            "input_origin_label": "Upstream real sample",
            "input_authoritative_url": "https://official.example/dataset",
            "input_immutable_revision": "a" * 40,
            "input_license": "CC-BY-4.0",
            "input_attribution": "Example author",
            "input_sha256": "b" * 64,
            "input_staged_uri": "s3://bucket/physical-ai-data-factory/run/input/",
            "input_provenance_uri": (
                "s3://bucket/physical-ai-data-factory/run/input/provenance.json"
            ),
        }
    )
    manifest = RunManifest(
        workflow="physical-ai-data-factory",
        run_id="run",
        api_version="npa.workflow/v0.0.1",
        input_source=source,
    )

    restored = RunManifest.from_dict(manifest.to_dict())

    assert restored.input_source == source
    assert restored.input_source["source_kind"] == "upstream_sample"
    assert restored.input_source["sha256"] == "b" * 64
    assert restored.input_source["staged_canonical_s3_uri"].endswith("/run/input/")


def test_persist_submitted_manifest_passes_configured_storage_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.orchestration.npa_workflow import run_state as rs

    captured: dict[str, object] = {}

    class _Store:
        run_prefix_uri = "s3://bucket/run-1"

        def write_manifest(self, manifest: object) -> None:
            captured["manifest"] = manifest

    def fake_store(config, *, run_id, **kwargs):  # noqa: ANN001
        captured.update(kwargs)
        return _Store()

    monkeypatch.setattr(rs, "store_for_config", fake_store)

    rs.persist_submitted_manifest(
        {"bucket": "bucket"},
        run_id="run-1",
        workflow="demo",
        endpoint_url="https://storage.example",
        aws_access_key_id="configured-access",
        aws_secret_access_key="configured-secret",
    )

    assert captured["endpoint_url"] == "https://storage.example"
    assert captured["aws_access_key_id"] == "configured-access"
    assert captured["aws_secret_access_key"] == "configured-secret"


def test_dispatch_step_records_carry_resources_for_any_executor() -> None:
    """The runtime tier's executor record must still describe the step's resources."""
    from npa.orchestration.npa_workflow.interpreter import PlanStep, _dispatch_step

    step = PlanStep(
        state="retrain",
        resources="trainer-gpu",
        resources_profile={"accelerators": "RTXPRO6000:2", "cpus": 16},
        inputs=[{"uri": "s3://bucket/data/", "schema": "dataset.v1"}],
        outputs=[{"uri": "s3://bucket/checkpoint.bin", "schema": "checkpoint.v1"}],
    )

    class _WaveExecutor:
        def execute(self, plan_step):
            return {"state": plan_step.state, "status": "ok", "job_id": "42"}

    record = _dispatch_step(step, _WaveExecutor())
    assert record["resources_profile"]["accelerators"] == "RTXPRO6000:2"
    assert record["resources"] == "trainer-gpu"
    assert record["job_id"] == "42"
    assert record["inputs"][0]["schema"] == "dataset.v1"
    assert record["outputs"][0]["schema"] == "checkpoint.v1"


@pytest.mark.parametrize("status", ["planned", "submitted", "running"])
def test_runtime_lifecycle_retains_existing_nonterminal_states(status):
    from npa.orchestration.npa_workflow.run_state import runtime_workflow_lifecycle

    manifest = RunManifest(
        "demo",
        "run-test",
        "npa.workflow/v0.0.1",
        status=status,
        updated_at="2001-01-01T00:00:00Z",
    )
    runtime = {"status": status, "updated_at": "2001-01-02T00:00:00Z"}
    observed, evidence = runtime_workflow_lifecycle(manifest, runtime)
    assert observed == status.upper()
    assert evidence["completion_recorded"] is False
    assert evidence["driver_liveness"] == "unknown"
    assert evidence["updated_at"] == runtime["updated_at"]
    assert manifest.status == status
    assert runtime == {"status": status, "updated_at": "2001-01-02T00:00:00Z"}


def test_manifest_completion_can_precede_runtime_finalization():
    from npa.orchestration.npa_workflow.run_state import runtime_workflow_lifecycle

    manifest = RunManifest(
        "demo",
        "run-test",
        "npa.workflow/v0.0.1",
        status="succeeded",
        updated_at="2026-01-02T03:04:05Z",
    )
    observed, evidence = runtime_workflow_lifecycle(manifest, {"status": "running"})
    assert observed == "SUCCEEDED"
    assert evidence["completion_recorded"] is True
    assert evidence["source"] == "authoritative_manifest"
    assert evidence["updated_at"] == manifest.updated_at
    assert evidence["driver_liveness"] == "unknown"


@pytest.mark.parametrize("raw_status", ["completed", "COMPLETED"])
@pytest.mark.parametrize("runtime_status", ["running", "succeeded"])
def test_manifest_completion_alias_retains_raw_provenance(raw_status, runtime_status):
    from npa.orchestration.npa_workflow.run_state import runtime_workflow_lifecycle

    manifest = RunManifest(
        "demo",
        "run-test",
        "npa.workflow/v0.0.1",
        status=raw_status,
        updated_at="2026-01-02T03:04:05Z",
    )
    runtime = {"status": runtime_status, "updated_at": "2026-01-02T03:04:06Z"}
    observed, evidence = runtime_workflow_lifecycle(manifest, runtime)
    assert observed == "SUCCEEDED"
    assert evidence["manifest_status"] == "SUCCEEDED"
    assert evidence["manifest_evidence"] == {
        "status": raw_status,
        "updated_at": "2026-01-02T03:04:05Z",
        "source": "authoritative_manifest",
    }
    assert evidence["source"] == (
        "authoritative_manifest"
        if runtime_status == "running"
        else "durable_runtime_ledger"
    )
    assert evidence["updated_at"] == (
        manifest.updated_at if runtime_status == "running" else runtime["updated_at"]
    )
    assert manifest.status == raw_status and runtime["status"] == runtime_status


@pytest.mark.parametrize("runtime_status", ["completed", "COMPLETED"])
def test_manifest_completion_alias_is_not_a_runtime_ledger_state(runtime_status):
    from npa.orchestration.npa_workflow.run_state import runtime_workflow_lifecycle

    manifest = RunManifest(
        "demo", "run-test", "npa.workflow/v0.0.1", status="completed"
    )
    with pytest.raises(ValueError, match="lifecycle status is missing or unsupported"):
        runtime_workflow_lifecycle(manifest, {"status": runtime_status})


@pytest.mark.parametrize("contents", [None, {}, "", False, 0])
def test_prefix_listing_rejects_falsey_malformed_contents(contents, mocker):
    from npa.orchestration.npa_workflow.run_state import s3_prefix_has_nonempty_object

    client = mocker.Mock()
    client.list_objects_v2.return_value = {
        "Contents": contents,
        "IsTruncated": False,
    }
    with pytest.raises(RuntimeError, match="malformed object records"):
        s3_prefix_has_nonempty_object(client, bucket="unit-bucket", prefix="runs/unit/")
