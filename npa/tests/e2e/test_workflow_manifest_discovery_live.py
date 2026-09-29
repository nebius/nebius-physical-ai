"""Live evidence that concurrent run-manifest discovery works end to end.

Run with ``NPA_INTEGRATION_E2E=1`` and ``NPA_E2E_PROJECT`` set to an
explicitly configured project alias with real, writable S3 storage. Seeds a
handful of durable run manifests under a unique owned prefix, proves
``list_runs`` and ``discover_workflow_run_state`` find them correctly
through the concurrent, single-shared-client path, and deletes every seeded
object afterward. The wall-clock speedup itself is covered by a separate,
non-committed diagnostic comparison (base vs candidate against the same
real bucket), since comparing two source trees in one pytest run is a
benchmark, not a regression test.
"""

from __future__ import annotations

import json
import os
import uuid

import pytest

from npa.clients.config import resolve_project_storage
from npa.clients.storage import StorageClient
from npa.orchestration.skypilot.workflow_state import (
    WorkflowS3Config,
    discover_workflow_run_state,
    list_runs,
)

from .s3_fixture_cleanup import delete_owned_prefix

pytestmark = pytest.mark.e2e

RUN_COUNT = 12


def _seed_manifests(client, bucket: str, prefix: str) -> None:
    """Write manifests under the unique prefix whose versions the fixture cleans."""

    for i in range(RUN_COUNT):
        run_id = f"run-{i:03d}"
        key = f"{prefix}/{run_id}/manifest.json"
        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "workflow_name": "live-test",
            "stages": {},
            "updated_at": f"2026-01-01T00:{i:02d}:00Z",
        }
        client.s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(payload).encode("utf-8"),
            ContentType="application/json",
        )


@pytest.fixture
def live_manifests():
    project = os.environ.get("NPA_E2E_PROJECT", "").strip()
    if not project:
        pytest.skip("Set NPA_E2E_PROJECT to an explicitly configured test project")
    storage = resolve_project_storage(project)
    if not all(
        (
            storage.checkpoint_bucket,
            storage.endpoint_url,
            storage.aws_access_key_id,
            storage.aws_secret_access_key,
        )
    ):
        pytest.fail("Selected live project needs complete object storage configuration")
    bucket = storage.checkpoint_bucket.removeprefix("s3://").split("/", 1)[0]
    prefix = "workflow-manifest-discovery-live-test/" + uuid.uuid4().hex
    client = StorageClient(
        endpoint_url=storage.endpoint_url,
        aws_access_key_id=storage.aws_access_key_id,
        aws_secret_access_key=storage.aws_secret_access_key,
    )
    state_parent = WorkflowS3Config(
        bucket=bucket,
        prefix=prefix,
        endpoint_url=storage.endpoint_url,
        aws_access_key_id=storage.aws_access_key_id,
        aws_secret_access_key=storage.aws_secret_access_key,
    )
    try:
        _seed_manifests(client, bucket, prefix)
        yield client, bucket, prefix, state_parent
    finally:
        delete_owned_prefix(client.s3, bucket, prefix + "/")


def test_list_runs_finds_every_seeded_manifest(live_manifests):
    _client, _bucket, _prefix, state_parent = live_manifests

    runs = list_runs(state_parent=state_parent, limit=1000)

    assert sorted(r["run_id"] for r in runs) == [
        f"run-{i:03d}" for i in range(RUN_COUNT)
    ]
    assert all(r["workflow_name"] == "live-test" for r in runs)


def test_discover_workflow_run_state_finds_the_exact_run(live_manifests):
    _client, _bucket, prefix, state_parent = live_manifests
    wanted = "run-007"

    found = discover_workflow_run_state(state_parent=state_parent, run_id=wanted)

    assert found is not None
    assert found.prefix == f"{prefix}/{wanted}"


def test_discover_workflow_run_state_returns_none_for_unknown_run(live_manifests):
    _client, _bucket, _prefix, state_parent = live_manifests

    found = discover_workflow_run_state(
        state_parent=state_parent, run_id="does-not-exist"
    )

    assert found is None
