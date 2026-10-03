"""Verify run-prefix ownership against real S3 without launching compute jobs.

Set NPA_INTEGRATION_E2E=1 and NPA_E2E_PROJECT to a configured test project.
Each case uses a unique prefix and removes only its own objects afterward.
"""

from __future__ import annotations

import os
import uuid

import pytest
import yaml

from npa.clients.config import resolve_project_storage
from npa.clients.storage import StorageClient
from npa.orchestration.npa_workflow import load_spec, runtime
from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.run_state import (
    RunManifest,
    RunStateStore,
    RuntimeRunState,
)

pytestmark = pytest.mark.e2e


def _snapshot(client, store):
    objects = {}
    pages = client.get_paginator("list_objects_v2").paginate(
        Bucket=store.bucket, Prefix=store.prefix + "/"
    )
    for page in pages:
        for row in page.get("Contents", []):
            response = client.get_object(Bucket=store.bucket, Key=row["Key"])
            with response["Body"] as body:
                objects[row["Key"]] = body.read()
    return objects


@pytest.fixture
def owned_storage():
    project = os.environ.get("NPA_E2E_PROJECT", "").strip()
    if not project:
        pytest.skip("Set NPA_E2E_PROJECT to an explicitly configured test project")
    config = resolve_project_storage(
        project, include_shared_credentials=False, include_environment=False
    )
    credentials = {
        "endpoint_url": config.endpoint_url,
        "aws_access_key_id": config.aws_access_key_id,
        "aws_secret_access_key": config.aws_secret_access_key,
    }
    if not config.checkpoint_bucket or not all(credentials.values()):
        pytest.fail("Selected project needs complete object storage configuration")
    bucket = config.checkpoint_bucket.removeprefix("s3://").split("/", 1)[0]
    store = RunStateStore(
        bucket=bucket,
        prefix="workflow-prefix-ownership-test/" + uuid.uuid4().hex,
        **credentials,
    )
    client = StorageClient(**credentials)._s3
    try:
        assert _snapshot(client, store) == {}
        yield store, client
    finally:
        for key in _snapshot(client, store):
            client.delete_object(Bucket=store.bucket, Key=key)
        assert _snapshot(client, store) == {}


def _seed_owner(store, include_runtime):
    store.write_artifact("workflow.yaml", b"original submitted workflow\n")
    store.write_manifest(
        RunManifest(
            workflow="prefix-guard-live",
            run_id="first-run",
            api_version="npa.workflow/v0.0.1",
        )
    )
    if include_runtime:
        store.write_runtime_state(
            RuntimeRunState(workflow="prefix-guard-live", run_id="first-run")
        )


def _workflow(tmp_path, store):
    document = {
        "apiVersion": "npa.workflow/v0.0.1",
        "kind": "Workflow",
        "metadata": {"name": "prefix-guard-live"},
        "config": {"bucket": store.bucket, "prefix": store.prefix},
        "initial": "probe",
        "states": {"probe": {"run": {"shell": "exit 0"}, "terminal": True}},
    }
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(document))
    return load_spec(path)


@pytest.mark.parametrize("include_runtime", [True, False])
def test_live_foreign_run_preserves_every_object(
    owned_storage, tmp_path, monkeypatch, include_runtime
):
    store, client = owned_storage
    _seed_owner(store, include_runtime)
    original = _snapshot(client, store)

    def forbid_executor(*args, **kwargs):
        pytest.fail("Prefix rejection must happen before constructing a job executor")

    monkeypatch.setattr(runtime, "SkyPilotWaveExecutor", forbid_executor)
    with pytest.raises(NpaWorkflowError, match="already belongs.*first-run"):
        runtime.run_workflow_runtime(
            _workflow(tmp_path, store),
            run_id="second-run",
            state_store=store,
            workflow_yaml=b"replacement workflow must never be stored\n",
        )
    assert _snapshot(client, store) == original
