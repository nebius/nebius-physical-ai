"""Audit a completed fresh default-starter PAIDF run without submitting work."""

from datetime import datetime
import hashlib
import json
import os
from urllib.parse import urlparse

import pytest

from .paidf_runtime_audit import assert_completed_fresh_runtime
from .test_npa_workflow_submit_live_e2e import _assert_paidf_live_artifacts
from .test_paidf_cosmos3_mp4_live import _assert_recording_identity


pytestmark = [pytest.mark.e2e, pytest.mark.e2e_skypilot, pytest.mark.gpu]


def test_completed_fresh_default_starter_pipeline() -> None:
    uri = os.environ.get("NPA_E2E_PAIDF_STARTER_RUN_URI", "")
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or not uri:
        pytest.skip("Requires the guide's completed fresh default-starter run")
    from npa.clients.project_credentials import s3_client_for_project

    selected = urlparse(uri)
    assert selected.scheme == "s3" and selected.netloc
    prefix = selected.path.strip("/") + "/"
    run_id = prefix.strip("/").split("/")[-1]
    assert prefix == f"paidf-cosmos3/{run_id}/"
    project = os.environ.get("NPA_E2E_PROJECT") or None
    client = s3_client_for_project(project, allow_host_creds=True)
    runtime = _read(client, selected.netloc, prefix + "npa-workflow/runtime.json")
    assert runtime["status"] == "succeeded" and runtime["run_id"] == run_id
    assert_completed_fresh_runtime(client, selected.netloc, prefix, runtime)
    _assert_starter_input(client, selected.netloc, run_id)
    _assert_default_batch(client, selected.netloc, prefix)
    _assert_fresh_objects(client, selected.netloc, run_id)
    _assert_paidf_live_artifacts(
        spec="paidf-cosmos3.yaml", waves=runtime["waves"], bucket=selected.netloc,
        run_id=run_id, e2e_project=project,
    )
    _assert_recording_identity(client, selected.netloc, prefix, run_id)


def _read(client, bucket, key):
    response = client.get_object(Bucket=bucket, Key=key)
    with response["Body"] as body:
        return json.loads(body.read())


def _assert_starter_input(client, bucket, run_id) -> None:
    from npa.workflows.data_factory_input import load_starter_contract

    contract = load_starter_contract()
    input_prefix = f"physical-ai-data-factory/{run_id}/input/"
    provenance = _read(client, bucket, input_prefix + "provenance.json")
    assert provenance["run_id"] == run_id
    assert provenance["source_kind"] == "upstream_sample"
    assert provenance["immutable_revision"] == contract["source"]["immutable_revision"]
    assert provenance["sha256"] == contract["integrity"]["sha256"]
    keys = [input_prefix + "source.mp4", f"paidf-cosmos3/{run_id}/input/original_source.mp4"]
    for key in keys:
        response = client.get_object(Bucket=bucket, Key=key)
        with response["Body"] as body:
            contents = body.read()
        assert len(contents) == contract["integrity"]["byte_size"]
        assert hashlib.sha256(contents).hexdigest() == contract["integrity"]["sha256"]


def _assert_default_batch(client, bucket, prefix) -> None:
    manifest = _read(client, bucket, prefix + "cosmos_augmented/manifest.json")
    assert len(manifest["variants"]) == manifest["variant_count"] == 2
    assert manifest["variant_parallelism"] == 1
    assert manifest["motion_preservation"]["source_weight"] == 0
    configs = _read(client, bucket, prefix + "configs/manifest.json")
    assert configs["n_augmentations"] == 2
    assert str(configs["augmentation_seed"]) == "30"
    evaluator = _read(client, bucket, prefix + "grade/cosmos_evaluator.json")
    assert evaluator["threshold"] == 0.2
    assert evaluator["attribute_threshold"] == 0.25


def _assert_fresh_objects(client, bucket, run_id) -> None:
    fresh_after = datetime.fromisoformat(os.environ["NPA_E2E_PAIDF_STARTER_FRESH_AFTER"])
    assert fresh_after.tzinfo is not None, "Freshness timestamp must include UTC offset"
    paginator = client.get_paginator("list_objects_v2")
    for root in ("paidf-cosmos3", "physical-ai-data-factory"):
        count = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{root}/{run_id}/"):
            for item in page.get("Contents", []):
                assert item["LastModified"] >= fresh_after, "Object predates this submission"
                count += 1
        assert count > 0, f"Missing {root} artifacts"
