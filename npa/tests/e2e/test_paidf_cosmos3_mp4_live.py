"""Verify the guide's completed R3b local-MP4 run from durable artifacts.

Run R3b with its freshly downloaded public sample first. Set NPA_INTEGRATION_E2E=1,
NPA_E2E_PAIDF_MP4_RUN_URI, NPA_E2E_PROJECT, and NPA_E2E_PAIDF_MP4_FRESH_AFTER
(the UTC timestamp recorded before submission). This test only reads that run.
"""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlparse

import pytest

from .test_npa_workflow_submit_live_e2e import _assert_paidf_live_artifacts


pytestmark = [pytest.mark.e2e, pytest.mark.e2e_skypilot, pytest.mark.gpu]
SOURCE_SHA256 = "caadec919abfebe7ac7f571f52d0c579dbe86ceacc0d0bdbf9a862ed1a908198"


def test_completed_fresh_local_mp4_pipeline() -> None:
    uri = os.environ.get("NPA_E2E_PAIDF_MP4_RUN_URI", "")
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or not uri:
        pytest.skip("Requires the guide's completed fresh local-MP4 run")
    from npa.clients.project_credentials import s3_client_for_project

    parsed = urlparse(uri)
    assert parsed.scheme == "s3" and parsed.netloc
    prefix = parsed.path.strip("/") + "/"
    run_id = prefix.strip("/").split("/")[-1]
    assert prefix == f"paidf-cosmos3/{run_id}/"
    project = os.environ.get("NPA_E2E_PROJECT") or None
    client = s3_client_for_project(project, allow_host_creds=True)

    def read(relative):
        response = client.get_object(Bucket=parsed.netloc, Key=prefix + relative)
        with response["Body"] as body:
            return json.loads(body.read())

    runtime = read("npa-workflow/runtime.json")
    assert runtime["status"] == "succeeded" and runtime["run_id"] == run_id
    assert all(wave["status"] == "succeeded" for wave in runtime["waves"])
    assert all(wave["replayed"] is False for wave in runtime["waves"])
    assert all(wave["adopted"] is False for wave in runtime["waves"])
    provenance = read("input/provenance.json")
    assert provenance["source_kind"] == "video_uri"
    assert provenance["run_id"] == run_id
    _assert_public_mp4_input(client, parsed.netloc, run_id, read)
    _assert_fresh_objects(client, parsed.netloc, run_id)
    _assert_paidf_live_artifacts(
        spec="paidf-cosmos3.yaml", waves=runtime["waves"], bucket=parsed.netloc,
        run_id=run_id, e2e_project=project,
    )
    _assert_recording_identity(client, parsed.netloc, prefix, run_id)


def _assert_public_mp4_input(client, bucket, run_id, read) -> None:
    input_prefix = f"physical-ai-data-factory/{run_id}/input/"
    response = client.get_object(Bucket=bucket, Key=input_prefix + "provenance.json")
    with response["Body"] as body:
        provenance = json.loads(body.read())
    assert provenance["source_kind"] == "user_supplied"
    assert provenance["transport"] == "local_file"
    assert provenance["sha256"] == SOURCE_SHA256
    assert provenance["run_id"] == run_id
    original_key = f"paidf-cosmos3/{run_id}/input/original_source.mp4"
    for key in (input_prefix + "source.mp4", original_key):
        response = client.get_object(Bucket=bucket, Key=key)
        with response["Body"] as body:
            assert hashlib.sha256(body.read()).hexdigest() == SOURCE_SHA256
    timeline = read("input/timeline.json")
    assert timeline["original"]["decoded_frames"] == 169
    assert timeline["original"]["duration_seconds"] == pytest.approx(3.38)
    assert timeline["prepared"]["decoded_frames"] == 81
    assert timeline["prepared"]["duration_seconds"] == pytest.approx(3.375)
    assert len(read("cosmos_augmented/manifest.json")["variants"]) == 2


def _assert_fresh_objects(client, bucket, run_id) -> None:
    fresh_after = datetime.fromisoformat(os.environ["NPA_E2E_PAIDF_MP4_FRESH_AFTER"])
    assert fresh_after.tzinfo is not None, "Freshness timestamp must include UTC offset"
    paginator = client.get_paginator("list_objects_v2")
    for root in ("paidf-cosmos3", "physical-ai-data-factory"):
        count = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{root}/{run_id}/"):
            for item in page.get("Contents", []):
                assert item["LastModified"] >= fresh_after, "Object predates this submission"
                count += 1
        assert count > 0, f"Missing {root} artifacts"


def _assert_recording_identity(client, bucket, prefix, run_id) -> None:
    from rerun.recording import load_recording

    with tempfile.TemporaryDirectory(prefix="paidf-mp4-recordings-") as temporary:
        for name in ("quality-evidence.rrd", "sim2real.rrd"):
            path = Path(temporary) / name
            client.download_file(bucket, prefix + "reports/" + name, str(path))
            recording = load_recording(path)
            assert recording.recording_id() == run_id
            assert recording.application_id() == "neural-reconstruction"
