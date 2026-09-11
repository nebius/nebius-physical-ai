"""Verify the guide's completed public LeRobot v3 run from durable artifacts.

Run the guide's R3a command first, then set NPA_INTEGRATION_E2E=1,
NPA_E2E_PAIDF_LEROBOT_RUN_URI and NPA_E2E_PROJECT. This test reads the completed
run; it does not submit jobs or change their inputs. The pinned example is
episode 1, camera observation.images.top, from the simulated ALOHA dataset.
"""

import json
import os
from urllib.parse import urlparse

import pytest

from .test_npa_workflow_submit_live_e2e import _assert_paidf_live_artifacts


pytestmark = [pytest.mark.e2e, pytest.mark.e2e_skypilot, pytest.mark.gpu]


def test_completed_public_lerobot_v3_pipeline() -> None:
    uri = os.environ.get("NPA_E2E_PAIDF_LEROBOT_RUN_URI", "")
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or not uri:
        pytest.skip("Requires the guide's completed LeRobot v3 run")
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
    provenance = read("input/provenance.json")
    assert provenance["source_kind"] == "lerobot_dataset"
    assert provenance["episode"] == 1
    assert provenance["camera"] == "observation.images.top"
    _assert_example_timeline(read)
    _assert_paidf_live_artifacts(
        spec="paidf-cosmos3.yaml", waves=runtime["waves"], bucket=parsed.netloc,
        run_id=run_id, e2e_project=project,
    )


def _assert_example_timeline(read) -> None:
    timeline = read("input/timeline.json")
    assert timeline["original"]["duration_seconds"] == pytest.approx(8.0)
    assert timeline["original"]["decoded_frames"] == 400
    assert timeline["prepared"]["duration_seconds"] == pytest.approx(8.0)
    assert timeline["prepared"]["decoded_frames"] == 192
    augment = read("cosmos_augmented/manifest.json")
    assert len(augment["variants"]) == 2
    for variant in augment["variants"]:
        transfer = read(f"cosmos_augmented/{variant['clip']}/transfer.json")
        assert transfer["native_chunks"] > 1
        assert transfer["source_frames"] == 192
