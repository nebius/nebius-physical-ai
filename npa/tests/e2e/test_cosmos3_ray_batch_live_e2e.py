"""Real guarded Cosmos Ray batch proof against an operator-started service.

Run with NPA_INTEGRATION_E2E=1, NPA_COSMOS3_RAY_ENDPOINT,
NPA_COSMOS3_RAY_TOKEN and NPA_COSMOS3_RAY_LIVE_OUTPUT_URI (an owned S3 prefix).
Uses public synthetic prompts; creates no compute and never accepts vendor terms.
The operator retains outputs and owns service/storage cleanup.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import uuid

from PIL import Image
import pytest

from npa.clients.storage import StorageClient
from npa.workbench.cosmos import ray_serve

pytestmark = pytest.mark.e2e


def test_guarded_batch_publishes_two_decodable_samples_and_rejects_incomplete_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = os.environ.get("NPA_COSMOS3_RAY_LIVE_OUTPUT_URI", "").rstrip("/")
    if not prefix:
        pytest.skip("requires an operator-owned Cosmos Ray live output prefix")
    assert prefix.startswith("s3://")
    assert os.environ.get(ray_serve.DEFAULT_ENDPOINT_ENV)
    assert os.environ.get(ray_serve.DEFAULT_TOKEN_ENV)
    ready = ray_serve.service_health()
    assert ready["status"] == "ready"
    assert ready["guardrails"] is True
    storage = StorageClient.from_environment()
    request_id = uuid.uuid4().hex
    root = prefix + "/" + request_id
    request = {
        "model": "Cosmos3-Nano",
        "request_id": request_id,
        "samples": [
            {
                "name": "red-cube",
                "model_mode": "text2image",
                "num_steps": 4,
                "prompt": "a red cube on a robotics workbench",
                "seed": 17,
            },
            {
                "name": "blue-cube",
                "model_mode": "text2image",
                "num_steps": 4,
                "prompt": "a blue cube on a robotics workbench",
                "seed": 23,
            },
        ],
    }
    local_request = tmp_path / "input.json"
    local_request.write_text(json.dumps(request))
    input_uri = storage.upload_file(str(local_request), root + "/input.json")
    result = ray_serve.submit_batch(
        input_path=input_uri,
        output_path=root + "/outputs/",
        storage_client=storage,
    )
    assert result["status"] == "completed"
    assert result["request_id"] == request_id
    assert result["guardrails"] is True
    assert len(result["artifacts"]) == 2
    records = {}
    for name in ["request", "response", "provenance"]:
        path = tmp_path / (name + ".json")
        storage.download_file(root + f"/outputs/{name}.json", str(path))
        records[name] = json.loads(path.read_text())
    assert records["request"] == request
    assert records["provenance"] == result
    assert records["response"]["outputs"] == result["structured_outputs"]
    for index, artifact in enumerate(result["artifacts"]):
        path = tmp_path / f"sample-{index}.jpg"
        storage.download_file(
            root + "/outputs/artifacts/" + artifact["path"], str(path)
        )
        data = path.read_bytes()
        assert len(data) == artifact["bytes"]
        assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
        with Image.open(path) as image:
            image.load()
            assert image.width > 0 and image.height > 0

    # Replay a malformed copy of the real response without another GPU call.
    # The response itself still declares both real successful samples.
    invalid = copy.deepcopy(records["response"])
    invalid["artifacts"].pop()
    monkeypatch.setattr(ray_serve, "_request_json", lambda *a, **kw: invalid)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid response reached artifact download or publication")

    monkeypatch.setattr(ray_serve, "_request_bytes", forbidden)
    monkeypatch.setattr(storage, "upload_directory", forbidden)
    with pytest.raises(ray_serve.Cosmos3RayServeError, match="exactly cover"):
        ray_serve.submit_batch(
            input_path=input_uri,
            output_path=root + "/invalid/",
            storage_client=storage,
        )
