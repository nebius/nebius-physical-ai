"""Verify model provenance against an operator-provisioned GPU VLM endpoint."""

import hashlib
import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.vlm_eval import select_rollout_frames


@pytest.mark.e2e
@pytest.mark.gpu
def test_self_hosted_result_retains_served_model() -> None:
    config_path = os.environ.get("NPA_VLM_PROVENANCE_LIVE_CONFIG", "")
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or not config_path:
        pytest.skip("provide an operator-owned NPA_VLM_PROVENANCE_LIVE_CONFIG")
    config = json.loads(Path(config_path).read_text())

    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "vlm-eval",
            "run",
            "--input-path",
            config["input_path"],
            "--output-path",
            config["output_path"],
            "--task",
            config["task"],
            "--backend",
            "self-hosted",
            "--endpoint-url",
            config["endpoint_url"],
            "--model",
            config["model"],
            "--api-key-env",
            config.get("api_key_env", "VLM_EVAL_API_KEY"),
            "--frame-selection",
            "keyframes",
            "--max-frames",
            "8",
            "--success-threshold",
            "0.8",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["served_model"] == config["expected_served_model"]
    assert payload["model"] == config["model"]
    assert 0 <= payload["score"] <= 1
    assert payload["backend"] == "self-hosted" and not payload["dry_run"]
    evidence = payload["evidence"]
    assert evidence["schema_version"] == "npa_vlm_eval_evidence_v1"
    assert evidence["request"]["endpoint_role"] == "self-hosted"
    manifest = json.dumps(
        evidence["request"]["request_manifest"],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert (
        evidence["request"]["request_manifest_sha256"]
        == hashlib.sha256(manifest.encode()).hexdigest()
    )
    raw_response = evidence["provider"]["raw_response"]
    assert (
        evidence["provider"]["raw_response_sha256"]
        == hashlib.sha256(raw_response.encode()).hexdigest()
    )
    assert evidence["provider"]["parser_version"] == "npa_vlm_eval_compatible_json_v1"
    if not config["input_path"].startswith("s3://"):
        selected = select_rollout_frames(
            config["input_path"], frame_selection="keyframes", max_frames=8
        )
        assert [frame["sha256"] for frame in evidence["request"]["frames"]] == [
            hashlib.sha256(frame.data).hexdigest() for frame in selected
        ]
    assert payload.pop("written_uri") == config["output_path"]
    saved = json.loads(Path(config["output_path"]).read_text())
    assert saved == payload
