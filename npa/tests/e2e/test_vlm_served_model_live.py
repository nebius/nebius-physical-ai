"""Verify model provenance against an operator-provisioned GPU VLM endpoint."""

import hashlib
import json
import os
import re
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
    result = CliRunner().invoke(app, _evaluation_args(config))
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["served_model"] == config["expected_served_model"]
    assert payload["model"] == config["model"]
    assert 0 <= payload["score"] <= 1
    assert payload["backend"] == "self-hosted" and not payload["dry_run"]
    _assert_request_evidence(payload["evidence"], config)
    _assert_provider_evidence(payload["evidence"]["provider"], config)
    assert payload.pop("written_uri") == config["output_path"]
    saved = json.loads(Path(config["output_path"]).read_text())
    assert saved == payload
    _assert_self_hosted_judge_claims(saved)


def _evaluation_args(config: dict) -> list[str]:
    return [
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
    ]


def _assert_request_evidence(evidence: dict, config: dict) -> None:
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
    if not config["input_path"].startswith("s3://"):
        selected = select_rollout_frames(
            config["input_path"], frame_selection="keyframes", max_frames=8
        )
        assert [frame["sha256"] for frame in evidence["request"]["frames"]] == [
            hashlib.sha256(frame.data).hexdigest() for frame in selected
        ]


def _assert_provider_evidence(provider: dict, config: dict) -> None:
    raw_response = provider["raw_response"]
    assert (
        provider["raw_response_sha256"]
        == hashlib.sha256(raw_response.encode()).hexdigest()
    )
    assert provider["status_code"] == 200
    assert provider["returned_model"] == config["expected_served_model"]
    assert provider["finish_reason"] == "stop"
    response = json.loads(raw_response)
    assert response["model"] == config["expected_served_model"]
    assert response["choices"][0]["finish_reason"] == "stop"
    content = response["choices"][0]["message"]["content"].strip()
    fence = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE
    )
    expected_parser = "npa_vlm_eval_compatible_json_v2"
    if fence:
        content = fence.group(1)
        expected_parser += "+markdown-fence-v1"
    assert provider["parser_version"] == expected_parser
    assert isinstance(json.loads(content), dict)


def _assert_self_hosted_judge_claims(saved: dict) -> None:
    from npa.workbench.vlm_eval import _build_prompt, parse_structured_response

    evidence = saved["evidence"]
    response = json.loads(evidence["provider"]["raw_response"])
    verdict = parse_structured_response(response["choices"][0]["message"]["content"])
    assert saved["score"] == round(verdict.score, 4)
    assert saved["passed"] is (saved["score"] >= saved["success_threshold"])
    assert saved["status"] == ("passed" if saved["passed"] else "needs_iteration")
    assert saved["provider_success"] is verdict.provider_success
    matches_gate = (
        None
        if verdict.provider_success is None
        else verdict.provider_success == saved["passed"]
    )
    assert saved["provider_success_matches_score_gate"] is matches_gate
    prompt = _build_prompt(
        **{
            field: saved[field]
            for field in ("task", "rubric", "frame_selection", "frame_count")
        }
    )
    assert (
        evidence["request"]["prompt_sha256"]
        == hashlib.sha256(prompt.encode()).hexdigest()
    )
    assert (
        evidence["request"]["rubric_sha256"]
        == hashlib.sha256(saved["rubric"].encode()).hexdigest()
    )
