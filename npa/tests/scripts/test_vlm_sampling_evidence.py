"""Verify the served-model sampling oracle without calling a provider."""

import base64
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from npa.workbench import vlm_eval


@pytest.mark.parametrize("strategy", ["final", "keyframes", "sequence"])
@pytest.mark.parametrize("kind", ["image-sequence", "numpy-episode", "video"])
def test_live_sampling_oracle_with_local_media(monkeypatch, tmp_path, strategy, kind):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    suite = importlib.import_module("e2e.test_vlm_served_model_live")
    requests = []

    def recorded_response(**kwargs):
        requests.append(kwargs["request"])
        return _synthetic_response()

    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", recorded_response)
    config = {
        "endpoint_url": "https://example.test/v1",
        "model": "test-model",
        "expected_served_model": "test-model",
    }
    suite.test_self_hosted_sampling_binds_source_payloads(
        config, tmp_path, kind, strategy
    )
    payload = json.loads((tmp_path / "evaluation.json").read_text())
    frames = payload["evidence"]["request"]["frames"]
    content = requests[0]["messages"][-1]["content"]
    encoded = [
        item["image_url"]["url"].split(",", 1)[1]
        for item in content
        if item["type"] == "image_url"
    ]
    assert [
        hashlib.sha256(base64.b64decode(value)).hexdigest() for value in encoded
    ] == [frame["sha256"] for frame in frames]
    _assert_oracle_rejects_tampering(suite, payload, tmp_path, kind, strategy)


def _synthetic_response(*, fenced=False):
    content = '{"score":0.9,"success":true,"rationale":"synthetic test"}'
    if fenced:
        content = f"```json\n{content}\n```"
    data = {
        "model": "test-model",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": content},
            }
        ],
    }
    return vlm_eval._VlmBackendResponse(data, json.dumps(data), 200, None, 0.1)


@pytest.mark.parametrize("fenced", [False, True])
def test_served_evidence_accepts_only_documented_parser_versions(
    monkeypatch, tmp_path, fenced
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    suite = importlib.import_module("e2e.test_vlm_served_model_live")
    monkeypatch.setattr(
        vlm_eval,
        "_post_with_readiness_retry",
        lambda **_: _synthetic_response(fenced=fenced),
    )
    input_path, _ = suite.make_sampling_input(tmp_path / "input", "image-sequence")
    config = {
        "input_path": str(input_path),
        "output_path": str(tmp_path / "result.json"),
        "endpoint_url": "https://example.test/v1",
        "model": "test-model",
        "expected_served_model": "test-model",
        "task": "Inspect the colors",
    }
    payload = suite._run_evaluation(config, strategy="keyframes", max_frames=3)
    for version in (
        "npa_vlm_eval_compatible_json_v123",
        "npa_vlm_eval_compatible_json_v1+unknown",
    ):
        payload["evidence"]["provider"]["parser_version"] = version
        with pytest.raises(AssertionError):
            suite._assert_provider_evidence(payload, config)


def _assert_oracle_rejects_tampering(suite, payload, tmp_path, kind, strategy):
    _, expected = suite.make_sampling_input(tmp_path / "independent", kind)
    for field, bad_value in (
        ("sha256", "0" * 64),
        ("source_index", 4),
        ("source_count", 8),
        ("source_timestamp_s", 999.0),
    ):
        tampered = json.loads(json.dumps(payload))
        request = tampered["evidence"]["request"]
        request["frames"][0][field] = bad_value
        request["request_manifest"]["frames"][0][field] = bad_value
        with pytest.raises(AssertionError):
            suite.assert_sampling_evidence(tampered, kind, strategy, expected)
    tampered = json.loads(json.dumps(payload))
    tampered["evidence"]["request"]["request_manifest"]["sampling"][
        "coverage_complete"
    ] = False
    with pytest.raises(AssertionError):
        suite.assert_sampling_evidence(tampered, kind, strategy, expected)
