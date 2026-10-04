"""Verify the served-model sampling oracle without calling a provider."""

import base64
import copy
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from npa.workbench import vlm_eval
from npa.workbench.vlm_eval import VlmEvalError


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
    _assert_scalar_tampering_rejected(suite, payload)


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


@pytest.mark.parametrize("provider_success", [True, False, None, "true"])
def test_sampling_judge_claims_use_effective_rubric_and_provider_boolean(
    monkeypatch, tmp_path, provider_success
):
    from dataclasses import asdict

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    suite = importlib.import_module("e2e.test_vlm_served_model_live")
    response = _synthetic_response()
    verdict = {"score": 0.9, "rationale": "synthetic test"}
    if provider_success is not None:
        verdict["success"] = provider_success
    response.data["choices"][0]["message"]["content"] = json.dumps(verdict)
    response = vlm_eval._VlmBackendResponse(
        response.data, json.dumps(response.data), 200, None, 0.1
    )
    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", lambda **_: response)
    source, _ = suite.make_sampling_input(tmp_path / "input", "image-sequence")
    if not isinstance(provider_success, bool):

        def forbidden_result(**_):
            pytest.fail("Invalid provider success reached result construction")

        monkeypatch.setattr(vlm_eval, "_result_from_structured", forbidden_result)
        with pytest.raises(VlmEvalError, match="success must be a boolean"):
            _sampling_judge_result(source, tmp_path)
        return
    payload = asdict(_sampling_judge_result(source, tmp_path))
    suite._assert_self_hosted_judge_claims(payload)
    _assert_judge_claim_tampering_rejected(suite, payload)


def _sampling_judge_result(source, tmp_path):
    return vlm_eval.evaluate_vlm(
        input_path=str(source),
        output_path=str(tmp_path / "result.json"),
        task="Inspect the colors",
        rubric="Judge the visible colors, not motion.",
        backend="self-hosted",
        endpoint_url="https://example.test/v1",
        model="test-model",
        frame_selection="final",
        max_frames=1,
    )


def _assert_judge_claim_tampering_rejected(suite, payload):
    for field, value in (
        ("rubric", vlm_eval.DEFAULT_RUBRIC),
        ("provider_success", not payload["provider_success"]),
        (
            "provider_success_matches_score_gate",
            not payload["provider_success_matches_score_gate"],
        ),
    ):
        changed = copy.deepcopy(payload)
        changed[field] = value
        with pytest.raises(AssertionError):
            suite._assert_self_hosted_judge_claims(changed)


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
        "npa_vlm_eval_compatible_json_v1",
        "npa_vlm_eval_compatible_json_v2+unknown",
        "npa_vlm_eval_compatible_json_v2"
        if fenced
        else "npa_vlm_eval_compatible_json_v2+markdown-fence-v1",
    ):
        payload["evidence"]["provider"]["parser_version"] = version
        with pytest.raises(AssertionError):
            suite._assert_result_evidence(payload, config)


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


def _assert_scalar_tampering_rejected(suite, payload):
    tampered = copy.deepcopy(payload)
    tampered["evidence"]["provider"]["status_code"] = 200.0
    with pytest.raises(ValueError, match="status_code"):
        suite.validate_sampling_scalars(tampered)
    for location in ("frames", "request_manifest"):
        for field, bad_value in (
            ("source_index", False),
            ("source_count", 6.0),
            ("width", 64.0),
            ("height", "48"),
            ("byte_count", True),
            ("source_timestamp_s", False),
            ("source_timestamp_s", float("nan")),
        ):
            tampered = copy.deepcopy(payload)
            request = tampered["evidence"]["request"]
            frames = (
                request["frames"]
                if location == "frames"
                else request[location]["frames"]
            )
            frames[0][field] = bad_value
            with pytest.raises(ValueError, match=field):
                suite.validate_sampling_scalars(tampered)
    _assert_sampling_scalar_tampering_rejected(suite, payload)


def _assert_sampling_scalar_tampering_rejected(suite, payload):
    for field, bad_value in (
        ("source_count", 6.0),
        ("max_frames", "3"),
        ("selected_count", True),
        ("coverage_complete", 1),
        ("timestamps_complete", 1),
        ("selected_indices", [False]),
        ("selected_timestamps_s", [False]),
        ("selected_timestamps_s", [float("inf")]),
    ):
        tampered = copy.deepcopy(payload)
        tampered["evidence"]["request"]["request_manifest"]["sampling"][field] = (
            bad_value
        )
        with pytest.raises(ValueError, match=field):
            suite.validate_sampling_scalars(tampered)
    for field, value in (
        ("frame_count", 1.0),
        ("dry_run", 0),
        ("passed", 1),
        ("score", True),
        ("success_threshold", "0.8"),
    ):
        tampered = copy.deepcopy(payload)
        tampered[field] = value
        with pytest.raises(ValueError, match=field):
            suite.validate_sampling_scalars(tampered)
    _assert_unknown_scalars_preserved(suite, payload)


def _assert_unknown_scalars_preserved(suite, payload):
    unknown = copy.deepcopy(payload)
    request = unknown["evidence"]["request"]
    for frames in (request["frames"], request["request_manifest"]["frames"]):
        for frame in frames:
            for field in ("source_index", "source_count", "source_timestamp_s"):
                frame[field] = None
    sampling = request["request_manifest"]["sampling"]
    sampling.update(
        source_count=None,
        coverage_complete=False,
        timestamps_complete=None,
        selected_indices=[None] * len(request["frames"]),
        selected_timestamps_s=[None] * len(request["frames"]),
    )
    suite.validate_sampling_scalars(unknown)
