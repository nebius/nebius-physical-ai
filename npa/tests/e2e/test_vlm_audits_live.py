"""Exercise real VLM audits using an operator-owned, private test configuration.

Set NPA_INTEGRATION_E2E=1 and NPA_VLM_AUDIT_LIVE_CONFIG to a private JSON file
with a cases object keyed by the AUDIT_CASES below. Each case has a request
object matching its public request dataclass and an optional expectations object.
Credentials belong in the request's api_key_env environment variable, never in
the JSON. Supply new private output paths: these APIs preserve existing evidence.

The cases invoke real hosted APIs. No stub, score override, fixture score, or
response replay is used. Failed responses and failed expectations stay in the
private product artifacts. These tests establish request/response contracts and
operator-declared visual controls, not physical correctness or robot safety.
"""

from __future__ import annotations

import base64
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from npa.workbench import vlm_eval
from npa.live_verification.vlm_audit_controls import audit_controls

pytestmark = [pytest.mark.e2e, pytest.mark.token_factory_e2e]
AUDIT_CASES = ("blinded-preference",)


def _canonical_sha256(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()


def _invoke_audit_case(case: str, request: dict[str, Any]) -> Any:
    if case == "blinded-preference":
        report = vlm_eval.compare_vlm_preference(
            vlm_eval.VlmPreferenceComparisonRequest(**request)
        )
        vlm_eval.write_preference_report(asdict(report), result_uri=report.result_uri)
        return report
    raise ValueError("Unknown live audit case")


def _assert_request_evidence(request: Any) -> None:
    assert request.endpoint_role == "hosted-api"
    assert request.frames
    assert request.request_manifest_sha256 == _canonical_sha256(
        request.request_manifest
    )
    assert request.request_manifest["frames"] == [
        asdict(frame) for frame in request.frames
    ]
    for frame in request.frames:
        assert len(frame.sha256) == 64
        assert frame.byte_count > 0 and frame.width > 0 and frame.height > 0


def _assert_provider_evidence(provider: Any, expected_model: str) -> None:
    assert provider is not None
    assert provider.status_code == 200
    assert provider.returned_model == expected_model
    assert provider.finish_reason == "stop"
    assert (
        provider.raw_response_sha256
        == hashlib.sha256(provider.raw_response.encode()).hexdigest()
    )
    raw = json.loads(provider.raw_response)
    assert raw["model"] == expected_model
    assert raw["choices"][0]["finish_reason"] == "stop"
    assert raw["choices"][0]["message"].get("refusal") in (None, "")
    assert isinstance(raw.get("usage"), dict)
    assert provider.provider_request_id
    assert raw["usage"].get("prompt_tokens", 0) > 0
    assert raw["usage"].get("completion_tokens", 0) > 0


def _assert_private_local_report(report: Any) -> None:
    if report.result_uri.startswith("s3://"):
        return
    path = Path(report.result_uri)
    assert path.is_file() and not path.is_symlink()
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text()) == json.loads(json.dumps(asdict(report)))


def _preference_request_skeleton(outcome: Any) -> dict[str, Any]:
    transport = json.loads(json.dumps(outcome.transport_request))
    content = transport["messages"][0]["content"]
    assert [part.get("text") for part in content[1::2]] == ["IMAGE A", "IMAGE B"]
    assert len(content) == 5
    for part, frame in zip(content[2::2], outcome.request.frames, strict=True):
        prefix, encoded = part["image_url"]["url"].split(",", 1)
        assert prefix == "data:image/png;base64"
        data = base64.b64decode(encoded, validate=True)
        assert hashlib.sha256(data).hexdigest() == frame.sha256
        assert len(data) == frame.byte_count
        with Image.open(BytesIO(data)) as image:
            assert image.size == (frame.width, frame.height)
            assert image.info == {}
        part["image_url"]["url"] = "<normalized-image>"
    text = json.dumps(transport).casefold()
    assert "baseline" not in text and "candidate" not in text
    return transport


def _assert_preference(report: Any, request: dict[str, Any]) -> None:
    assert report.deployment_status == "audit_only"
    assert report.operational_rate_estimated is False
    assert report.requests_counterbalanced is True
    assert not {"passed", "score", "mean_score"}.intersection(asdict(report))
    first, second = report.first_order, report.reversed_order
    for outcome in (first, second):
        assert outcome.error is None and outcome.verdict is not None
        _assert_request_evidence(outcome.request)
        _assert_provider_evidence(outcome.provider, request["model"])
        assert outcome.transport_request_sha256 == _canonical_sha256(
            outcome.transport_request
        )
    assert _preference_request_skeleton(first) == _preference_request_skeleton(second)
    assert [frame.sha256 for frame in first.request.frames] == [
        frame.sha256 for frame in reversed(second.request.frames)
    ]
    assert [frame.sha256 for frame in first.request.frames] == [
        report.normalized_baseline_sha256,
        report.normalized_candidate_sha256,
    ]
    if any(outcome.verdict.confidence != "high" for outcome in (first, second)):
        assert report.escalation_required is True
        assert report.agreement_eligible is False
    if report.mapped_preferences[0] != report.mapped_preferences[1]:
        assert report.escalation_required is True
        assert report.agreement_eligible is False


def _assert_expected_controls(report: Any, expected: dict[str, Any]) -> None:
    payload = json.loads(json.dumps(asdict(report)))
    for field, value in expected.items():
        actual: Any = payload
        for component in field.split("."):
            actual = actual[component]
        assert actual == value, f"Frozen visual expectation was not met: {field}"


def _audit_parameters() -> list[tuple[str, str]]:
    path = os.environ.get("NPA_VLM_AUDIT_LIVE_CONFIG", "")
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or not path:
        return [(case, "operator") for case in AUDIT_CASES]
    config = json.loads(Path(path).read_text())
    return [
        (case, control)
        for case in AUDIT_CASES
        for control in audit_controls(config["cases"][case])
    ]


@pytest.mark.parametrize("case,control", _audit_parameters())
def test_real_hosted_audit_contract(case: str, control: str) -> None:
    config_path = os.environ.get("NPA_VLM_AUDIT_LIVE_CONFIG", "")
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or not config_path:
        pytest.skip("provide an operator-owned NPA_VLM_AUDIT_LIVE_CONFIG")
    config = json.loads(Path(config_path).read_text())
    configured = audit_controls(config["cases"][case])[control]
    request = asdict(vlm_eval.VlmPreferenceComparisonRequest(**configured["request"]))
    for arm in ("baseline", "candidate"):
        expected_hash = configured.get(f"{arm}_sha256")
        if expected_hash is not None:
            assert (
                hashlib.sha256(Path(request[f"{arm}_path"]).read_bytes()).hexdigest()
                == expected_hash
            )
    report = _invoke_audit_case(case, request)
    _assert_private_local_report(report)
    _assert_preference(report, request)
    _assert_expected_controls(report, configured.get("expectations", {}))
