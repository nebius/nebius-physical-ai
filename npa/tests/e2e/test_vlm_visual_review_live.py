"""Check frozen rich-review controls through a real hosted vision provider."""

from __future__ import annotations

import base64
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import pytest

from npa.workbench.vlm_eval import VlmVisualReviewRequest, review_visual


pytestmark = pytest.mark.token_factory_e2e


def _digest(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _frozen_configuration() -> dict:
    source = os.environ.get("NPA_VLM_VISUAL_REVIEW_LIVE_CONFIG", "")
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or not source:
        pytest.skip("provide a private frozen NPA_VLM_VISUAL_REVIEW_LIVE_CONFIG")
    return json.loads(Path(source).read_text())


def _assert_bound_outcome(outcome, model: str) -> None:
    assert outcome.transport_request_sha256 == _digest(outcome.transport_request)
    assert outcome.request.request_manifest_sha256 == _digest(
        outcome.request.request_manifest
    )
    provider = outcome.provider
    assert provider is not None
    assert provider.status_code == 200
    assert provider.returned_model == model
    assert provider.finish_reason == "stop"
    assert provider.provider_request_id
    assert (
        provider.raw_response_sha256
        == hashlib.sha256(provider.raw_response.encode()).hexdigest()
    )
    assert outcome.error is None
    assert outcome.verdict is not None
    wire = outcome.response_bytes
    assert wire is not None
    body = base64.b64decode(wire.body_base64, validate=True)
    assert wire.body_sha256 == hashlib.sha256(body).hexdigest()
    assert wire.byte_count == len(body)
    assert wire.status_code == 200


@pytest.mark.parametrize("case", ["complete-vs-incomplete", "gray-absence"])
def test_frozen_hosted_visual_review(case: str) -> None:
    configuration = _frozen_configuration()["cases"][case]
    request = VlmVisualReviewRequest(**configuration["request"])
    previous = os.umask(0o077)
    try:
        report = review_visual(request)
    finally:
        os.umask(previous)
    assert report.deployment_status == "audit_only"
    assert report.score_gate_affected is False
    assert report.normalized_task_completion_score is None
    assert report.attempt_count == (2 if request.baseline_path else 1)
    assert Path(report.result_uri).stat().st_mode & 0o777 == 0o600
    for outcome in report.outcomes:
        _assert_bound_outcome(outcome, request.model)
    payload = asdict(report)
    for field, expected in configuration["expectations"].items():
        actual = payload
        for component in field.split("."):
            actual = actual[component]
        assert actual == expected, f"Frozen visual expectation failed: {field}"
