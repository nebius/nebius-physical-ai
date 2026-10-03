"""Check frozen sampler controls and schema-v2 evidence with hosted inference."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import pytest

from npa.workbench import vlm_eval


pytestmark = pytest.mark.token_factory_e2e


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _configuration(case: str, strategy: str) -> dict:
    source = os.environ.get("NPA_VLM_SAMPLING_LIVE_CONFIG", "")
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or not source:
        pytest.skip("provide a private frozen NPA_VLM_SAMPLING_LIVE_CONFIG")
    return json.loads(Path(source).read_text())["cases"][f"{case}-{strategy}"]


def _record_transport(monkeypatch, destination: Path) -> None:
    original = vlm_eval._post_with_readiness_retry

    def post(**kwargs):
        request = kwargs["request"]
        body = _canonical(request)
        (destination / "request.json").write_bytes(body)
        kwargs["request_body"] = body

        def retain(response):
            payload = asdict(response)
            payload["raw_body_sha256"] = _digest(response.raw_body.encode())
            (destination / "response.json").write_bytes(_canonical(payload))

        kwargs["response_sink"] = retain
        return original(**kwargs)

    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", post)


def _assert_evidence(result, configuration: dict) -> None:
    evidence = result.evidence
    assert evidence is not None
    assert evidence.schema_version == "npa_vlm_eval_evidence_v2"
    request = evidence.request
    assert request.request_manifest_sha256 == _digest(
        vlm_eval._canonical_json(request.request_manifest).encode()
    )
    sampling = request.request_manifest["sampling"]
    expected_indices = configuration["selected_indices"]
    assert sampling["strategy"] == configuration["request"]["frame_selection"]
    assert sampling["max_frames"] == 3
    assert sampling["selected_indices"] == expected_indices
    assert sampling["source_count"] == 6
    assert [frame.source_index for frame in request.frames] == expected_indices
    assert [frame.sha256 for frame in request.frames] == configuration["frame_hashes"]
    provider = evidence.provider
    assert provider.status_code == 200
    assert provider.returned_model == result.model
    assert provider.finish_reason == "stop"
    assert provider.provider_request_id
    assert provider.raw_response_sha256 == _digest(provider.raw_response.encode())
    assert result.passed == (result.score >= result.success_threshold)


@pytest.mark.parametrize("strategy", ["sequence", "keyframes"])
@pytest.mark.parametrize("case", ["complete", "incomplete", "gray"])
def test_frozen_hosted_sampler_controls(monkeypatch, case: str, strategy: str) -> None:
    configuration = _configuration(case, strategy)
    destination = Path(configuration["artifacts"])
    previous = os.umask(0o077)
    try:
        destination.mkdir(mode=0o700, parents=True)
        _record_transport(monkeypatch, destination)
        result = vlm_eval.evaluate_vlm(**configuration["request"])
        vlm_eval.write_result(result, result_uri=result.result_uri)
    finally:
        os.umask(previous)
    _assert_evidence(result, configuration)
    assert result.passed is configuration["expected_passed"], (
        f"Frozen {case}/{strategy} control disagreed: score={result.score}"
    )
