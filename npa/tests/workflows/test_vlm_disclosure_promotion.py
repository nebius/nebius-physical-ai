"""Compose canonical VLM writes, call disclosures, and promotion integrity."""

from __future__ import annotations

from dataclasses import asdict
import json

import httpx
from PIL import Image
import pytest

from npa.workbench import vlm_eval
from npa.workflows import data_factory_stages
from npa.workflows.vlm_grade_evidence import vlm_grade_block_details


def _synthetic_provider_post(calls):
    def post(client, url, *args, **kwargs):
        calls.append(kwargs["json"])
        return httpx.Response(
            200,
            json={
                "model": "MiniMaxAI/MiniMax-M3",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "score": 0.9,
                                    "success": False,
                                    "rationale": "Synthetic transport control only.",
                                }
                            )
                        },
                    }
                ],
            },
            request=httpx.Request("POST", "https://example.test/v1/chat/completions"),
        )

    return post


@pytest.fixture
def called_report(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (32, 32), "green").save(source / "frame.png")
    calls = []
    monkeypatch.setattr(httpx.Client, "post", _synthetic_provider_post(calls))
    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **_: "synthetic-key")
    monkeypatch.setattr(
        vlm_eval, "_resolve_endpoint_url", lambda **_: "https://example.test/v1"
    )
    output = tmp_path / "grade"
    result = vlm_eval.evaluate_vlm(
        input_path=str(source),
        output_path=str(output),
        backend="api",
        model="MiniMaxAI/MiniMax-M3",
        task="Identify the green frame.",
        success_threshold=0.8,
        frame_selection="keyframes",
        max_frames=4,
    )
    assert len(calls) == 1
    vlm_eval.write_result(asdict(result), result_uri=result.result_uri)
    assert (output / vlm_eval.RESULT_FILENAME).is_file()
    assert not (output / vlm_eval.LEGACY_RESULT_FILENAME).exists()
    report = json.loads((output / vlm_eval.RESULT_FILENAME).read_text())
    assert report == json.loads(json.dumps(asdict(result)))
    return report, output


def test_canonical_provider_call_preserves_disclosures_and_strict_gate(called_report):
    report, output = called_report
    assert report["provider_call_made"] is True
    assert report["independent_human_label_calibration_established"] is False
    assert report["limitations"]
    assert report["provider_success"] is False
    assert report["provider_success_matches_score_gate"] is False
    assert report["passed"] is True
    assert report["score"] == 0.9
    sampling = report["evidence"]["request"]["request_manifest"]["sampling"]
    assert sampling["max_frames"] == 4
    assert sampling["selected_indices"] == [0]
    assert vlm_grade_block_details(report) == {}
    assert (
        data_factory_stages.grade_gate(str(output), str(output / "decision.json"))
        == "promote_checkpoint"
    )


@pytest.mark.parametrize("claimed_call", [False, None, 0, 1, "true", "false"])
def test_promotion_rejects_nonliteral_or_no_call_disclosure(
    called_report, claimed_call
):
    report, output = called_report
    # Even copied internally consistent evidence cannot override an explicit
    # no-call or nonliteral call-status disclosure. Historical absent keys remain
    # separately compatible, and this is not provider authentication.
    report["provider_call_made"] = claimed_call
    (output / vlm_eval.RESULT_FILENAME).write_text(json.dumps(report))
    assert vlm_grade_block_details(report) == {"reason": "vlm_non_inference_backend"}
    assert (
        data_factory_stages.grade_gate(str(output), str(output / "decision.json"))
        == "loop_back"
    )


def test_promotion_retains_legacy_absent_call_disclosure_compatibility(called_report):
    report, _ = called_report
    del report["provider_call_made"]
    assert vlm_grade_block_details(report) == {}


@pytest.mark.parametrize("backend", ["api", "stub"])
def test_canonical_override_is_no_call_uncalibrated_and_cannot_promote(
    tmp_path, monkeypatch, backend
):
    def unexpected_call(*args, **kwargs):
        raise AssertionError("A score override must not call the provider")

    monkeypatch.setattr(httpx.Client, "post", unexpected_call)
    output = tmp_path / "grade"
    result = vlm_eval.evaluate_vlm(
        input_path="unused",
        output_path=str(output),
        backend=backend,
        score=0.9,
    )
    assert result.provider_call_made is False
    assert result.independent_human_label_calibration_established is False
    assert result.evidence is None
    assert "no VLM call occurred" in result.limitations[-1]
    vlm_eval.write_result(asdict(result), result_uri=result.result_uri)
    assert (output / vlm_eval.RESULT_FILENAME).is_file()
    assert not (output / vlm_eval.LEGACY_RESULT_FILENAME).exists()
    assert (
        data_factory_stages.grade_gate(str(output), str(output / "decision.json"))
        == "loop_back"
    )
