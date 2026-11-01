"""Compose canonical VLM writes, call disclosures, and promotion integrity."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import runpy

import httpx
from PIL import Image
import pytest

from npa.workbench import vlm_eval
from npa.workflows import data_factory_stages
from npa.workflows.vlm_grade_evidence import vlm_grade_block_details


@pytest.mark.parametrize("body", [b'{"control":"utf8"}', b"\xff\xfecontrol"])
def test_live_recorder_retains_original_bytes_before_decoding(
    monkeypatch, tmp_path, body
):
    import hashlib

    response = httpx.Response(200, content=body)
    monkeypatch.setattr(httpx.Client, "post", lambda *_, **__: response)
    lane = runpy.run_path(
        str(Path(__file__).parents[1] / "e2e/test_vlm_disclosures_live_e2e.py")
    )
    records = lane["_record_transport"](monkeypatch, tmp_path)
    with httpx.Client() as client:
        client.post("https://example.test/v1", json={"model": "synthetic"})
    assert len(records) == 1
    record = records[0]
    assert (tmp_path / record["raw_response_bytes_file"]).read_bytes() == body
    assert record["response_sha256"] == hashlib.sha256(body).hexdigest()


def test_live_recorder_refuses_existing_raw_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(
        httpx.Client, "post", lambda *_, **__: httpx.Response(200, content=b"new")
    )
    prior = tmp_path / "transport-01.response.bin"
    prior.write_bytes(b"retained failure")
    lane = runpy.run_path(
        str(Path(__file__).parents[1] / "e2e/test_vlm_disclosures_live_e2e.py")
    )
    records = lane["_record_transport"](monkeypatch, tmp_path)
    with httpx.Client() as client, pytest.raises(FileExistsError):
        client.post("https://example.test/v1", json={"model": "synthetic"})
    assert prior.read_bytes() == b"retained failure"
    assert records == []


@pytest.mark.parametrize("positional_count", [11, 20, 23])
def test_result_disclosures_append_after_inherited_constructor_fields(positional_count):
    # The reviewed model-identity parent has20 fields; our three disclosure
    # additions must not silently reinterpret valid inherited positional values.
    values = {
        "status": "passed",
        "backend": "api",
        "input_path": "input",
        "output_path": "output",
        "result_uri": "result",
        "task": "task",
        "model": "fixture-model",
        "score": 0.9,
        "success_threshold": 0.8,
        "passed": True,
        "generated_at": "2026-10-03T00:00:00Z",
        "frame_selection": "final",
        "frame_count": 1,
        "rationale": "fixture",
        "served_model": "fixture-model",
        "evidence": None,
        "rubric": "fixture",
        "provider_success": True,
        "provider_success_matches_score_gate": False,
        "served_model_match_enforced": True,
        "independent_human_label_calibration_established": False,
        "limitations": ("fixture only",),
        "provider_call_made": True,
    }
    selected = dict(list(values.items())[:positional_count])
    positional = vlm_eval.VlmEvalResult(*selected.values())
    assert asdict(positional) == asdict(vlm_eval.VlmEvalResult(**selected))
    for name, value in selected.items():
        assert getattr(positional, name) == value


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
                                    "provider_observable_support": [
                                        "Frame 1 visibly contains a green image.",
                                        "Frame 1 contains no visible red region.",
                                        "Frame 1 contains no visible red region.",
                                    ],
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
    assert report["provider_observable_support"] == [
        "Frame 1 visibly contains a green image.",
        "Frame 1 contains no visible red region.",
        "Frame 1 contains no visible red region.",
    ]
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


@pytest.mark.parametrize(
    "mutation", ["missing", "changed", "reordered", "scalar", "tuple"]
)
def test_promotion_rejects_rebound_provider_support(called_report, mutation):
    report, _ = called_report
    if mutation == "missing":
        del report["provider_observable_support"]
    elif mutation == "changed":
        report["provider_observable_support"] = ["Frame 1 is red."]
    elif mutation == "reordered":
        report["provider_observable_support"].reverse()
    elif mutation == "tuple":
        report["provider_observable_support"] = tuple(
            report["provider_observable_support"]
        )
    else:
        report["provider_observable_support"] = "Frame 1 is green."
    assert vlm_grade_block_details(report) == {
        "reason": "vlm_provider_evidence_invalid",
        "evidence_reason": "result_mismatch",
    }


def test_promotion_rejects_deduplicated_serialized_support(called_report):
    report, output = called_report
    provider = report["evidence"]["provider"]
    raw_response = provider["raw_response"]
    raw_response_sha256 = provider["raw_response_sha256"]
    raw = json.loads(raw_response)
    raw_support = json.loads(raw["choices"][0]["message"]["content"])[
        "provider_observable_support"
    ]
    assert raw_support == [
        "Frame 1 visibly contains a green image.",
        "Frame 1 contains no visible red region.",
        "Frame 1 contains no visible red region.",
    ]

    report["provider_observable_support"] = list(
        dict.fromkeys(report["provider_observable_support"])
    )
    assert report["provider_observable_support"] == [
        "Frame 1 visibly contains a green image.",
        "Frame 1 contains no visible red region.",
    ]
    assert provider["raw_response"] == raw_response
    assert provider["raw_response_sha256"] == raw_response_sha256
    assert vlm_grade_block_details(report) == {
        "reason": "vlm_provider_evidence_invalid",
        "evidence_reason": "result_mismatch",
    }

    (output / vlm_eval.RESULT_FILENAME).write_text(json.dumps(report))
    assert (
        data_factory_stages.grade_gate(str(output), str(output / "decision.json"))
        == "loop_back"
    )


def test_promotion_rejects_format_only_provider_support(called_report):
    report, output = called_report
    provider = report["evidence"]["provider"]
    raw = json.loads(provider["raw_response"])
    content = json.loads(raw["choices"][0]["message"]["content"])
    blank_support = [" \t\u200b\ufeff\u200d\n"]
    content["provider_observable_support"] = blank_support
    raw["choices"][0]["message"]["content"] = json.dumps(content, ensure_ascii=False)
    provider["raw_response"] = json.dumps(
        raw, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    provider["raw_response_sha256"] = vlm_eval._sha256_text(provider["raw_response"])
    report["provider_observable_support"] = blank_support

    assert vlm_grade_block_details(report) == {
        "reason": "vlm_provider_evidence_invalid",
        "evidence_reason": "provider_verdict_invalid",
    }
    (output / vlm_eval.RESULT_FILENAME).write_text(json.dumps(report))
    assert (
        data_factory_stages.grade_gate(str(output), str(output / "decision.json"))
        == "loop_back"
    )


def test_promotion_classifies_old_three_field_parser_before_reparse(called_report):
    report, _ = called_report
    provider = report["evidence"]["provider"]
    raw = json.loads(provider["raw_response"])
    content = json.loads(raw["choices"][0]["message"]["content"])
    del content["provider_observable_support"]
    raw["choices"][0]["message"]["content"] = json.dumps(content)
    provider["raw_response"] = json.dumps(raw, separators=(",", ":"), sort_keys=True)
    provider["raw_response_sha256"] = vlm_eval._sha256_text(provider["raw_response"])
    provider["parser_version"] = "npa_vlm_eval_hosted_json_v1"
    assert vlm_grade_block_details(report) == {
        "reason": "vlm_provider_evidence_invalid",
        "evidence_reason": "provider_parser_obsolete",
    }


def test_promotion_rejects_raw_support_rebound_from_serialized_result(called_report):
    report, _ = called_report
    provider = report["evidence"]["provider"]
    raw = json.loads(provider["raw_response"])
    content = json.loads(raw["choices"][0]["message"]["content"])
    content["provider_observable_support"] = ["Frame 1 is red."]
    raw["choices"][0]["message"]["content"] = json.dumps(content)
    provider["raw_response"] = json.dumps(raw, separators=(",", ":"), sort_keys=True)
    provider["raw_response_sha256"] = vlm_eval._sha256_text(provider["raw_response"])
    assert vlm_grade_block_details(report) == {
        "reason": "vlm_provider_evidence_invalid",
        "evidence_reason": "result_mismatch",
    }


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
