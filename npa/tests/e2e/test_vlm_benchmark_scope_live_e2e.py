"""Verify illustrative scope around the unchanged fixture and real responses."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import httpx
import pytest

from npa.clients.token_factory import (
    DEFAULT_VISION_MODEL,
    TokenFactoryClient,
    resolve_config,
)
from npa.workbench.vlm_eval import (
    DEFAULT_SAMPLE_BENCHMARK_PATH,
    VlmEvalError,
    benchmark_vlm_eval,
)

pytestmark = [pytest.mark.e2e, pytest.mark.token_factory_e2e]
FROZEN_SCOPE_PROTOCOL_PATH = DEFAULT_SAMPLE_BENCHMARK_PATH.with_name(
    "scope_protocol_v1.json"
)
FROZEN_SCOPE_PROTOCOL_SHA256 = (
    "25dee38f69e05f6b1b236b8a2a9c598aa219fdc54a1525eb65a8abeac5f31595"
)


def _evidence_path(tmp_path: Path) -> Path:
    configured = os.environ.get("NPA_BENCHMARK_SCOPE_EVIDENCE_DIR")
    path = Path(configured) if configured else tmp_path / "evidence"
    path.mkdir(parents=True, mode=0o700)
    return path


def _record_transport(monkeypatch, evidence):
    original = httpx.Client.post
    records = []

    def post(client, url, *args, **kwargs):
        response = original(client, url, *args, **kwargs)
        record = {
            "request": kwargs.get("json"),
            "http_status": response.status_code,
            "raw_response": response.text,
            "response_sha256": hashlib.sha256(response.content).hexdigest(),
        }
        records.append(record)
        (evidence / f"transport-{len(records):02d}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        return response

    monkeypatch.setattr(httpx.Client, "post", post)
    return records


def _frozen_protocol(evidence):
    # Keep the original four-item protocol byte-for-byte. The evolving bundled
    # default is a distinct five-item wiring fixture, not a substitute protocol.
    dataset = FROZEN_SCOPE_PROTOCOL_PATH
    dataset_sha256 = hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert dataset_sha256 == FROZEN_SCOPE_PROTOCOL_SHA256
    protocol = {
        "model": DEFAULT_VISION_MODEL,
        "thresholds": [0.8],
        "rubrics": ["default"],
        "dataset_sha256": dataset_sha256,
        "protocol_manifest_original_execution_sha": (
            "a01dda9c915a731e13bec6d72a5affcf9d49745d"
        ),
        "request_scope": "Current runtime prompt and image anchors; not the original request payload.",
        "use_fixture_scores": False,
        "limitations": "Original physical-task labels are unverified on 2x2 swatches; measured metrics are illustrative only.",
    }
    (evidence / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return dataset


def test_live_benchmark_sample_scope_does_not_imply_task_validation(
    monkeypatch, tmp_path
):
    if not resolve_config(require_api_key=False).api_key:
        pytest.skip("Token Factory credential is required")
    assert DEFAULT_VISION_MODEL in TokenFactoryClient().list_models()
    evidence = _evidence_path(tmp_path)
    dataset = _frozen_protocol(evidence)
    transports = _record_transport(monkeypatch, evidence)
    report = benchmark_vlm_eval(
        dataset=str(dataset),
        backend="api",
        models=[DEFAULT_VISION_MODEL],
        thresholds=[0.8],
        rubrics=["default"],
        use_fixture_scores=False,
    )
    payload = asdict(report)
    (evidence / "hosted-report.json").write_text(json.dumps(payload, indent=2) + "\n")
    fixture = benchmark_vlm_eval(
        dataset=str(dataset), backend="stub", thresholds=[0.8], rubrics=["default"]
    )
    (evidence / "fixture-report.json").write_text(
        json.dumps(asdict(fixture), indent=2) + "\n"
    )
    assert len(transports) == report.item_count == 4
    assert report.dataset_evidence_scope == "illustrative_only"
    assert len(report.dataset_limitations) == 3
    assert report.independent_human_label_calibration_established is False
    assert fixture.dataset_limitations == report.dataset_limitations
    _assert_real_evidence(report, transports)
    _assert_invalid_scope_fails_without_call(evidence, transports)


def _assert_real_evidence(report, transports):
    for case, record in zip(report.best_config.results, transports):
        assert case.score_source == "api"
        assert case.evidence is not None
        provider = case.evidence.provider
        assert provider.status_code == record["http_status"] == 200
        assert provider.finish_reason == "stop"
        assert provider.returned_model == DEFAULT_VISION_MODEL
        assert provider.raw_response_sha256 == record["response_sha256"]
        assert case.evidence.schema_version == "npa_vlm_eval_evidence_v2"
        sampling = case.evidence.request.request_manifest["sampling"]
        assert sampling["strategy"] == "keyframes"
        assert sampling["max_frames"] == 4
        assert sampling["source_kind"] == "image-sequence"
        assert sampling["source_count"] == sampling["selected_count"] == 1
        assert sampling["selected_indices"] == [0]
        assert sampling["selected_timestamps_s"] == [None]
        assert sampling["coverage_complete"] is True
        assert case.evidence.request.frames[0].source_index == 0
    assert all("fixture" not in limitation for limitation in report.limitations)


def _assert_invalid_scope_fails_without_call(evidence, transports):
    path = evidence / "invalid.json"
    path.write_text(
        json.dumps(
            {
                "evidence_scope": "calibrated",
                "items": [
                    {
                        "id": "invalid",
                        "rollout": "unused",
                        "expected_label": True,
                    }
                ],
            }
        )
        + "\n"
    )
    with pytest.raises(VlmEvalError, match="evidence_scope must be exactly one of"):
        benchmark_vlm_eval(
            dataset=str(path), backend="api", models=[DEFAULT_VISION_MODEL]
        )
    assert len(transports) == 4
