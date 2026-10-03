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

pytestmark = pytest.mark.token_factory_e2e


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
    dataset = DEFAULT_SAMPLE_BENCHMARK_PATH
    protocol = {
        "model": DEFAULT_VISION_MODEL,
        "thresholds": [0.8],
        "rubrics": ["default"],
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "use_fixture_scores": False,
        "limitations": "Original physical-task labels are unverified on 2x2 swatches; measured metrics are illustrative only.",
    }
    (evidence / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


def test_live_benchmark_sample_scope_does_not_imply_task_validation(
    monkeypatch, tmp_path
):
    if not resolve_config(require_api_key=False).api_key:
        pytest.skip("Token Factory credential is required")
    assert DEFAULT_VISION_MODEL in TokenFactoryClient().list_models()
    evidence = _evidence_path(tmp_path)
    _frozen_protocol(evidence)
    transports = _record_transport(monkeypatch, evidence)
    report = benchmark_vlm_eval(
        dataset="sample",
        backend="api",
        models=[DEFAULT_VISION_MODEL],
        thresholds=[0.8],
        rubrics=["default"],
        use_fixture_scores=False,
    )
    payload = asdict(report)
    (evidence / "hosted-report.json").write_text(json.dumps(payload, indent=2) + "\n")
    fixture = benchmark_vlm_eval(
        dataset="sample", backend="stub", thresholds=[0.8], rubrics=["default"]
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
