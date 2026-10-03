"""Verify disclosure persistence around real hosted visual controls."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import httpx
from PIL import Image, ImageDraw
import pytest

from npa.clients.token_factory import (
    DEFAULT_VISION_MODEL,
    TokenFactoryClient,
    resolve_config,
)
from npa.workbench.vlm_eval import (
    benchmark_vlm_eval,
    evaluate_rollout_set,
    evaluate_vlm,
)

pytestmark = [pytest.mark.e2e, pytest.mark.token_factory_e2e]
_TASK = (
    "Identify the visible objects first. Score whether a single green disk is "
    "centered on a white background. A red disk or a blank image does not meet "
    "this visual target."
)
_RUBRIC = (
    "Use 1.0 only when the stated visual target is clearly visible, 0.0 when it "
    "is absent, and intermediate values for uncertainty. These synthetic "
    "controls do not demonstrate physical task completion or model calibration."
)


def _private_evidence(tmp_path: Path) -> Path:
    destination = os.environ.get("NPA_VLM_DISCLOSURE_EVIDENCE_DIR")
    evidence = Path(destination) if destination else tmp_path / "evidence"
    evidence.mkdir(parents=True, mode=0o700)
    return evidence


def _record_transport(monkeypatch, evidence: Path) -> list[dict]:
    original = httpx.Client.post
    records = []

    def post(client, url, *args, **kwargs):
        response = original(client, url, *args, **kwargs)
        raw_name = f"transport-{len(records) + 1:02d}.response.bin"
        with (evidence / raw_name).open("xb") as raw:
            raw.write(response.content)
        record = {
            "request": kwargs.get("json"),
            "http_status": response.status_code,
            "raw_response": response.text,
            "response_sha256": hashlib.sha256(response.content).hexdigest(),
            "raw_response_bytes_file": raw_name,
        }
        records.append(record)
        (evidence / f"transport-{len(records):02d}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        return response

    monkeypatch.setattr(httpx.Client, "post", post)
    return records


def _controls(evidence: Path) -> tuple[Path, Path]:
    rollouts = evidence / "rollouts"
    items = []
    for name, color, expected in (
        ("green", "green", True),
        ("red", "red", False),
        ("blank", None, False),
    ):
        directory = rollouts / name
        directory.mkdir(parents=True)
        image = Image.new("RGB", (192, 192), "white")
        if color:
            ImageDraw.Draw(image).ellipse((48, 48, 144, 144), fill=color)
        image.save(directory / "frame.png")
        items.append(
            {
                "id": name,
                "rollout": str(directory),
                "expected_label": expected,
                "task": _TASK,
            }
        )
    manifest = evidence / "benchmark.json"
    manifest.write_text(json.dumps({"items": items}) + "\n")
    (evidence / "protocol.json").write_text(
        json.dumps(
            {
                "task": _TASK,
                "rubric": _RUBRIC,
                "threshold": 0.8,
                "model": DEFAULT_VISION_MODEL,
                "items": items,
                "limitations": "Synthetic visual controls; caller labels are not independent human calibration.",
            },
            indent=2,
        )
        + "\n"
    )
    return rollouts, manifest


def test_live_disclosures_preserve_real_and_no_call_provenance(monkeypatch, tmp_path):
    if not resolve_config(require_api_key=False).api_key:
        pytest.skip("Token Factory credential is required")
    assert DEFAULT_VISION_MODEL in TokenFactoryClient().list_models()
    evidence = _private_evidence(tmp_path)
    rollouts, manifest = _controls(evidence)
    transports = _record_transport(monkeypatch, evidence)
    loop = evaluate_rollout_set(
        input_path=str(rollouts),
        output_path=str(evidence / "loop"),
        task=_TASK,
        rubric=_RUBRIC,
        backend="api",
        model=DEFAULT_VISION_MODEL,
        success_threshold=0.8,
    )
    (evidence / "loop.json").write_text(json.dumps(loop, indent=2) + "\n")
    benchmark = benchmark_vlm_eval(
        dataset=str(manifest),
        backend="api",
        models=[DEFAULT_VISION_MODEL],
        thresholds=[0.8],
        rubrics=[_RUBRIC],
    )
    (evidence / "benchmark-report.json").write_text(
        json.dumps(asdict(benchmark), indent=2) + "\n"
    )
    override = evaluate_vlm(
        input_path="unused",
        output_path=str(evidence / "override"),
        backend="api",
        model=DEFAULT_VISION_MODEL,
        score=0.9,
    )
    (evidence / "override.json").write_text(
        json.dumps(asdict(override), indent=2) + "\n"
    )
    _assert_disclosures(evidence, loop, benchmark, override, transports)


def _assert_disclosures(evidence, loop, benchmark, override, transports):
    assert len(transports) == 6
    assert all(record["http_status"] == 200 for record in transports)
    assert loop["independent_human_label_calibration_established"] is False
    assert "mean-score gate" in loop["limitations"][0]
    assert loop["task_success"] == (loop["mean_score"] >= 0.8)
    assert benchmark.independent_human_label_calibration_established is False
    assert len(benchmark.limitations) == 3
    assert override.evidence is None
    assert override.provider_call_made is False
    assert "no VLM call occurred" in override.limitations[-1]
    for result in (evidence / "loop/rollouts").glob("*/*.json"):
        payload = json.loads(result.read_text())
        assert payload["independent_human_label_calibration_established"] is False
        assert payload["provider_call_made"] is True
        assert len(payload["limitations"]) == 2
        _assert_v2_sampling(payload["evidence"])
        provider = payload["evidence"]["provider"]
        assert (
            provider["raw_response_sha256"]
            == hashlib.sha256(provider["raw_response"].encode()).hexdigest()
        )
    predictions = {
        case.item_id: case.predicted_label for case in benchmark.best_config.results
    }
    assert predictions == {"green": True, "red": False, "blank": False}
    for case in benchmark.best_config.results:
        _assert_v2_sampling(asdict(case.evidence))


def _assert_v2_sampling(evidence):
    assert evidence["schema_version"] == "npa_vlm_eval_evidence_v2"
    request = evidence["request"]
    manifest = request["request_manifest"]
    assert manifest["sampling"] == {
        "strategy": "keyframes",
        "max_frames": 4,
        "selected_count": 1,
        "source_kind": "image-sequence",
        "source_count": 1,
        "selected_indices": [0],
        "selected_timestamps_s": [None],
        "coverage_complete": True,
        "timestamps_complete": None,
    }
    assert request["frames"][0]["source_index"] == 0
    assert request["frames"][0]["source_count"] == 1
    assert request["frames"][0]["source_kind"] == "image-sequence"
    assert (
        request["request_manifest_sha256"]
        == hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
