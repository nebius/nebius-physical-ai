"""Measure a frozen outcome/agency pair with real hosted model responses."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import pytest

from npa.clients.token_factory import DEFAULT_VISION_MODEL, resolve_config
from npa.workbench.vlm_eval import benchmark_vlm_eval, DEFAULT_RUBRIC


pytestmark = [pytest.mark.e2e, pytest.mark.token_factory_e2e]
PROTOCOL = {
    "dataset": "isaac-agency",
    "threshold": 0.5,
    "frame_selection": "sequence",
    "max_frames": 6,
    "rubric": DEFAULT_RUBRIC,
    "expected_labels": {
        "cube-elevated-positive": True,
        "robot-grasp-lift-negative": False,
    },
    "scope": "Stylized paired controls; no physical or safety qualification.",
}


def _models() -> list[str]:
    value = os.environ.get("NPA_VLM_AGENCY_LIVE_MODELS", DEFAULT_VISION_MODEL)
    return [model.strip() for model in value.split(",") if model.strip()]


def _digest_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize("model", _models())
def test_frozen_agency_pair_retains_real_provider_observations(
    model: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """Execute both immutable claims once and retain measured label agreement."""
    if not resolve_config(require_api_key=False).api_key:
        pytest.skip("Live agency calibration requires the configured hosted key")
    report = benchmark_vlm_eval(
        dataset=PROTOCOL["dataset"],
        thresholds=[PROTOCOL["threshold"]],
        rubrics=[PROTOCOL["rubric"]],
        models=[model],
        backend="api",
        frame_selection=PROTOCOL["frame_selection"],
        max_frames=PROTOCOL["max_frames"],
    )
    _retain_report(model, report, tmp_path, request)
    _verify_pair(report, model)


def _retain_report(
    model, report, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    payload = {
        "protocol": PROTOCOL,
        "protocol_sha256": _digest_json(PROTOCOL),
        "model": model,
        "report": asdict(report),
    }
    destination = Path(os.environ.get("NPA_VLM_AGENCY_EVIDENCE_DIR", str(tmp_path)))
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    output = destination / (hashlib.sha256(model.encode()).hexdigest() + ".json")
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    output.chmod(0o600)
    request.node.user_properties.append(
        (
            "agency_quality",
            json.dumps(asdict(report.best_config.metrics), sort_keys=True),
        )
    )


def _verify_pair(report, model: str) -> None:
    cases = report.best_config.results
    assert report.item_count == len(cases) == 2
    assert {case.item_id: case.expected_label for case in cases} == PROTOCOL[
        "expected_labels"
    ]
    frames = None
    for case in cases:
        assert case.evidence is not None
        provider = case.evidence.provider
        assert provider.returned_model == model
        assert provider.finish_reason == "stop"
        assert provider.status_code == 200
        assert (
            hashlib.sha256(provider.raw_response.encode()).hexdigest()
            == provider.raw_response_sha256
        )
        assert (
            case.evidence.request.rubric_sha256
            == hashlib.sha256(DEFAULT_RUBRIC.encode()).hexdigest()
        )
        selected = tuple(frame.sha256 for frame in case.evidence.request.frames)
        _verify_sampling(case.evidence)
        assert len(selected) == 6
        assert frames is None or frames == selected
        frames = selected
    # Label disagreement is a measured calibration result, not a transport error.
    assert (
        sum(
            (
                report.best_config.metrics.true_positives,
                report.best_config.metrics.true_negatives,
                report.best_config.metrics.false_positives,
                report.best_config.metrics.false_negatives,
            )
        )
        == 2
    )


def _verify_sampling(evidence) -> None:
    assert evidence.schema_version == "npa_vlm_eval_evidence_v2"
    frames = evidence.request.frames
    assert [frame.source_kind for frame in frames] == ["image-sequence"] * 6
    assert [frame.source_index for frame in frames] == list(range(6))
    assert [frame.source_count for frame in frames] == [6] * 6
    sampling = evidence.request.request_manifest["sampling"]
    assert sampling["strategy"] == PROTOCOL["frame_selection"]
    assert (
        sampling["max_frames"] == sampling["selected_count"] == PROTOCOL["max_frames"]
    )
    assert sampling["source_kind"] == "image-sequence"
    assert sampling["source_count"] == 6
    assert sampling["selected_indices"] == list(range(6))
    assert sampling["selected_timestamps_s"] == [None] * 6
    assert sampling["coverage_complete"] is True
