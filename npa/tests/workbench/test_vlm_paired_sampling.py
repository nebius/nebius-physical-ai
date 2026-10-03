"""Bind paired judge provenance to the actual shared sampling and payload bytes."""

import base64
from dataclasses import asdict
import hashlib
from io import BytesIO
import json

from PIL import Image
import pytest

from npa.workbench import vlm_eval


def _completion(model):
    content = json.dumps({"success": True, "score": 1.0, "rationale": "visible frames"})
    return {
        "model": model,
        "choices": [{"finish_reason": "stop", "message": {"content": content}}],
    }


def _recorded_pair(monkeypatch, tmp_path, strategy, cap):
    source = tmp_path / "source"
    source.mkdir()
    for index in range(5):
        Image.new("RGB", (16, 12), (index * 40, 100, 0)).save(source / f"{index}.png")
    requests = []

    def post(**kwargs):
        request = kwargs["request"]
        requests.append(request)
        return _completion(request["model"])

    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **_: "synthetic-key")
    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", post)
    report = vlm_eval.compare_vlm_judges(
        vlm_eval.VlmJudgeComparisonRequest(
            input_path=str(source),
            output_path=str(tmp_path / "report"),
            primary_model="first/model",
            secondary_model="second/model",
            frame_selection=strategy,
            max_frames=cap,
        )
    )
    return report, requests


def _assert_transported_frames(request, evidence, indices):
    content = request["messages"][1]["content"]
    images = [
        part["image_url"]["url"] for part in content if part["type"] == "image_url"
    ]
    assert len(images) == len(evidence.frames) == len(indices)
    for uri, frame, index in zip(images, evidence.frames, indices, strict=True):
        payload = base64.b64decode(uri.split(",", 1)[1], validate=True)
        assert hashlib.sha256(payload).hexdigest() == frame.sha256
        assert len(payload) == frame.byte_count
        with Image.open(BytesIO(payload)) as image:
            assert image.size == (frame.width, frame.height) == (16, 12)
            assert image.getpixel((0, 0)) == (index * 40, 100, 0)


@pytest.mark.parametrize(
    ("strategy", "cap", "indices"),
    [("final", 7, [4]), ("keyframes", 3, [0, 2, 4]), ("sequence", 2, [0, 4])],
)
def test_paired_manifests_keep_actual_strategy_cap_and_transported_indices(
    monkeypatch, tmp_path, strategy, cap, indices
):
    report, requests = _recorded_pair(monkeypatch, tmp_path, strategy, cap)
    assert report.passed is True and report.escalation_required is False
    for outcome, request in zip(
        (report.primary, report.secondary), requests, strict=True
    ):
        assert outcome.error is None and outcome.result is not None
        evidence = outcome.result.evidence.request
        manifest = evidence.request_manifest
        assert manifest["schema_version"] == "npa_vlm_eval_evidence_v2"
        assert manifest["sampling"] == {
            "strategy": strategy,
            "max_frames": cap,
            "selected_count": len(indices),
            "source_kind": "image-sequence",
            "source_count": 5,
            "selected_indices": indices,
            "selected_timestamps_s": [None] * len(indices),
            "coverage_complete": True,
            "timestamps_complete": None,
        }
        assert manifest["frames"] == [asdict(frame) for frame in evidence.frames]
        assert vlm_eval._sha256_json(manifest) == evidence.request_manifest_sha256
        _assert_transported_frames(request, evidence, indices)
    assert {key: value for key, value in requests[0].items() if key != "model"} == {
        key: value for key, value in requests[1].items() if key != "model"
    }
