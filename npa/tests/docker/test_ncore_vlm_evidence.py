from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
import urllib.error

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))

from npa.clients.storage import StoragePreconditionFailed  # noqa: E402
from ncore_publication import vlm_evidence  # noqa: E402


class _Response:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class _Storage:
    def __init__(self):
        self.objects = {}

    def put_bytes_conditional(self, payload, uri, *, if_none_match, content_type):
        assert if_none_match is True
        assert content_type == "application/json"
        if uri in self.objects:
            raise StoragePreconditionFailed("exists")
        self.objects[uri] = (payload, f'"etag-{len(self.objects)}"')
        return self.objects[uri][1]

    def read_bytes_with_etag(self, uri):
        return self.objects.get(uri)


def _root(tmp_path: Path) -> tuple[Path, list[dict]]:
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    directory = root / "final"
    directory.mkdir(mode=0o700)
    records = []
    for index in range(4):
        path = directory / f"frame-{index:03d}.png"
        image = Image.new("RGB", (64, 64), (20 + index, 40, 60))
        image.putpixel((0, 0), (200, 10, 20))
        image.save(path)
        body = path.read_bytes()
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": vlm_evidence._sha_bytes(body),
                "bytes": len(body),
                "order": index,
            }
        )
    return root, records


def _response(*, model: str = vlm_evidence.MODEL):
    return {
        "model": model,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "success": True,
                            "score": 0.9,
                            "rationale": "Visible scene geometry remains coherent.",
                        }
                    )
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }


def test_one_shot_call_retains_exact_transport_without_labels(
    monkeypatch, tmp_path: Path
) -> None:
    root, records = _root(tmp_path)
    storage = _Storage()
    monkeypatch.setattr(
        vlm_evidence.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(_response()),
    )

    result = vlm_evidence._call_once(
        root=root,
        attempt_id="opaque-case",
        freeze_sha256="a" * 64,
        purpose="calibration",
        frame_records=records,
        task="Review visible coherence.",
        rubric="Use visible pixels only.",
        api_key="secret-not-retained",
        external_attempt_prefix="s3://private/run/vlm/",
        storage_client=storage,
    )

    assert result["served_model"] == vlm_evidence.MODEL
    assert result["score"] == 0.9
    assert result["provider_request_id"] == "unavailable"
    assert type(result["latency_ms"]) is int
    request = (root / "transport/opaque-case/request.json").read_text()
    assert "expected_label" not in request
    assert "secret-not-retained" not in request
    assert (root / "transport/opaque-case/response.json").is_file()
    assert len(storage.objects) == 1
    assert (
        vlm_evidence._verified_transport(
            root,
            attempt_id="opaque-case",
            freeze_sha256="a" * 64,
            purpose="calibration",
            frame_records=records,
            task="Review visible coherence.",
            rubric="Use visible pixels only.",
        )
        == result
    )
    with pytest.raises(vlm_evidence.VlmEvidenceError, match="retry is prohibited"):
        vlm_evidence._call_once(
            root=root,
            attempt_id="opaque-case",
            freeze_sha256="a" * 64,
            purpose="calibration",
            frame_records=records,
            task="Review visible coherence.",
            rubric="Use visible pixels only.",
            api_key="secret-not-retained",
            external_attempt_prefix="s3://private/run/vlm/",
            storage_client=storage,
        )


def test_one_shot_call_rejects_served_model_drift(monkeypatch, tmp_path: Path) -> None:
    root, records = _root(tmp_path)
    storage = _Storage()
    monkeypatch.setattr(
        vlm_evidence.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(_response(model="other/model")),
    )

    with pytest.raises(vlm_evidence.VlmEvidenceError, match="model or finish"):
        vlm_evidence._call_once(
            root=root,
            attempt_id="opaque-case",
            freeze_sha256="a" * 64,
            purpose="calibration",
            frame_records=records,
            task="Review visible coherence.",
            rubric="Use visible pixels only.",
            api_key="secret-not-retained",
            external_attempt_prefix="s3://private/run/vlm/",
            storage_client=storage,
        )


def test_block_control_is_deterministic_and_nonidentical(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    image = Image.new("RGB", (64, 64))
    for y in range(64):
        for x in range(64):
            image.putpixel((x, y), (x, y, (x + y) % 256))
    image.save(source)

    first = vlm_evidence._block_corrupt([source])
    second = vlm_evidence._block_corrupt([source])

    assert first == second
    assert first[0] != vlm_evidence._image_bytes(source)[0]
    assert vlm_evidence._indices(10) == [0, 3, 6, 9]


def test_render_defect_control_is_deterministic_and_visibly_changed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "render.png"
    image = Image.new("RGB", (96, 64))
    for y in range(64):
        for x in range(96):
            image.putpixel((x, y), (x * 2 % 256, y * 3 % 256, (x + y) % 256))
    image.save(source)
    first = vlm_evidence._render_defect_control([source])
    second = vlm_evidence._render_defect_control([source])
    assert first == second
    assert first[0] != vlm_evidence._image_bytes(source)[0]


def test_transport_verifier_rederives_outcome_from_raw_response(
    monkeypatch, tmp_path: Path
) -> None:
    root, records = _root(tmp_path)
    monkeypatch.setattr(
        vlm_evidence.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(_response()),
    )
    vlm_evidence._call_once(
        root=root,
        attempt_id="opaque-case",
        freeze_sha256="a" * 64,
        purpose="calibration",
        frame_records=records,
        task="Review visible coherence.",
        rubric="Use visible pixels only.",
        api_key="secret-not-retained",
        external_attempt_prefix="s3://private/run/vlm/",
        storage_client=_Storage(),
    )
    outcome_path = root / "transport/opaque-case/outcome.json"
    outcome = json.loads(outcome_path.read_text())
    outcome["score"] = 0.1
    outcome_path.write_text(json.dumps(outcome))
    with pytest.raises(vlm_evidence.VlmEvidenceError, match="binding differs"):
        vlm_evidence._verified_transport(
            root,
            attempt_id="opaque-case",
            freeze_sha256="a" * 64,
            purpose="calibration",
            frame_records=records,
            task="Review visible coherence.",
            rubric="Use visible pixels only.",
        )


def test_weak_rationale_is_rejected_after_external_attempt_commit(
    monkeypatch, tmp_path: Path
) -> None:
    root, records = _root(tmp_path)
    response = _response()
    response["choices"][0]["message"]["content"] = json.dumps(
        {"success": True, "score": 0.9, "rationale": "looks good"}
    )
    monkeypatch.setattr(
        vlm_evidence.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(response),
    )
    storage = _Storage()
    with pytest.raises(vlm_evidence.VlmEvidenceError, match="visible evidence"):
        vlm_evidence._call_once(
            root=root,
            attempt_id="opaque-case",
            freeze_sha256="a" * 64,
            purpose="calibration",
            frame_records=records,
            task="Review visible coherence.",
            rubric="Use visible pixels only.",
            api_key="secret-not-retained",
            external_attempt_prefix="s3://private/run/vlm/",
            storage_client=storage,
        )
    assert len(storage.objects) == 1


def test_freeze_acceptance_requires_exact_independent_review(
    monkeypatch, tmp_path: Path
) -> None:
    root, records = _root(tmp_path)
    manifest = {
        "case_order": ["a", "b", "c", "d"],
        "final_frames": records,
        "label_commitment_sha256": "1" * 64,
        "final_frame_manifest_sha256": "2" * 64,
    }
    monkeypatch.setattr(vlm_evidence, "_verified_freeze", lambda _args: manifest)
    prefix = "s3://private/run/vlm/"
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "format": vlm_evidence.FREEZE_REVIEW_FORMAT,
                "verdict": "ACCEPTED",
                "freeze_sha256": "a" * 64,
                "label_commitment_sha256": "1" * 64,
                "final_frame_manifest_sha256": "2" * 64,
                "external_attempt_prefix_sha256": vlm_evidence._sha_bytes(
                    prefix.encode()
                ),
                "controls_opened": 4,
                "final_frames_opened": 4,
                "prompts_reviewed": True,
                "reviewer_id": "independent-reviewer",
            }
        )
    )
    review.chmod(0o600)
    args = argparse.Namespace(
        evidence_root=root,
        freeze_sha256="a" * 64,
        review_path=review,
        external_attempt_prefix=prefix,
    )
    result = vlm_evidence.accept_freeze(args)
    assert (root / "external-attempt-prefix.txt").read_text().strip() == prefix
    assert (root / "external-attempt-prefix.txt").stat().st_mode & 0o077 == 0
    assert (
        vlm_evidence._verified_freeze_acceptance(
            root=root,
            manifest=manifest,
            freeze_sha256="a" * 64,
            acceptance_sha256=result["sha256"],
            external_attempt_prefix=prefix,
        )["status"]
        == "accepted"
    )


def test_http_error_retains_status_and_response_without_retry(
    monkeypatch, tmp_path: Path
) -> None:
    root, records = _root(tmp_path)
    storage = _Storage()
    calls = 0

    def fail(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(
            vlm_evidence.ENDPOINT,
            429,
            "limited",
            {},
            BytesIO(b'{"error":"limited"}'),
        )

    monkeypatch.setattr(vlm_evidence.urllib.request, "urlopen", fail)

    with pytest.raises(vlm_evidence.VlmEvidenceError, match="HTTP error"):
        vlm_evidence._call_once(
            root=root,
            attempt_id="opaque-case",
            freeze_sha256="a" * 64,
            purpose="calibration",
            frame_records=records,
            task="Review visible coherence.",
            rubric="Use visible pixels only.",
            api_key="secret-not-retained",
            external_attempt_prefix="s3://private/run/vlm/",
            storage_client=storage,
        )

    assert calls == 1
    attempt = root / "transport/opaque-case"
    assert (attempt / "response.json").read_bytes() == b'{"error":"limited"}'
    outcome = json.loads((attempt / "outcome.json").read_text())
    assert outcome["http_status"] == 429
    assert outcome["status"] == "response_failed"


def test_final_rejects_passing_calibration_from_another_freeze(
    monkeypatch, tmp_path: Path
) -> None:
    root, records = _root(tmp_path)
    labels = {
        f"case-{index}": {
            "role": "positive" if index < 2 else "negative",
            "expected_label": index < 2,
        }
        for index in range(4)
    }
    vlm_evidence._write_json(root / "labels.json", {"cases": labels})
    calibration_task = tmp_path / "calibration-task.txt"
    calibration_task.write_text("Review calibration pixels.")
    rubric = tmp_path / "rubric.txt"
    rubric.write_text("Use visible pixels only.")
    manifest = {
        "case_order": list(labels),
        "label_commitment_sha256": vlm_evidence._sha_file(root / "labels.json"),
        "final_frames": records,
        "final_task_sha256": "b" * 64,
        "calibration_task_sha256": vlm_evidence._sha_file(calibration_task),
        "rubric_sha256": vlm_evidence._sha_file(rubric),
    }
    calibration = {
        "format": vlm_evidence.CALIBRATION_FORMAT,
        "status": "pass",
        "freeze_sha256": "0" * 64,
        "model": vlm_evidence.MODEL,
        "served_model": vlm_evidence.MODEL,
        "threshold": vlm_evidence.THRESHOLD,
        "total": 4,
        "attempt_count": 4,
        "one_shot": True,
        "true_positives": 2,
        "true_negatives": 2,
        "false_positives": 0,
        "false_negatives": 0,
        "results": [],
    }
    vlm_evidence._write_json(root / "calibration.json", calibration)
    monkeypatch.setattr(vlm_evidence, "_verified_freeze", lambda _args: manifest)
    monkeypatch.setattr(
        vlm_evidence, "_verified_freeze_acceptance", lambda **_kwargs: {}
    )
    args = argparse.Namespace(
        evidence_root=root,
        freeze_sha256="a" * 64,
        freeze_acceptance_sha256="d" * 64,
        external_attempt_prefix="s3://private/run/vlm/",
        calibration_sha256=vlm_evidence._sha_file(root / "calibration.json"),
        calibration_task=calibration_task,
        rubric=rubric,
    )

    with pytest.raises(vlm_evidence.VlmEvidenceError, match="contract differs"):
        vlm_evidence.final(args)
