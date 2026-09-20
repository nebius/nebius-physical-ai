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
    )

    assert result["served_model"] == vlm_evidence.MODEL
    assert result["score"] == 0.9
    request = (root / "transport/opaque-case/request.json").read_text()
    assert "expected_label" not in request
    assert "secret-not-retained" not in request
    assert (root / "transport/opaque-case/response.json").is_file()
    with pytest.raises(FileExistsError):
        vlm_evidence._call_once(
            root=root,
            attempt_id="opaque-case",
            freeze_sha256="a" * 64,
            purpose="calibration",
            frame_records=records,
            task="Review visible coherence.",
            rubric="Use visible pixels only.",
            api_key="secret-not-retained",
        )


def test_one_shot_call_rejects_served_model_drift(monkeypatch, tmp_path: Path) -> None:
    root, records = _root(tmp_path)
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


def test_http_error_retains_status_and_response_without_retry(
    monkeypatch, tmp_path: Path
) -> None:
    root, records = _root(tmp_path)
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
    manifest = {
        "case_order": list(labels),
        "label_commitment_sha256": vlm_evidence._sha_file(root / "labels.json"),
        "final_frames": records,
        "final_task_sha256": "b" * 64,
        "rubric_sha256": "c" * 64,
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
    args = argparse.Namespace(
        evidence_root=root,
        freeze_sha256="a" * 64,
        calibration_sha256=vlm_evidence._sha_file(root / "calibration.json"),
    )

    with pytest.raises(vlm_evidence.VlmEvidenceError, match="contract differs"):
        vlm_evidence.final(args)
