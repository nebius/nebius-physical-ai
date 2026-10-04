from __future__ import annotations

import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import sys
import httpx

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))

from npa.clients.storage import StoragePreconditionFailed  # noqa: E402
from ncore_publication import vlm_evidence  # noqa: E402


class _Response(httpx.Response):
    def __init__(self, payload):
        super().__init__(
            200, json=payload, request=httpx.Request("POST", vlm_evidence.ENDPOINT)
        )


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


def test_minimax_request_changes_only_model_and_required_provider_profile(monkeypatch):
    frames = [b"first exact image", b"second exact image"]
    arguments = {
        "task": "Review scene geometry.",
        "rubric": "Reject defects.",
        "frames": frames,
    }
    monkeypatch.setattr(vlm_evidence, "MODEL", "openbmb/MiniCPM-V-4_5")
    original_bytes, original_prompt = vlm_evidence._build_request(**arguments)
    original = json.loads(original_bytes)
    assert original["response_format"] == {"type": "json_object"}
    assert "chat_template_kwargs" not in original
    monkeypatch.setattr(vlm_evidence, "MODEL", "MiniMaxAI/MiniMax-M3")
    current_bytes, current_prompt = vlm_evidence._build_request(**arguments)
    current = json.loads(current_bytes)
    assert current_prompt == original_prompt
    assert current["model"] == "MiniMaxAI/MiniMax-M3"
    assert current["chat_template_kwargs"] == {"thinking_mode": "disabled"}
    assert "response_format" not in current
    assert current["temperature"] == original["temperature"] == 0
    original.pop("model")
    original.pop("response_format")
    current.pop("model")
    current.pop("chat_template_kwargs")
    assert current == original


def test_request_respects_profile_without_temperature(monkeypatch):
    monkeypatch.setattr(vlm_evidence, "MODEL", "moonshotai/Kimi-K3")
    body, _ = vlm_evidence._build_request(
        task="Visible scene.", rubric="Visible geometry.", frames=[b"image"]
    )
    payload = json.loads(body)
    assert "temperature" not in payload
    assert payload["reasoning_effort"] == "low"
    assert payload["response_format"] == {"type": "json_object"}


def _ambiguous_response(corruption: str) -> bytes:
    payload = _response()
    message = payload["choices"][0]["message"]
    if corruption == "content-score":
        message["content"] = message["content"].replace(
            '"score": 0.9', '"score": 0.1, "score": 0.9'
        )
    elif corruption == "content-success":
        message["content"] = message["content"].replace(
            '"success": true', '"success": false, "success": true'
        )
    elif corruption == "refusal":
        message["refusal"] = "I cannot evaluate this request."
    raw = json.dumps(payload)
    duplicates = {
        "envelope-model": ('"model": ', '"model": "other/model", "model": '),
        "nested-finish": (
            '"finish_reason": "stop"',
            '"finish_reason": "length", "finish_reason": "stop"',
        ),
        "nested-usage": (
            '"prompt_tokens": 10',
            '"prompt_tokens": 1, "prompt_tokens": 10',
        ),
        "same-key": ('"score": 0.9', '"score": 0.9, "score": 0.9'),
    }
    if corruption == "same-key":
        before, after = duplicates[corruption]
        message["content"] = message["content"].replace(before, after)
        raw = json.dumps(payload)
    elif corruption in duplicates:
        raw = raw.replace(*duplicates[corruption])
    return raw.encode()


def _one_shot_arguments(root, records):
    return {
        "root": root,
        "attempt_id": "strict-control",
        "freeze_sha256": "a" * 64,
        "purpose": "calibration",
        "frame_records": records,
        "task": "Review visible coherence.",
        "rubric": "Use visible pixels only.",
    }


@pytest.mark.parametrize(
    "corruption",
    [
        "content-score",
        "content-success",
        "envelope-model",
        "nested-finish",
        "nested-usage",
        "same-key",
        "refusal",
    ],
)
def test_one_shot_rejects_ambiguous_json_and_refusal(monkeypatch, tmp_path, corruption):
    root, records = _root(tmp_path)
    storage = _Storage()
    raw = _ambiguous_response(corruption)
    calls = []

    def respond(*args):
        calls.append(args)
        return httpx.Response(200, content=raw)

    monkeypatch.setattr(vlm_evidence, "_post_hosted_bytes", respond)
    arguments = _one_shot_arguments(root, records)
    with pytest.raises(vlm_evidence.VlmEvidenceError, match="duplicate|refusal"):
        vlm_evidence._call_once(
            **arguments,
            api_key="not-retained",
            external_attempt_prefix="s3://private/run/vlm/",
            storage_client=storage,
        )
    transport = root / "transport/strict-control"
    assert (transport / "response.json").read_bytes() == raw
    outcome = json.loads((transport / "outcome.json").read_bytes())
    assert outcome["status"] == "response_failed"
    assert outcome["response_sha256"] == vlm_evidence._sha_bytes(raw)
    with pytest.raises(vlm_evidence.VlmEvidenceError, match="duplicate|refusal"):
        vlm_evidence._verified_transport(**arguments)
    with pytest.raises(vlm_evidence.VlmEvidenceError, match="retry is prohibited"):
        vlm_evidence._call_once(
            **arguments,
            api_key="not-retained",
            external_attempt_prefix="s3://private/run/vlm/",
            storage_client=storage,
        )
    assert len(calls) == 1 and len(storage.objects) == 1


@pytest.mark.parametrize("score", [0.0, 0.2, 0.8, 0.9, 1.0])
@pytest.mark.parametrize("refusal", [None, ""])
def test_strict_response_preserves_positive_and_negative_scores(score, refusal):
    payload = _response()
    message = payload["choices"][0]["message"]
    verdict = json.loads(message["content"])
    verdict.update(score=score, success=score >= vlm_evidence.THRESHOLD)
    message.update(content=json.dumps(verdict), refusal=refusal)
    result = vlm_evidence._response_semantics(json.dumps(payload).encode())
    assert result["score"] == score
    assert result["success"] is (score >= vlm_evidence.THRESHOLD)


@pytest.mark.parametrize("refusal", [False, 0, [], {}, ["refused"], "refused"])
def test_response_refusal_is_not_accepted_as_valid_content(refusal):
    payload = _response()
    payload["choices"][0]["message"]["refusal"] = refusal
    with pytest.raises(vlm_evidence.VlmEvidenceError, match="refusal"):
        vlm_evidence._response_semantics(json.dumps(payload).encode())


def test_one_shot_call_retains_exact_transport_without_labels(
    monkeypatch, tmp_path: Path
) -> None:
    root, records = _root(tmp_path)
    storage = _Storage()
    monkeypatch.setattr(
        vlm_evidence,
        "_post_hosted_bytes",
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
        vlm_evidence,
        "_post_hosted_bytes",
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


def test_block_control_breaks_global_geometry_not_just_translation(tmp_path):
    source = tmp_path / "coordinate-grid.png"
    image = Image.new("RGB", (256, 192))
    for index in range(48):
        column, row = index % 8, index // 8
        image.paste(
            (index, column, row),
            (column * 32, row * 32, (column + 1) * 32, (row + 1) * 32),
        )
    image.save(source)
    body = vlm_evidence._block_corrupt([source])[0]
    with Image.open(BytesIO(body)) as decoded:
        order = [
            decoded.getpixel((index % 8 * 32 + 16, index // 8 * 32 + 16))[0]
            for index in range(48)
        ]
    assert sorted(order) == list(range(48))
    assert all(destination != origin for destination, origin in enumerate(order))
    displacements = Counter(
        (origin % 8 - destination % 8, origin // 8 - destination // 8)
        for destination, origin in enumerate(order)
    )
    assert max(displacements.values()) < 24
    preserved_neighbors = sum(
        right == left + 1 for left, right in zip(order, order[1:])
    )
    assert preserved_neighbors < 12


@pytest.mark.parametrize("size", [(1, 1), (32, 32), (32, 128), (128, 32)])
def test_block_control_rejects_images_without_two_dimensional_tiles(tmp_path, size):
    source = tmp_path / "tiny.png"
    image = Image.new("RGB", size)
    image.putpixel((0, 0), (1, 2, 3))
    image.save(source)
    with pytest.raises(vlm_evidence.VlmEvidenceError, match="two-dimensional"):
        vlm_evidence._block_corrupt([source])


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
        vlm_evidence,
        "_post_hosted_bytes",
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
        vlm_evidence,
        "_post_hosted_bytes",
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
        request = httpx.Request("POST", vlm_evidence.ENDPOINT)
        response = httpx.Response(429, content=b'{"error":"limited"}', request=request)
        raise httpx.HTTPStatusError("limited", request=request, response=response)

    monkeypatch.setattr(vlm_evidence, "_post_hosted_bytes", fail)

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


def _calibrated_schedule(monkeypatch, tmp_path, *, final_score=0.9):
    """Exercise real evidence producers/checkers with synthetic media and transport."""
    root, records = _root(tmp_path)
    labels = {
        f"case-{index}": {"role": "synthetic-control", "expected_label": index < 2}
        for index in range(4)
    }
    frames = [(root / record["path"]).read_bytes() for record in records]
    cases = {
        case_id: {"frames": vlm_evidence._write_case(root, case_id, frames)}
        for case_id in labels
    }
    vlm_evidence._write_json(root / "labels.json", {"cases": labels})
    vlm_evidence._write_json(
        root / "final-frame-manifest.json",
        {
            "format": "npa_ncore_vlm_final_frames_v1",
            "render_inventory_sha256": "b" * 64,
            "frames": records,
        },
    )
    prompts = {}
    for name in ("rubric", "calibration_task", "final_task"):
        path = tmp_path / f"{name}.txt"
        path.write_text(f"Review visible scene geometry: {name}.")
        prompts[name] = path
    manifest = {
        "format": vlm_evidence.FREEZE_FORMAT,
        "schedule_id": "a" * 32,
        "model": vlm_evidence.MODEL,
        "threshold": vlm_evidence.THRESHOLD,
        "case_order": list(labels),
        "cases": cases,
        "label_commitment_sha256": vlm_evidence._sha_file(root / "labels.json"),
        "final_frame_manifest_sha256": vlm_evidence._sha_file(
            root / "final-frame-manifest.json"
        ),
        "render_inventory_sha256": "b" * 64,
        "final_frames": records,
        **{name + "_sha256": vlm_evidence._sha_file(p) for name, p in prompts.items()},
    }
    vlm_evidence._write_json(root / "freeze.json", manifest)
    prefix = "s3://test-bucket/synthetic-schedule/"
    review = tmp_path / "synthetic-review.json"
    vlm_evidence._write_json(
        review,
        {
            "format": vlm_evidence.FREEZE_REVIEW_FORMAT,
            "verdict": "ACCEPTED",
            "freeze_sha256": vlm_evidence._sha_file(root / "freeze.json"),
            "label_commitment_sha256": manifest["label_commitment_sha256"],
            "final_frame_manifest_sha256": manifest["final_frame_manifest_sha256"],
            "external_attempt_prefix_sha256": vlm_evidence._sha_bytes(prefix.encode()),
            "controls_opened": 4,
            "final_frames_opened": 4,
            "prompts_reviewed": True,
            "reviewer_id": "synthetic-test-reviewer",
        },
    )
    args = argparse.Namespace(
        evidence_root=root,
        freeze_sha256=vlm_evidence._sha_file(root / "freeze.json"),
        review_path=review,
        external_attempt_prefix=prefix,
        api_key_env="NCORE_SYNTHETIC_TEST_KEY",
        **prompts,
    )
    args.freeze_acceptance_sha256 = vlm_evidence.accept_freeze(args)["sha256"]
    storage = _Storage()
    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment", lambda: storage
    )
    monkeypatch.setenv(args.api_key_env, "synthetic-not-retained")
    calls = []
    scores = (0.9, 0.9, 0.2, 0.2, final_score)

    def respond(request_bytes, _api_key):
        assert len(calls) < len(scores), "unexpected extra inference"
        score = scores[len(calls)]
        calls.append(request_bytes)
        payload = _response()
        verdict = json.loads(payload["choices"][0]["message"]["content"])
        verdict.update(score=score, success=score >= vlm_evidence.THRESHOLD)
        payload["choices"][0]["message"]["content"] = json.dumps(verdict)
        return _Response(payload)

    monkeypatch.setattr(vlm_evidence, "_post_hosted_bytes", respond)
    result = vlm_evidence.calibrate(args)
    assert result["verdict"] == "pass"
    args.calibration_sha256 = result["sha256"]
    assert len(calls) == len(storage.objects) == 4
    return args, calls, storage


def _verify_schedule(args, final_sha256):
    return vlm_evidence.verify_complete_evidence(
        args.evidence_root,
        freeze_sha256=args.freeze_sha256,
        freeze_acceptance_sha256=args.freeze_acceptance_sha256,
        calibration_sha256=args.calibration_sha256,
        final_sha256=final_sha256,
        external_attempt_prefix=args.external_attempt_prefix,
        rubric_path=args.rubric,
        calibration_task_path=args.calibration_task,
        final_task_path=args.final_task,
    )


def test_completed_schedule_replays_five_attempts_without_new_calls(
    monkeypatch, tmp_path
):
    args, calls, storage = _calibrated_schedule(monkeypatch, tmp_path)
    final = vlm_evidence.final(args)

    verified = _verify_schedule(args, final["sha256"])

    assert verified["calibration"]["attempt_count"] == 4
    assert verified["final"]["attempt_count"] == 5
    assert verified["final"]["status"] == "pass"
    assert len(calls) == len(storage.objects) == 5
    with pytest.raises(vlm_evidence.VlmEvidenceError, match="schedule differs"):
        vlm_evidence.final(args)
    assert len(calls) == len(storage.objects) == 5


@pytest.mark.parametrize("score", [0.2, 0.799999])
def test_completed_schedule_rejects_semantically_relabeled_failed_final(
    monkeypatch, tmp_path, score
):
    args, calls, storage = _calibrated_schedule(
        monkeypatch, tmp_path, final_score=score
    )
    final = vlm_evidence.final(args)
    assert final["verdict"] == "failed"
    path = args.evidence_root / "final.json"
    result = json.loads(path.read_text())
    transport = args.evidence_root / "transport/final-one-shot/outcome.json"
    original_transport_sha256 = vlm_evidence._sha_file(transport)
    result["status"] = "pass"
    path.write_text(json.dumps(result))

    with pytest.raises(vlm_evidence.VlmEvidenceError, match="final VLM result differs"):
        _verify_schedule(args, vlm_evidence._sha_file(path))
    assert vlm_evidence._sha_file(transport) == original_transport_sha256
    assert len(calls) == len(storage.objects) == 5


def test_completed_schedule_accepts_exact_threshold_without_new_calls(
    monkeypatch, tmp_path
):
    args, calls, storage = _calibrated_schedule(
        monkeypatch, tmp_path, final_score=vlm_evidence.THRESHOLD
    )
    final = vlm_evidence.final(args)

    verified = _verify_schedule(args, final["sha256"])

    assert verified["final"]["score"] == vlm_evidence.THRESHOLD
    assert verified["final"]["status"] == "pass"
    assert len(calls) == len(storage.objects) == 5


def test_calibration_only_schedule_is_not_complete(monkeypatch, tmp_path):
    args, calls, storage = _calibrated_schedule(monkeypatch, tmp_path)
    with pytest.raises(vlm_evidence.VlmEvidenceError, match="schedule differs"):
        _verify_schedule(args, "f" * 64)
    assert len(calls) == len(storage.objects) == 4


@pytest.mark.parametrize("directory", ["attempt-ledger", "transport"])
@pytest.mark.parametrize("change", ["extra", "missing-calibration", "missing-final"])
def test_completed_schedule_rejects_changed_attempt_population(
    monkeypatch, tmp_path, directory, change
):
    args, calls, storage = _calibrated_schedule(monkeypatch, tmp_path)
    final = vlm_evidence.final(args)
    attempt = "case-0" if change == "missing-calibration" else "final-one-shot"
    target = args.evidence_root / directory
    if change == "extra":
        if directory == "attempt-ledger":
            vlm_evidence._write_json(target / "unexpected.json", {})
        else:
            (target / "unexpected").mkdir()
    else:
        source = target / (
            attempt + ".json" if directory == "attempt-ledger" else attempt
        )
        source.rename(tmp_path / "removed-attempt")

    with pytest.raises(vlm_evidence.VlmEvidenceError, match="schedule differs"):
        _verify_schedule(args, final["sha256"])
    assert len(calls) == len(storage.objects) == 5


@pytest.mark.parametrize(
    "relative",
    [
        "labels.json",
        "freeze-acceptance.json",
        "calibration.json",
        "final.json",
        "final/frame-000.png",
        "attempt-ledger/case-0.json",
        "attempt-ledger/final-one-shot.json",
        "transport/case-0/response.json",
        "transport/final-one-shot/request.json",
        "transport/final-one-shot/response.json",
        "transport/final-one-shot/outcome.json",
        "transport-manifest.json",
    ],
)
def test_completed_schedule_rejects_substituted_evidence(
    monkeypatch, tmp_path, relative
):
    args, calls, storage = _calibrated_schedule(monkeypatch, tmp_path)
    final = vlm_evidence.final(args)
    path = args.evidence_root / relative
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(vlm_evidence.VlmEvidenceError, match="differ"):
        _verify_schedule(args, final["sha256"])
    assert len(calls) == len(storage.objects) == 5


def test_completed_schedule_does_not_accept_failed_final(monkeypatch, tmp_path):
    args, calls, storage = _calibrated_schedule(monkeypatch, tmp_path, final_score=0.2)
    final = vlm_evidence.final(args)
    assert final["verdict"] == "failed"

    with pytest.raises(vlm_evidence.VlmEvidenceError, match="final VLM result differs"):
        _verify_schedule(args, final["sha256"])
    assert len(calls) == len(storage.objects) == 5


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_provider_redirect_is_retained_without_following_or_retry(
    monkeypatch, tmp_path, status
):
    root, records = _root(tmp_path)
    storage = _Storage()
    requests = []
    original_client = httpx.Client

    def transport(request):
        requests.append(request)
        return httpx.Response(
            status,
            headers={"Location": "https://other.invalid/collect"},
            content=b"redirect rejected",
        )

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(transport), **kwargs
        ),
    )
    with pytest.raises(vlm_evidence.VlmEvidenceError, match="HTTP error"):
        vlm_evidence._call_once(
            root=root,
            attempt_id="redirect-case",
            freeze_sha256="a" * 64,
            purpose="calibration",
            frame_records=records,
            task="Review visible coherence.",
            rubric="Use visible pixels only.",
            api_key="synthetic-test-credential",
            external_attempt_prefix="s3://private/run/vlm/",
            storage_client=storage,
        )
    assert len(requests) == 1
    assert str(requests[0].url) == vlm_evidence.ENDPOINT
    attempt = root / "transport/redirect-case"
    assert requests[0].content == (attempt / "request.json").read_bytes()
    assert (attempt / "response.json").read_bytes() == b"redirect rejected"
    assert json.loads((attempt / "outcome.json").read_text())["http_status"] == status
    assert len(storage.objects) == 1
    assert all(
        "synthetic-test-credential" not in p.read_text() for p in attempt.glob("*.json")
    )
