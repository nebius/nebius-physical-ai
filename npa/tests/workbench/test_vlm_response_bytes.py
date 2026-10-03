"""Verify durable response bytes precede decoding and strict failure parsing."""

import base64
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import httpx
from PIL import Image
import pytest

from npa.workbench import vlm_eval
from npa.workbench.vlm_eval import visual_review


def _client(response):
    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            return response

    return Client


def _request(root: Path):
    root.chmod(0o700)
    source = root / "frame.png"
    Image.new("RGB", (12, 9), "green").save(source)
    return vlm_eval.VlmVisualReviewRequest(
        input_path=str(source),
        output_path=str(root / "private-review"),
        model="MiniMaxAI/MiniMax-M3",
        task="Describe visible placement evidence.",
    )


@pytest.mark.parametrize(
    ("body", "content_type", "status"),
    [
        (b"\xff", "application/json; charset=utf-8", 502),
        (b"\xfe", "application/json; charset=utf-8", 502),
        (b"\xff", "application/json; charset=utf-8", 200),
        (b'{"error":"caf\xe9"}', "application/json; charset=iso-8859-1", 502),
        (b'{"error":"ordinary UTF-8 failure"}', "application/json", 502),
    ],
)
def test_visual_journal_retains_bytes_before_decoding(
    monkeypatch, tmp_path, body, content_type, status
) -> None:
    request = _request(tmp_path)
    evidence = Path(request.output_path) / "vlm_visual_review.evidence"
    wire_path = evidence / "response-bytes-01.json"

    class Response(httpx.Response):
        @property
        def text(self):
            assert wire_path.is_file(), "response decoded before durable byte journal"
            return super().text

    response = Response(
        status,
        content=body,
        headers={"content-type": content_type},
        request=httpx.Request("POST", "https://example.test/v1/chat/completions"),
    )
    monkeypatch.setattr(vlm_eval.httpx, "Client", _client(response))
    monkeypatch.setattr(
        visual_review, "_resolve_provider_key", lambda *_a, **_k: "synthetic"
    )
    report = vlm_eval.review_visual(request)
    assert report.status == "judge_error"
    assert report.outcomes[0].error is not None
    wire = json.loads(wire_path.read_text())
    assert base64.b64decode(wire["body_base64"], validate=True) == body
    assert wire["body_sha256"] == hashlib.sha256(body).hexdigest()
    assert wire["byte_count"] == len(body)
    assert wire["status_code"] == status
    assert wire_path.stat().st_mode & 0o777 == 0o600
    exact = report.outcomes[0].response_bytes
    assert exact is not None
    assert asdict(exact) == wire
    canonical = json.loads(Path(report.result_uri).read_text())
    assert canonical["outcomes"][0]["response_bytes"] == wire


def test_distinct_invalid_utf8_remains_distinct_in_canonical_outcomes(
    monkeypatch, tmp_path
) -> None:
    retained = []
    for index, body in enumerate((b"\xff", b"\xfe")):
        root = tmp_path / str(index)
        root.mkdir(mode=0o700)
        request = _request(root)
        response = httpx.Response(
            502,
            content=body,
            request=httpx.Request("POST", "https://example.test/v1/chat/completions"),
        )
        monkeypatch.setattr(vlm_eval.httpx, "Client", _client(response))
        monkeypatch.setattr(
            visual_review, "_resolve_provider_key", lambda *_a, **_k: "synthetic"
        )
        report = vlm_eval.review_visual(request)
        retained.append(report.outcomes[0])
    assert retained[0].provider.raw_response == retained[1].provider.raw_response
    assert (
        retained[0].response_bytes.body_sha256 != retained[1].response_bytes.body_sha256
    )
    assert (
        retained[0].response_bytes.body_base64 != retained[1].response_bytes.body_base64
    )


@pytest.mark.parametrize("damage", ["missing", "hash", "count", "body", "identity"])
def test_final_report_rejects_missing_or_divergent_wire_journal(
    monkeypatch, tmp_path, damage
) -> None:
    request = _request(tmp_path)
    response = httpx.Response(
        502,
        content=b"\xff",
        request=httpx.Request("POST", "https://example.test/v1/chat/completions"),
    )
    monkeypatch.setattr(vlm_eval.httpx, "Client", _client(response))
    monkeypatch.setattr(
        visual_review, "_resolve_provider_key", lambda *_a, **_k: "synthetic"
    )
    finalize = visual_review._finalize_visual_review_report

    def damage_then_finalize(report, context, journal, outcomes):
        path = Path(journal.root_uri) / "response-bytes-01.json"
        if damage == "missing":
            path.unlink()
        else:
            wire = json.loads(path.read_text())
            field, value = {
                "hash": ("body_sha256", "0" * 64),
                "count": ("byte_count", 2),
                "body": ("body_base64", "/g=="),
                "identity": ("request_id_header", "different-synthetic-id"),
            }[damage]
            wire[field] = value
            path.write_text(json.dumps(wire))
        finalize(report, context, journal, outcomes)

    monkeypatch.setattr(
        visual_review, "_finalize_visual_review_report", damage_then_finalize
    )
    with pytest.raises(vlm_eval.VlmVisualReviewError):
        vlm_eval.review_visual(request)
    assert not (Path(request.output_path) / "vlm_visual_review.json").exists()
    assert (
        Path(request.output_path)
        / "vlm_visual_review.evidence/transport-started-01.json"
    ).exists()


def test_textless_adapter_retains_actual_json_not_placeholder(monkeypatch) -> None:
    payload = {"model": "synthetic", "choices": []}

    class Response:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            return None

        def json(self):
            return payload

    observed = []
    monkeypatch.setattr(vlm_eval.httpx, "Client", _client(Response()))
    response = vlm_eval._post_backend_once(
        url="https://example.test/v1/chat/completions",
        headers={},
        request={},
        timeout_s=10,
        started_at=0,
        response_sink=observed.append,
        error_response_sink=None,
        request_body=b"{}",
    )
    assert response.data == payload
    assert response.raw_body == vlm_eval._canonical_json(payload)
    assert observed == [response]
    assert response.raw_body_base64 is None  # Never invent wire bytes from JSON.


def test_byte_retention_failure_stops_before_text_decoding(monkeypatch) -> None:
    response = httpx.Response(200, content=b"{}")
    monkeypatch.setattr(vlm_eval.httpx, "Client", _client(response))
    monkeypatch.setattr(
        vlm_eval,
        "_backend_response_from_http",
        lambda *_a, **_k: pytest.fail("decoded after failed byte retention"),
    )

    def fail(_wire):
        raise vlm_eval.VlmEvalError("synthetic retention failure")

    with pytest.raises(vlm_eval._VlmEvidenceRetentionError):
        vlm_eval._post_backend_once(
            url="https://example.test/v1/chat/completions",
            headers={},
            request={},
            timeout_s=10,
            started_at=0,
            response_sink=None,
            error_response_sink=None,
            request_body=b"{}",
            response_bytes_sink=fail,
        )
