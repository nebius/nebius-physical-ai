"""Verify durable response bytes precede decoding and strict failure parsing."""

import base64
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
