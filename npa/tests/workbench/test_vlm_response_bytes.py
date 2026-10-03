"""Verify durable response bytes precede decoding and strict failure parsing."""

import base64
from dataclasses import asdict, replace
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


def _preference_request(root: Path):
    first, second = root / "first.png", root / "second.png"
    Image.new("RGB", (12, 9), "red").save(first)
    Image.new("RGB", (12, 9), "blue").save(second)
    return vlm_eval.VlmPreferenceComparisonRequest(
        baseline_path=str(first),
        candidate_path=str(second),
        output_path=str(root / "private-preference"),
        model="MiniMaxAI/MiniMax-M3",
        task="Compare visible placement evidence.",
        rubric="Use only visible pixels.",
        endpoint_url="https://example.test/v1",
        api_key_env="SYNTHETIC_PREFERENCE_KEY",
    )


@pytest.mark.parametrize(
    ("bodies", "content_type", "status"),
    [
        ([b"\xff", b"\xfe"], "application/json; charset=utf-8", 502),
        ([b"\xff", b"\xfe"], "application/json; charset=utf-8", 200),
        ([b'{"error":"caf\xe9"}'] * 2, "application/json; charset=iso-8859-1", 502),
        ([b'{"error":"ordinary UTF-8 failure"}'] * 2, "application/json", 502),
        (
            ['{"error":"caf\u00e9"}'.encode()] * 2,
            "application/json; charset=utf-8",
            502,
        ),
    ],
)
def test_preference_retains_wire_before_decode_in_journal_and_outcome(
    monkeypatch, tmp_path, bodies, content_type, status
) -> None:
    request = _preference_request(tmp_path)
    journal = Path(request.output_path) / ".vlm_preference_comparison"
    count = 0

    class Response(httpx.Response):
        @property
        def text(self):
            assert (journal / f"response-bytes-{count:02d}.json").is_file()
            return super().text

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            nonlocal count
            body = bodies[count]
            count += 1
            return Response(
                status,
                content=body,
                headers={"content-type": content_type},
                request=httpx.Request(
                    "POST", "https://example.test/v1/chat/completions"
                ),
            )

    monkeypatch.setattr(vlm_eval.httpx, "Client", Client)
    monkeypatch.setattr(vlm_eval, "_preference_api_key", lambda **_kwargs: "synthetic")
    report = vlm_eval.compare_vlm_preference(request)
    assert count == 2
    assert report.deployment_status == "audit_only"
    assert report.status == "judge_error" and report.escalation_required
    payload = json.loads(json.dumps(asdict(report)))
    for index, name in enumerate(("first_order", "reversed_order"), 1):
        wire = payload[name]["response_bytes"]
        assert base64.b64decode(wire["body_base64"], validate=True) == bodies[index - 1]
        assert wire["body_sha256"] == hashlib.sha256(bodies[index - 1]).hexdigest()
        assert wire["byte_count"] == len(bodies[index - 1])
        assert wire == json.loads(
            (journal / f"response-bytes-{index:02d}.json").read_text()
        )
        assert (
            wire
            == json.loads(
                (journal / f"transport-boundary-{index:02d}.json").read_text()
            )["response_bytes"]
        )
        assert (
            wire
            == json.loads((journal / f"response-{index:02d}.json").read_text())[
                "response_bytes"
            ]
        )
    if bodies == [b"\xff", b"\xfe"]:
        assert (
            report.first_order.provider.raw_response
            == report.reversed_order.provider.raw_response
        )
        assert (
            report.first_order.response_bytes["body_sha256"]
            != report.reversed_order.response_bytes["body_sha256"]
        )


@pytest.mark.parametrize("damage", ["hash", "count", "encoding", "missing"])
def test_preference_byte_evidence_rejects_incomplete_or_tampered_capture(damage):
    response = vlm_eval._VlmBackendResponse(
        {},
        "\ufffd",
        502,
        None,
        0.1,
        raw_body_base64="/w==",
        raw_body_bytes_sha256=hashlib.sha256(b"\xff").hexdigest(),
        raw_body_byte_count=1,
    )
    fields = {
        "hash": {"raw_body_bytes_sha256": "0" * 64},
        "count": {"raw_body_byte_count": 2},
        "encoding": {"raw_body_base64": "!invalid!"},
        "missing": {"raw_body_bytes_sha256": None},
    }
    with pytest.raises(vlm_eval.VlmEvalError, match="provider response byte"):
        vlm_eval._backend_response_byte_evidence(replace(response, **fields[damage]))


def test_preference_text_only_adapters_do_not_invent_wire_bytes():
    response = vlm_eval._VlmBackendResponse({}, "legacy decoded text", 502, None, 0.1)
    assert vlm_eval._backend_response_byte_evidence(response) is None


def test_preference_success_retains_exact_wire_without_changing_request(
    monkeypatch, tmp_path
):
    request = _preference_request(tmp_path)
    verdict = {
        "preference": "tie",
        "confidence": "high",
        "observable_support": ["Both images have visible detail."],
        "critical_defects": {"A": ["No visible defect."], "B": ["No visible defect."]},
        "uncertainty": "Hidden state is not observable.",
    }
    body = json.dumps(
        {
            "model": request.model,
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(verdict)}}
            ],
        }
    ).encode()
    response = httpx.Response(
        200,
        content=body,
        request=httpx.Request("POST", "https://example.test/v1/chat/completions"),
    )
    monkeypatch.setattr(vlm_eval.httpx, "Client", _client(response))
    monkeypatch.setattr(vlm_eval, "_preference_api_key", lambda **_kwargs: "synthetic")
    report = vlm_eval.compare_vlm_preference(request)
    assert report.agreement_eligible
    assert report.deployment_status == "audit_only"
    payload = json.loads(json.dumps(asdict(report)))
    journal = Path(request.output_path) / ".vlm_preference_comparison"
    for index, name in enumerate(("first_order", "reversed_order"), 1):
        outcome = payload[name]
        assert outcome["transport_request_sha256"] == vlm_eval._sha256_json(
            outcome["transport_request"]
        )
        assert outcome["verdict"] == verdict
        assert (
            base64.b64decode(outcome["response_bytes"]["body_base64"], validate=True)
            == body
        )
        assert outcome["response_bytes"] == json.loads(
            (journal / f"response-bytes-{index:02d}.json").read_text()
        )
    assert json.loads((journal / "report-ready.json").read_text()) == payload


def test_preference_byte_journal_failure_stops_before_decoding_or_second_call(
    monkeypatch, tmp_path
):
    request = _preference_request(tmp_path)
    calls = []
    original_write = vlm_eval._write_preference_journal

    def write(journal, name, payload):
        if name.startswith("response-bytes-"):
            raise vlm_eval.VlmEvalError("synthetic byte journal refusal")
        return original_write(journal, name, payload)

    class Response(httpx.Response):
        @property
        def text(self):
            pytest.fail("Byte journal failure must stop before decoding")

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            calls.append("synthetic HTTP")
            return Response(
                502,
                content=b"\xff",
                request=httpx.Request(
                    "POST", "https://example.test/v1/chat/completions"
                ),
            )

    monkeypatch.setattr(vlm_eval.httpx, "Client", Client)
    monkeypatch.setattr(vlm_eval, "_preference_api_key", lambda **_kwargs: "synthetic")
    monkeypatch.setattr(vlm_eval, "_write_preference_journal", write)
    with pytest.raises(
        vlm_eval.VlmEvalError, match="provider response bytes could not be retained"
    ):
        vlm_eval.compare_vlm_preference(request)
    assert calls == ["synthetic HTTP"]


@pytest.mark.parametrize(
    "helper", ["_backend_response_from_http", "_captured_http_response"]
)
@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        (b"\xff", "application/json; charset=utf-8"),
        (b"\xfe", "application/json; charset=utf-8"),
        (b'{"error":"caf\xe9"}', "application/json; charset=iso-8859-1"),
        (b'{"error":"ordinary UTF-8 failure"}', "application/json"),
        ('{"value":"caf\u00e9"}'.encode(), "application/json; charset=utf-8"),
    ],
)
def test_shared_http_capture_helpers_preserve_wire_before_decoding(
    helper, body, content_type
) -> None:
    events = []

    class Response(httpx.Response):
        @property
        def content(self):
            events.append("wire")
            return super().content

        @property
        def text(self):
            assert events and events[0] == "wire"
            events.append("text")
            return super().text

        def json(self, **kwargs):
            assert events and events[0] == "wire"
            events.append("json")
            return super().json(**kwargs)

    response = Response(
        502,
        content=body,
        headers={"content-type": content_type, "x-request-id": "synthetic-id"},
        request=httpx.Request("POST", "https://example.test/v1/chat/completions"),
    )
    events.clear()
    arguments = {"started_at": 0}
    if helper == "_backend_response_from_http":
        arguments["data"] = {}
    captured = getattr(vlm_eval, helper)(response, **arguments)
    assert events[0] == "wire"
    assert base64.b64decode(captured.raw_body_base64, validate=True) == body
    assert captured.raw_body_bytes_sha256 == hashlib.sha256(body).hexdigest()
    assert captured.raw_body_byte_count == len(body)
    assert captured.status_code == 502
    assert captured.request_id_header == "synthetic-id"
    assert captured.raw_body == response.text


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
