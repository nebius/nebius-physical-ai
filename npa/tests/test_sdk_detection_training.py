"""Unit tests for `npa.sdk.workbench.detection_training` compatibility SDK."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from npa.sdk.workbench.detection_training import (
    DetectionTrainingServiceError,
    DetectionTrainingValidationError,
    _request_json,
    _resolve_mode,
    eval as eval_run,
    status as status_run,
    train,
)
from npa.workbench.detection_training.schemas import (
    EvalResponse,
    StatusResponse,
    TrainResponse,
)


# ── _resolve_mode ─────────────────────────────────────────────────────────


def test_resolve_mode_defaults_to_service_flag() -> None:
    assert _resolve_mode(mode=None, service=True) is True
    assert _resolve_mode(mode=None, service=False) is False


def test_resolve_mode_explicit_strings_override_service_flag() -> None:
    assert _resolve_mode(mode="service", service=False) is True
    assert _resolve_mode(mode="local", service=True) is False
    assert _resolve_mode(mode="  SERVICE  ", service=False) is True


def test_resolve_mode_rejects_unknown_string() -> None:
    with pytest.raises(DetectionTrainingValidationError, match="mode must be"):
        _resolve_mode(mode="remote", service=False)


# ── _request_json ─────────────────────────────────────────────────────────


def _install_fake_request(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status_code: int = 200,
    body: Any = None,
    raise_exc: Exception | None = None,
):
    captured: dict[str, Any] = {}

    def fake_request(
        method,
        url,
        headers=None,
        json=None,
        params=None,
        timeout=None,
    ):
        if raise_exc is not None:
            raise raise_exc
        captured.update(
            method=method,
            url=url,
            headers=headers or {},
            json=json,
            params=params,
            timeout=timeout,
        )
        return httpx.Response(
            status_code,
            json=body if body is not None else {},
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(httpx, "request", fake_request)
    return captured


def test_request_json_strips_trailing_slash_and_attaches_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DET_TOKEN", "secret")
    captured = _install_fake_request(monkeypatch, body={"ok": True})

    result = _request_json(
        "POST",
        "https://svc.example/",
        "/train",
        token_env="DET_TOKEN",
        timeout=5.0,
        payload={"a": 1},
    )

    assert result == {"ok": True}
    assert captured["url"] == "https://svc.example/train"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["json"] == {"a": 1}


def test_request_json_requires_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(DetectionTrainingValidationError, match="endpoint is required"):
        _request_json("GET", "", "/status", token_env="X", timeout=1.0)


def test_request_json_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_request(monkeypatch, status_code=500, body={"detail": "boom"})

    with pytest.raises(DetectionTrainingServiceError, match=r"\(500\)"):
        _request_json(
            "GET", "https://svc", "/status", token_env="X", timeout=1.0
        )


def test_request_json_raises_on_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_request(
        monkeypatch, raise_exc=httpx.ConnectError("refused")
    )

    with pytest.raises(DetectionTrainingServiceError, match="Cannot reach"):
        _request_json(
            "GET", "https://svc", "/status", token_env="X", timeout=1.0
        )


def test_request_json_raises_on_non_json_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(method, url, headers=None, json=None, params=None, timeout=None):
        return httpx.Response(
            200,
            content=b"<html>not json</html>",
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(httpx, "request", fake_request)

    with pytest.raises(DetectionTrainingServiceError, match="non-JSON"):
        _request_json("GET", "https://svc", "/p", token_env="X", timeout=1.0)


def test_request_json_raises_when_body_is_not_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_request(monkeypatch, body=["unexpected", "list"])

    with pytest.raises(DetectionTrainingServiceError, match="unexpected response"):
        _request_json("GET", "https://svc", "/p", token_env="X", timeout=1.0)


# ── Top-level train / eval / status ───────────────────────────────────────


_TRAIN_BODY = {
    "run_id": "r1",
    "status": "completed",
    "checkpoint_uri_pattern": "s3://b/ck-{epoch}.pt",
    "metrics_uri": "s3://b/metrics.json",
    "total_epochs": 3,
    "manifest_sha256": "deadbeef",
}

_EVAL_BODY = {
    "mAP": 0.5,
    "mAP_50": 0.6,
    "mAP_75": 0.45,
    "per_category_AP": {"car": 0.7},
    "eval_run_id": "e1",
    "manifest_sha256": "cafef00d",
}

_STATUS_BODY = {
    "run_id": "r1",
    "status": "running",
    "epochs_completed": 1,
    "total_epochs": 3,
}


def test_train_local_mode_delegates_to_training_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_train_detector(request):
        captured["request"] = request
        return TrainResponse(**_TRAIN_BODY)

    import npa.workbench.detection_training.training as training_mod

    monkeypatch.setattr(training_mod, "train_detector", fake_train_detector)

    result = train(
        view="train",
        output_uri="s3://bucket/out/",
        lance_uri="s3://bucket/lance/",
    )

    assert isinstance(result, TrainResponse)
    assert result.run_id == "r1"
    assert captured["request"].view == "train"
    assert captured["request"].output_uri == "s3://bucket/out/"


def test_train_service_mode_uses_endpoint_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fake_request(monkeypatch, body=_TRAIN_BODY)
    monkeypatch.setenv("NPA_DETECTION_TRAINING_ENDPOINT", "https://svc.example")

    result = train(
        view="train",
        output_uri="s3://bucket/out/",
        service=True,
    )

    assert isinstance(result, TrainResponse)
    assert captured["url"] == "https://svc.example/train"
    assert captured["json"]["view"] == "train"


def test_eval_service_mode_posts_to_eval_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fake_request(monkeypatch, body=_EVAL_BODY)

    result = eval_run(
        checkpoint_uri="s3://bucket/ck.pt",
        eval_view="val",
        output_uri="s3://bucket/eval/",
        endpoint="https://svc.example/",
        mode="service",
    )

    assert isinstance(result, EvalResponse)
    assert result.mAP == 0.5
    assert captured["url"] == "https://svc.example/eval"
    assert captured["json"]["checkpoint_uri"] == "s3://bucket/ck.pt"


def test_eval_local_mode_delegates_to_evaluation_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import npa.workbench.detection_training.evaluation as eval_mod

    monkeypatch.setattr(
        eval_mod, "evaluate_detector", lambda req: EvalResponse(**_EVAL_BODY)
    )

    result = eval_run(
        checkpoint_uri="s3://bucket/ck.pt",
        eval_view="val",
        output_uri="s3://bucket/eval/",
    )

    assert result.eval_run_id == "e1"


def test_status_service_mode_passes_run_id_as_query_param(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fake_request(monkeypatch, body=_STATUS_BODY)

    result = status_run(
        run_id="r1",
        endpoint="https://svc.example",
        mode="service",
    )

    assert isinstance(result, StatusResponse)
    assert result.run_id == "r1"
    assert captured["params"] == {"run_id": "r1"}
    assert captured["method"] == "GET"


def test_status_local_mode_delegates_to_service_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import npa.workbench.detection_training.service as service_mod

    monkeypatch.setattr(
        service_mod, "status_for_run", lambda rid: StatusResponse(**_STATUS_BODY)
    )

    result = status_run(run_id="r1")

    # RunStatus is a str-Enum; compare against the string value directly.
    assert str(result.status) == "running" or result.status == "running"
