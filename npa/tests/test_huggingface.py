from __future__ import annotations

import httpx
import pytest

from npa.clients.huggingface import validate_hf_access, validate_hf_identity


def test_validate_hf_access_accepts_200(mocker) -> None:
    head = mocker.patch("httpx.head", return_value=httpx.Response(200))

    result = validate_hf_access("hf-token", "nvidia/model")

    assert result.ok is True
    assert result.status_code == 200
    assert head.call_args.kwargs["headers"] == {"Authorization": "Bearer hf-token"}


def test_validate_hf_access_rejects_401() -> None:
    result = validate_hf_access_with_status(401)

    assert result.ok is False
    assert result.status_code == 401
    assert (
        "Error: HF_TOKEN does not have access to nvidia/model. "
        "Request access at https://huggingface.co/nvidia/model and retry."
    ) == result.error


def test_validate_hf_access_rejects_403() -> None:
    result = validate_hf_access_with_status(403)

    assert result.ok is False
    assert result.status_code == 403


def test_live_hf_access_is_blocked_by_unit_guard() -> None:
    with pytest.raises(AssertionError, match="Live Hugging Face HTTP is blocked"):
        validate_hf_access("hf-token", "nvidia/model")


def test_validate_hf_access_reports_rate_limit_without_network() -> None:
    result = validate_hf_access_with_status(429)

    assert result.ok is False
    assert result.status_code == 429
    assert (
        result.error
        == "Unable to validate Hugging Face access to nvidia/model: HTTP 429"
    )


def validate_hf_access_with_status(status_code: int):
    mocker = pytest.MonkeyPatch()
    try:
        mocker.setattr(
            "httpx.head", lambda *args, **kwargs: httpx.Response(status_code)
        )
        return validate_hf_access("hf-token", "nvidia/model")
    finally:
        mocker.undo()


def test_validate_hf_identity_uses_authenticated_whoami_without_redirects(
    mocker,
) -> None:
    get = mocker.patch("httpx.get", return_value=httpx.Response(200))

    result = validate_hf_identity("hf-token")

    assert result.ok is True
    assert get.call_args.args == ("https://huggingface.co/api/whoami-v2",)
    assert get.call_args.kwargs["headers"] == {"Authorization": "Bearer hf-token"}
    assert get.call_args.kwargs["follow_redirects"] is False


def test_validate_hf_identity_rejects_expired_token(mocker) -> None:
    mocker.patch("httpx.get", return_value=httpx.Response(401))

    result = validate_hf_identity("hf-expired")

    assert result.ok is False
    assert result.status_code == 401
