from __future__ import annotations

import httpx

from npa.clients.huggingface import validate_hf_access


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


def validate_hf_access_with_status(status_code: int):
    import pytest

    mocker = pytest.MonkeyPatch()
    try:
        mocker.setattr("httpx.head", lambda *args, **kwargs: httpx.Response(status_code))
        return validate_hf_access("hf-token", "nvidia/model")
    finally:
        mocker.undo()

