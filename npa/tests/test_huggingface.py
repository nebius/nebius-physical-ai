from __future__ import annotations

import httpx
import pytest

from npa.clients.huggingface import (
    validate_hf_access,
    validate_hf_file_access,
    validate_hf_identity,
)


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


class _Response:
    def __init__(self, status_code: int, *, location: str = "") -> None:
        self.status_code = status_code
        self.headers = {"location": location} if location else {}


def test_catalogued_gated_repo_uses_exact_artifact_not_public_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def head(url: str, **kwargs):
        calls.append(url)
        return _Response(403)

    monkeypatch.setattr(httpx, "head", head)
    monkeypatch.setattr(httpx, "get", lambda url, **kwargs: _Response(200))

    result = validate_hf_access("hf-synthetic", "nvidia/Cosmos-Reason2-2B")

    assert result.ok is False
    assert result.error_kind == "entitlement"
    assert calls == [
        "https://huggingface.co/nvidia/Cosmos-Reason2-2B/resolve/main/model.safetensors"
    ]


def test_exact_dataset_probe_uses_dataset_resolve_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, dict]] = []

    def head(url: str, **kwargs):
        observed.append((url, kwargs))
        return _Response(302, location="/api/resolve-cache/datasets/object")

    monkeypatch.setattr(httpx, "head", head)
    result = validate_hf_file_access(
        "hf-synthetic",
        "owner/data",
        "revision",
        "index.parquet",
        repo_type="dataset",
    )

    assert result.ok is True
    assert observed[0][0] == (
        "https://huggingface.co/datasets/owner/data/resolve/revision/index.parquet"
    )
    assert observed[0][1]["follow_redirects"] is False


@pytest.mark.parametrize(
    ("identity_status", "expected_kind"),
    [(200, "entitlement"), (401, "authentication"), (403, "authentication")],
)
def test_exact_probe_distinguishes_entitlement_from_invalid_token(
    monkeypatch: pytest.MonkeyPatch,
    identity_status: int,
    expected_kind: str,
) -> None:
    monkeypatch.setattr(httpx, "head", lambda *args, **kwargs: _Response(403))
    identity_calls: list[tuple[str, dict]] = []

    def get(url: str, **kwargs):
        identity_calls.append((url, kwargs))
        return _Response(identity_status)

    monkeypatch.setattr(httpx, "get", get)
    result = validate_hf_file_access(
        "hf-synthetic", "owner/gated", "revision", "weights.bin"
    )

    assert result.error_kind == expected_kind
    assert identity_calls[0][0] == "https://huggingface.co/api/whoami-v2"
    assert identity_calls[0][1]["follow_redirects"] is False


def test_exact_probe_keeps_catalog_drift_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(httpx, "head", lambda *args, **kwargs: _Response(404))
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *args, **kwargs: pytest.fail("404 must not trigger identity lookup"),
    )

    result = validate_hf_file_access(
        "hf-synthetic", "owner/gated", "stale-revision", "missing.bin"
    )

    assert result.ok is False
    assert result.error_kind == "catalog_drift"


def test_exact_probe_keeps_identity_network_uncertainty_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(httpx, "head", lambda *args, **kwargs: _Response(401))

    def timeout(*args, **kwargs):
        raise httpx.ReadTimeout("synthetic timeout")

    monkeypatch.setattr(httpx, "get", timeout)
    result = validate_hf_file_access(
        "hf-synthetic", "owner/gated", "revision", "weights.bin"
    )

    assert result.error_kind == "transient"
    assert "synthetic timeout" not in result.error


def test_exact_probe_never_forwards_token_to_redirect_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def respond(url: str, **kwargs):
        calls.append(kwargs)
        return _Response(
            302,
            location="https://objects.synthetic.invalid/file?Signature=synthetic",
        )

    monkeypatch.setattr(httpx, "head", respond)
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *args, **kwargs: pytest.fail("redirect target must never be requested"),
    )
    result = validate_hf_file_access(
        "hf-synthetic", "owner/gated", "revision", "weights.bin"
    )

    assert result.ok is False
    assert result.error_kind == "unverified_redirect"
    assert calls == [
        {
            "headers": {"Authorization": "Bearer hf-synthetic"},
            "timeout": 10.0,
            "follow_redirects": False,
        }
    ]


def test_exact_probe_uses_one_byte_get_when_head_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(httpx, "head", lambda *args, **kwargs: _Response(405))
    calls: list[tuple[str, dict]] = []

    def get(url: str, **kwargs):
        calls.append((url, kwargs))
        return _Response(206)

    monkeypatch.setattr(httpx, "get", get)
    result = validate_hf_file_access(
        "hf-synthetic", "owner/gated", "revision", "weights.bin"
    )

    assert result.ok is True
    assert calls[0][1]["headers"]["Range"] == "bytes=0-0"
    assert calls[0][1]["follow_redirects"] is False
