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


def test_metadata_success_does_not_override_payload_denial(mocker) -> None:
    def head(url: str, **kwargs):
        del kwargs
        return httpx.Response(200 if "/api/models/" in url else 403)

    request = mocker.patch("httpx.head", side_effect=head)

    metadata = validate_hf_access("hf-synthetic", "vendor/gated", revision="rev")
    payload = validate_hf_access(
        "hf-synthetic",
        "vendor/gated",
        "model",
        "rev",
        "weights/model.safetensors",
    )

    assert metadata.ok is True
    assert payload.ok is False
    assert payload.status_code == 403
    assert request.call_count == 2


@pytest.mark.parametrize(
    "location",
    [
        "/api/resolve-cache/models/vendor/gated/rev/weights/model.safetensors",
        "https://cdn-lfs-us-1.hf.co/object?X-Amz-Signature=synthetic",
        "https://cas-bridge.xethub.hf.co/object?X-Xet-Signed=synthetic",
        "https://region.cdn.hf.co/object?X-Xet-Signed=synthetic",
    ],
)
def test_exact_payload_trusted_redirect_is_ready_without_forwarding_token(
    mocker, location: str
) -> None:
    head = mocker.patch(
        "httpx.head",
        side_effect=[
            httpx.Response(302, headers={"location": location}),
            httpx.Response(200),
        ],
    )

    result = validate_hf_file_access(
        "hf-synthetic", "vendor/gated", "rev", "weights/model.safetensors"
    )

    assert result.ok is True
    assert head.call_count == 2
    assert head.call_args_list[0].kwargs["follow_redirects"] is False
    assert head.call_args_list[0].kwargs["headers"] == {
        "Authorization": "Bearer hf-synthetic"
    }
    assert head.call_args_list[1].kwargs["headers"] == {}


@pytest.mark.parametrize(
    "location",
    [
        "",
        "/login?next=/vendor/gated",
        "https://huggingface.co/join",
        "https://untrusted.invalid/object?signature=synthetic",
    ],
)
def test_exact_payload_login_missing_and_untrusted_redirects_are_not_ready(
    mocker, location: str
) -> None:
    mocker.patch(
        "httpx.head", return_value=httpx.Response(302, headers={"location": location})
    )

    result = validate_hf_file_access(
        "hf-synthetic", "vendor/gated", "rev", "weights/model.safetensors"
    )

    assert result.ok is False
    assert result.status_code == 302


def test_exact_dataset_payload_uses_dataset_revision_and_path(mocker) -> None:
    head = mocker.patch("httpx.head", return_value=httpx.Response(206))

    result = validate_hf_access(
        "hf-synthetic",
        "vendor/dataset",
        "dataset",
        "dataset-revision",
        "data/chunk-0000.parquet",
    )

    assert result.ok is True
    assert head.call_args.args == (
        "https://huggingface.co/datasets/vendor/dataset/resolve/"
        "dataset-revision/data/chunk-0000.parquet",
    )
    assert head.call_args.kwargs["follow_redirects"] is False


def test_head_not_allowed_falls_back_to_one_byte_range_without_redirects(mocker) -> None:
    mocker.patch("httpx.head", return_value=httpx.Response(405))
    get = mocker.patch("httpx.get", return_value=httpx.Response(206))

    result = validate_hf_file_access(
        "hf-synthetic", "vendor/gated", "rev", "weights/model.safetensors"
    )

    assert result.ok is True
    assert get.call_args.kwargs["headers"] == {
        "Authorization": "Bearer hf-synthetic",
        "Range": "bytes=0-0",
    }
    assert get.call_args.kwargs["follow_redirects"] is False


@pytest.mark.parametrize(
    ("revision", "filename"), [("", "weights/model.safetensors"), ("rev", "")]
)
def test_exact_payload_probe_requires_revision_and_payload_without_http(
    mocker, revision: str, filename: str
) -> None:
    head = mocker.patch("httpx.head")

    result = validate_hf_file_access(
        "hf-synthetic", "vendor/gated", revision, filename
    )

    assert result.ok is False
    assert "required" in result.error
    head.assert_not_called()
