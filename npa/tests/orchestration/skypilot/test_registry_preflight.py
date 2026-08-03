from __future__ import annotations

import json

import pytest

from npa.orchestration.skypilot.registry_preflight import (
    RegistryPreflightError,
    check_image_pull,
    check_image_pulls,
    parse_image_reference,
    resolve_registry_credentials,
)


REGISTRY = "cr.us-central1.nebius.cloud"
REPOSITORY = "u00j7q4jjkahvsx0jy/npa-cosmos2-transfer"
TAG = "2.5.1-golden-eval-smoke-20260616T033000Z"
IMAGE = f"{REGISTRY}/{REPOSITORY}:{TAG}"
MANIFEST_URL = f"https://{REGISTRY}/v2/{REPOSITORY}/manifests/{TAG}"
# Verbatim challenge from the live Nebius Container Registry.
CHALLENGE = {
    "www-authenticate": (
        f'Bearer realm="https://{REGISTRY}/v2/token/",service="{REGISTRY}"'
    )
}


class FakeRegistry:
    """A Docker Registry v2 endpoint with a scriptable manifest response."""

    def __init__(self, *, manifest_status: int, manifest_body: bytes = b"", token_status: int = 200):
        self.manifest_status = manifest_status
        self.manifest_body = manifest_body
        self.token_status = token_status
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, headers: dict[str, str], timeout: int):
        self.calls.append((url, dict(headers)))
        if "/v2/token/" in url:
            if self.token_status >= 400:
                body = json.dumps(
                    {"errors": [{"code": "UNAUTHORIZED", "message": "authentication required"}]}
                ).encode()
                return self.token_status, {}, body
            return self.token_status, {}, json.dumps({"token": "scoped-bearer"}).encode()
        if "Authorization" not in headers:
            return 401, dict(CHALLENGE), b""
        return self.manifest_status, {}, self.manifest_body


def _error_body(code: str, message: str) -> bytes:
    return json.dumps({"errors": [{"code": code, "message": message, "detail": None}]}).encode()


def test_parse_splits_registry_repository_and_tag() -> None:
    reference = parse_image_reference(IMAGE)

    assert reference.registry == REGISTRY
    assert reference.repository == REPOSITORY
    assert reference.reference == TAG
    assert reference.manifest_url == MANIFEST_URL
    assert reference.pull_scope == f"repository:{REPOSITORY}:pull"


def test_parse_handles_a_docker_prefix_and_a_digest() -> None:
    reference = parse_image_reference(f"docker:{REGISTRY}/{REPOSITORY}@sha256:abc123")

    assert reference.reference == "sha256:abc123"


def test_parse_defaults_a_missing_tag_to_latest() -> None:
    assert parse_image_reference(f"{REGISTRY}/{REPOSITORY}").reference == "latest"


@pytest.mark.parametrize("image", ["", "npa-lerobot:0.5.1", "lerobot"])
def test_parse_rejects_an_unqualified_reference(image: str) -> None:
    with pytest.raises(RegistryPreflightError):
        parse_image_reference(image)


def test_a_pullable_image_completes_the_bearer_exchange() -> None:
    registry = FakeRegistry(manifest_status=200, manifest_body=b"{}")

    check = check_image_pull(IMAGE, username="iam", password="iam-token", fetcher=registry)

    assert check.ok is True
    assert check.status == "ok"
    assert check.http_status == 200
    # Anonymous probe, token exchange, then the authenticated manifest fetch.
    assert len(registry.calls) == 3
    token_url = registry.calls[1][0]
    assert f"scope=repository%3A{REPOSITORY.replace('/', '%2F')}%3Apull" in token_url
    assert registry.calls[2][1]["Authorization"] == "Bearer scoped-bearer"


def test_a_403_is_reported_as_forbidden_with_the_pull_permission_remedy() -> None:
    registry = FakeRegistry(
        manifest_status=403,
        manifest_body=_error_body("DENIED", "requested access to the resource is denied"),
    )

    check = check_image_pull(IMAGE, username="iam", password="iam-token", fetcher=registry)

    assert check.ok is False
    assert check.status == "forbidden"
    assert check.http_status == 403
    assert "DENIED" in check.detail
    # The whole point: a readable tag list does not prove a pull will work.
    assert "list tags is a different permission" in check.remedy
    assert "ImagePullBackOff" in check.remedy


def test_a_missing_tag_is_reported_as_not_found_not_as_a_permission_problem() -> None:
    registry = FakeRegistry(
        manifest_status=404,
        manifest_body=_error_body("MANIFEST_UNKNOWN", "manifest unknown"),
    )

    check = check_image_pull(IMAGE, username="iam", password="iam-token", fetcher=registry)

    assert check.status == "not_found"
    assert TAG in check.remedy


def test_rejected_credentials_are_reported_as_unauthorized() -> None:
    registry = FakeRegistry(manifest_status=200, token_status=401)

    check = check_image_pull(IMAGE, username="iam", password="stale-token", fetcher=registry)

    assert check.status == "unauthorized"
    assert check.http_status == 401
    assert "UNAUTHORIZED" in check.detail
    assert "profile" in check.remedy


def test_a_challenge_without_credentials_tries_an_anonymous_token_first() -> None:
    # Previously this reported no_credentials outright, which is wrong for any
    # registry that grants anonymous pulls -- the public mirror among them.
    registry = FakeRegistry(manifest_status=200)

    check = check_image_pull(IMAGE, fetcher=registry)

    assert check.ok is True


def test_a_network_failure_is_not_mistaken_for_a_permission_failure() -> None:
    def broken(url: str, headers: dict[str, str], timeout: int):
        raise OSError("Name or service not known")

    check = check_image_pull(IMAGE, password="iam-token", fetcher=broken)

    assert check.status == "unreachable"
    assert "Name or service not known" in check.detail


def test_an_unparsable_reference_does_not_raise() -> None:
    check = check_image_pull("npa-lerobot:0.5.1")

    assert check.status == "invalid"
    assert check.ok is False


def test_render_includes_the_remedy() -> None:
    registry = FakeRegistry(manifest_status=403)

    rendered = check_image_pull(
        IMAGE, password="iam-token", fetcher=registry
    ).render()

    assert "forbidden (HTTP 403)" in rendered
    assert "Suggested action:" in rendered


def test_each_distinct_image_is_checked_once() -> None:
    registry = FakeRegistry(manifest_status=200, manifest_body=b"{}")

    checks = check_image_pulls([IMAGE, IMAGE, "", f"{REGISTRY}/other/repo:1"], password="t", fetcher=registry)

    assert [check.image for check in checks] == [IMAGE, f"{REGISTRY}/other/repo:1"]


def test_credentials_come_from_the_same_env_the_render_path_injects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKYPILOT_DOCKER_USERNAME", "svc")
    monkeypatch.setenv("SKYPILOT_DOCKER_PASSWORD", "injected-token")

    assert resolve_registry_credentials(mint=False) == ("svc", "injected-token")


def test_credentials_default_to_the_iam_user_and_do_not_mint_when_asked_not_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SKYPILOT_DOCKER_USERNAME", raising=False)
    monkeypatch.delenv("SKYPILOT_DOCKER_PASSWORD", raising=False)
    monkeypatch.delenv("NPA_REGISTRY_USERNAME", raising=False)
    monkeypatch.delenv("NPA_REGISTRY_PASSWORD", raising=False)

    assert resolve_registry_credentials(mint=False) == ("iam", "")


# --- public registries issue anonymous pull tokens ----------------------------


class AnonymousRegistry:
    """A public registry: the token endpoint serves callers with no credentials."""

    def __init__(self) -> None:
        self.token_auth_headers: dict[str, str] | None = None

    def __call__(self, url: str, headers: dict[str, str], timeout: int):
        if "/v2/token" in url or "/token" in url:
            self.token_auth_headers = dict(headers)
            return 200, {}, json.dumps({"token": "anon"}).encode()
        if "Authorization" not in headers:
            return 401, {"www-authenticate": 'Bearer realm="https://ghcr.io/token",service="ghcr.io"'}, b""
        return 200, {}, b"{}"


def test_a_public_image_is_pullable_without_credentials() -> None:
    # "No credentials" is not "cannot pull": GHCR/Docker Hub hand a pull token to
    # an anonymous caller, and the public mirror is exactly how a consumer avoids
    # building multi-GB images at all.
    registry = AnonymousRegistry()

    check = check_image_pull(
        "ghcr.io/nebius/nebius-physical-ai/npa-cosmos-curate:0.1.2", fetcher=registry
    )

    assert check.ok is True
    assert check.status == "ok"
    # The anonymous exchange carries no Authorization header.
    assert registry.token_auth_headers == {}


def test_credentials_are_still_sent_when_present() -> None:
    registry = AnonymousRegistry()

    check_image_pull(IMAGE, username="iam", password="tok", fetcher=registry)

    assert "Authorization" in (registry.token_auth_headers or {})


def test_a_private_registry_that_refuses_an_anonymous_token_still_says_so() -> None:
    def refuses(url: str, headers: dict[str, str], timeout: int):
        if "/v2/token/" in url:
            return 401, {}, _error_body("UNAUTHORIZED", "authentication required")
        return 401, dict(CHALLENGE), b""

    check = check_image_pull(IMAGE, fetcher=refuses)

    assert check.status == "no_credentials"
    assert "SKYPILOT_DOCKER_PASSWORD" in check.remedy
