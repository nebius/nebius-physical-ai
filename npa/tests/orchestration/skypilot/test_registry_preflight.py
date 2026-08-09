from __future__ import annotations

import base64
import json
import subprocess
import urllib.parse
import urllib.request

import pytest

from npa.orchestration.skypilot.registry_preflight import (
    _RegistryRedirectHandler,
    RegistryPreflightError,
    check_image_pull,
    check_image_pulls,
    check_image_pulls_with_credentials,
    parse_image_reference,
    resolve_registry_credentials,
    verify_kubernetes_pull_secret,
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


def test_registry_redirect_strips_authorization_cross_origin() -> None:
    request = urllib.request.Request(
        "https://cr.us-central1.nebius.cloud/v2/repo/blobs/sha256:a",
        headers={"Authorization": "Bearer secret", "Accept": "application/json"},
    )
    redirected = _RegistryRedirectHandler().redirect_request(
        request,
        None,
        307,
        "Temporary Redirect",
        {},
        "https://storage.us-central1.nebius.cloud/signed-object",
    )

    assert redirected is not None
    assert redirected.get_header("Authorization") is None
    assert redirected.get_header("Accept") == "application/json"


class FakeRegistry:
    """A Docker Registry v2 endpoint with a scriptable manifest response."""

    def __init__(
        self,
        *,
        manifest_status: int,
        manifest_body: bytes = b"",
        token_status: int = 200,
    ):
        self.manifest_status = manifest_status
        self.manifest_body = manifest_body
        self.token_status = token_status
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, headers: dict[str, str], timeout: int):
        self.calls.append((url, dict(headers)))
        if "/v2/token/" in url:
            if self.token_status >= 400:
                body = json.dumps(
                    {
                        "errors": [
                            {
                                "code": "UNAUTHORIZED",
                                "message": "authentication required",
                            }
                        ]
                    }
                ).encode()
                return self.token_status, {}, body
            return (
                self.token_status,
                {},
                json.dumps({"token": "scoped-bearer"}).encode(),
            )
        if "Authorization" not in headers:
            return 401, dict(CHALLENGE), b""
        return self.manifest_status, {}, self.manifest_body


def _error_body(code: str, message: str) -> bytes:
    return json.dumps(
        {"errors": [{"code": code, "message": message, "detail": None}]}
    ).encode()


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

    check = check_image_pull(
        IMAGE, username="iam", password="iam-token", fetcher=registry
    )

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
        manifest_body=_error_body(
            "DENIED", "requested access to the resource is denied"
        ),
    )

    check = check_image_pull(
        IMAGE, username="iam", password="iam-token", fetcher=registry
    )

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

    check = check_image_pull(
        IMAGE, username="iam", password="iam-token", fetcher=registry
    )

    assert check.status == "not_found"
    assert TAG in check.remedy


def test_rejected_credentials_are_reported_as_unauthorized() -> None:
    registry = FakeRegistry(manifest_status=200, token_status=401)

    check = check_image_pull(
        IMAGE, username="iam", password="stale-token", fetcher=registry
    )

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


def test_token_challenge_preserves_existing_realm_query_parameters() -> None:
    calls: list[str] = []

    def registry(url: str, headers: dict[str, str], timeout: int):
        calls.append(url)
        if "/token" in url:
            return 200, {}, json.dumps({"token": "anon"}).encode()
        if "Authorization" not in headers:
            return (
                401,
                {
                    "www-authenticate": (
                        'Bearer realm="https://ghcr.io/token?client_id=npa",service="ghcr.io"'
                    )
                },
                b"",
            )
        return 200, {}, b"{}"

    check = check_image_pull(
        "ghcr.io/nebius/nebius-physical-ai/npa-cosmos-curate:0.1.2",
        fetcher=registry,
    )

    assert check.ok
    token_query = urllib.parse.parse_qs(urllib.parse.urlsplit(calls[1]).query)
    assert token_query["client_id"] == ["npa"]
    assert token_query["service"] == ["ghcr.io"]
    assert token_query["scope"] == [
        "repository:nebius/nebius-physical-ai/npa-cosmos-curate:pull"
    ]


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

    rendered = check_image_pull(IMAGE, password="iam-token", fetcher=registry).render()

    assert "forbidden (HTTP 403)" in rendered
    assert "Suggested action:" in rendered


def test_each_distinct_image_is_checked_once() -> None:
    registry = FakeRegistry(manifest_status=200, manifest_body=b"{}")

    checks = check_image_pulls(
        [IMAGE, IMAGE, "", f"{REGISTRY}/other/repo:1"], password="t", fetcher=registry
    )

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
            return (
                401,
                {
                    "www-authenticate": 'Bearer realm="https://ghcr.io/token",service="ghcr.io"'
                },
                b"",
            )
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


def test_public_registry_never_receives_foreign_nebius_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKYPILOT_DOCKER_SERVER", REGISTRY)
    monkeypatch.setenv("SKYPILOT_DOCKER_PASSWORD", "nebius-iam-token")
    registry = AnonymousRegistry()

    checks = check_image_pulls_with_credentials(
        ["ghcr.io/nebius/nebius-physical-ai/npa-cosmos-curate:0.1.2"],
        fetcher=registry,
    )

    assert checks[0].ok
    assert registry.token_auth_headers == {}


def test_matching_private_registry_uses_configured_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_REGISTRY_SERVER", REGISTRY)
    monkeypatch.setenv("NPA_REGISTRY_USERNAME", "svc")
    monkeypatch.setenv("NPA_REGISTRY_PASSWORD", "private-token")
    registry = FakeRegistry(manifest_status=200)

    checks = check_image_pulls_with_credentials([IMAGE], fetcher=registry)

    assert checks[0].ok
    assert registry.calls[1][1]["Authorization"].startswith("Basic ")


def test_npa_registry_itself_scopes_private_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_REGISTRY", f"{REGISTRY}/project-repository")
    monkeypatch.setenv("NPA_REGISTRY_USERNAME", "svc")
    monkeypatch.setenv("NPA_REGISTRY_PASSWORD", "private-token")
    registry = FakeRegistry(manifest_status=200)

    checks = check_image_pulls_with_credentials([IMAGE], fetcher=registry)

    assert checks[0].ok
    assert registry.calls[1][1]["Authorization"].startswith("Basic ")


def test_other_nebius_registry_mints_fresh_token_instead_of_using_ambient_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "NPA_REGISTRY", "cr.eu-north1.nebius.cloud/project-repository"
    )
    monkeypatch.setenv("NPA_REGISTRY_USERNAME", "wrong-user")
    monkeypatch.setenv("NPA_REGISTRY_PASSWORD", "wrong-project-token")
    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth.mint_nebius_registry_token",
        lambda: "fresh-target-token",
    )

    assert resolve_registry_credentials(
        "cr.us-central1.nebius.cloud", mint=True
    ) == ("iam", "fresh-target-token")


def test_a_private_registry_that_refuses_an_anonymous_token_still_says_so() -> None:
    def refuses(url: str, headers: dict[str, str], timeout: int):
        if "/v2/token/" in url:
            return 401, {}, _error_body("UNAUTHORIZED", "authentication required")
        return 401, dict(CHALLENGE), b""

    check = check_image_pull(IMAGE, fetcher=refuses)

    assert check.status == "no_credentials"
    assert "SKYPILOT_DOCKER_PASSWORD" in check.remedy


def _docker_secret_result(registry: str, *, name: str = "pull-secret"):
    config = base64.b64encode(
        json.dumps({"auths": {registry: {"auth": "redacted-test-value"}}}).encode()
    ).decode()
    payload = {
        "metadata": {"name": name},
        "type": "kubernetes.io/dockerconfigjson",
        "data": {".dockerconfigjson": config},
    }
    return subprocess.CompletedProcess(
        ["kubectl"], 0, stdout=json.dumps(payload), stderr=""
    )


@pytest.mark.parametrize(
    "registry",
    ["ghcr.io", "123456789012.dkr.ecr.us-east-1.amazonaws.com", "oci.example.test"],
)
def test_verified_target_pull_secret_can_satisfy_private_foreign_registry(
    registry: str,
) -> None:
    image = f"{registry}/team/private:1"

    def operator_unreachable(url, headers, timeout):  # noqa: ANN001
        raise OSError("operator route blocked")

    checks = check_image_pulls_with_credentials(
        [image],
        mint=False,
        fetcher=operator_unreachable,
        pull_secret_names=("pull-secret",),
        context="target-context",
        secret_runner=lambda *args, **kwargs: _docker_secret_result(registry),
    )

    assert checks[0].ok
    assert checks[0].operator_status == "unreachable"
    assert checks[0].target_status == "verified_pull_secret"
    assert checks[0].authority == "kubernetes_image_pull_secret"
    assert "redacted-test-value" not in checks[0].render()


def test_missing_or_rbac_denied_pull_secret_is_not_target_pull_proof() -> None:
    image = "ghcr.io/team/private:1"

    def operator_unreachable(url, headers, timeout):  # noqa: ANN001
        raise OSError("operator route blocked")

    checks = check_image_pulls_with_credentials(
        [image],
        mint=False,
        fetcher=operator_unreachable,
        pull_secret_names=("missing-secret",),
        secret_runner=lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="forbidden: cannot get secret"
        ),
    )

    assert checks[0].status == "target_pull_unverified"
    assert checks[0].target_status == "unverified"
    assert "forbidden" in checks[0].detail


def test_invalid_pull_secret_reference_runs_no_kubectl() -> None:
    called = False

    def runner(cmd, **kwargs):  # noqa: ANN001
        nonlocal called
        called = True
        raise AssertionError("kubectl ran")

    verified, detail = verify_kubernetes_pull_secret(
        "ghcr.io", ("../../secret",), runner=runner
    )

    assert verified is False
    assert "invalid secret reference" in detail
    assert called is False


def test_target_secret_with_wrong_registry_is_unverified() -> None:
    verified, detail = verify_kubernetes_pull_secret(
        "ghcr.io",
        ("pull-secret",),
        runner=lambda *args, **kwargs: _docker_secret_result("oci.example.test"),
    )

    assert verified is False
    assert "does not cover registry ghcr.io" in detail


def test_target_secret_with_empty_auth_entry_is_unverified() -> None:
    config = base64.b64encode(json.dumps({"auths": {"ghcr.io": {}}}).encode()).decode()
    payload = {
        "type": "kubernetes.io/dockerconfigjson",
        "data": {".dockerconfigjson": config},
    }

    verified, detail = verify_kubernetes_pull_secret(
        "ghcr.io",
        ("pull-secret",),
        runner=lambda *args, **kwargs: subprocess.CompletedProcess(
            ["kubectl"], 0, stdout=json.dumps(payload), stderr=""
        ),
    )

    assert verified is False
    assert "contains no usable credential fields" in detail
