from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import urllib.parse
import urllib.request

import pytest

from npa.orchestration.skypilot.registry_preflight import (
    _RegistryRedirectHandler,
    KubernetesPullCheck,
    RegistryPreflightError,
    check_image_pull,
    check_image_pulls,
    check_image_pulls_with_credentials,
    fetch_image_config_metadata,
    parse_image_reference,
    resolve_kubernetes_pull_target,
    resolve_registry_credentials,
    verify_kubernetes_image_pull,
    verify_kubernetes_pull_secret,
)


REGISTRY = "registry-us.example"
REPOSITORY = "u00j7q4jjkahvsx0jy/npa-cosmos2-transfer"
TAG = "2.5.1-golden-eval-smoke-20260616T033000Z"
IMAGE = f"{REGISTRY}/{REPOSITORY}:{TAG}"
DIGEST = f"sha256:{'a' * 64}"
MANIFEST_URL = f"https://{REGISTRY}/v2/{REPOSITORY}/manifests/{TAG}"
# Representative Docker Registry v2 bearer challenge.
CHALLENGE = {
    "www-authenticate": (
        f'Bearer realm="https://{REGISTRY}/v2/token/",service="{REGISTRY}"'
    )
}


def test_registry_redirect_strips_authorization_cross_origin() -> None:
    request = urllib.request.Request(
        "https://registry-us.example/v2/repo/blobs/sha256:a",
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
        manifest_headers: dict[str, str] | None = None,
        token_status: int = 200,
    ):
        self.manifest_status = manifest_status
        self.manifest_body = manifest_body
        self.manifest_headers = manifest_headers or {}
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
        return self.manifest_status, self.manifest_headers, self.manifest_body


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


def test_docker_hub_uses_distinct_registry_api_host() -> None:
    reference = parse_image_reference("docker.io/vllm/vllm-omni:cosmos3")

    assert reference.registry == "docker.io"
    assert reference.api_registry == "registry-1.docker.io"
    assert reference.manifest_url == (
        "https://registry-1.docker.io/v2/vllm/vllm-omni/manifests/cosmos3"
    )


def test_parse_handles_a_docker_prefix_and_a_digest() -> None:
    reference = parse_image_reference(f"docker:{REGISTRY}/{REPOSITORY}@sha256:abc123")

    assert reference.reference == "sha256:abc123"


def test_parse_strips_tag_from_tag_and_digest_reference() -> None:
    reference = parse_image_reference(f"{REGISTRY}/{REPOSITORY}:{TAG}@sha256:abc123")

    assert reference.repository == REPOSITORY
    assert reference.reference == "sha256:abc123"
    assert reference.manifest_url == (
        f"https://{REGISTRY}/v2/{REPOSITORY}/manifests/sha256:abc123"
    )
    assert reference.pull_scope == f"repository:{REPOSITORY}:pull"


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
    assert "registry's standard authentication flow" in check.remedy
    assert "profile" not in check.remedy


def test_a_challenge_without_credentials_tries_an_anonymous_token_first() -> None:
    # Previously this reported no_credentials outright, which is wrong for any
    # registry that grants anonymous pulls -- the public release channel among them.
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
    assert check.detail == "registry manifest request unavailable (OSError)"
    assert "Name or service not known" not in check.render()


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


def test_credentials_default_to_anonymous_and_never_mint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SKYPILOT_DOCKER_USERNAME", raising=False)
    monkeypatch.delenv("SKYPILOT_DOCKER_PASSWORD", raising=False)
    monkeypatch.delenv("NPA_REGISTRY_USERNAME", raising=False)
    monkeypatch.delenv("NPA_REGISTRY_PASSWORD", raising=False)

    assert resolve_registry_credentials(mint=False) == ("", "")


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
    # an anonymous caller, and the public release channel is exactly how a consumer avoids
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


def test_official_public_image_ignores_matching_stale_ghcr_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_REGISTRY", "ghcr.io/operator/private")
    monkeypatch.setenv("NPA_REGISTRY_USERNAME", "stale-user")
    monkeypatch.setenv("NPA_REGISTRY_PASSWORD", "stale-token")
    registry = AnonymousRegistry()
    public_image = "ghcr.io/nebius/nebius-physical-ai/npa-cosmos-curate:0.1.2"

    checks = check_image_pulls_with_credentials(
        [public_image],
        fetcher=registry,
        secret_runner=lambda *args, **kwargs: pytest.fail(
            "public path must not query Kubernetes secrets"
        ),
    )

    assert checks[0].ok
    assert registry.token_auth_headers == {}


def test_matching_private_registry_uses_configured_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_REGISTRY_SERVER", REGISTRY)
    monkeypatch.setenv("NPA_REGISTRY_USERNAME", "svc")
    monkeypatch.setenv("NPA_REGISTRY_PASSWORD", "private-token")
    registry = FakeRegistry(
        manifest_status=200,
        manifest_headers={"docker-content-digest": DIGEST},
    )

    checks = check_image_pulls_with_credentials(
        [IMAGE],
        fetcher=registry,
        secret_runner=lambda *args, **kwargs: pytest.fail(
            "credentialed VM path must not query Kubernetes secrets"
        ),
    )

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


def test_other_registry_never_receives_mismatched_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_REGISTRY", "registry.example/project-repository")
    monkeypatch.setenv("NPA_REGISTRY_USERNAME", "wrong-user")
    monkeypatch.setenv("NPA_REGISTRY_PASSWORD", "wrong-project-token")
    assert resolve_registry_credentials("registry-us.example", mint=True) == ("", "")


def test_a_private_registry_that_refuses_an_anonymous_token_still_says_so() -> None:
    def refuses(url: str, headers: dict[str, str], timeout: int):
        if "/v2/token/" in url:
            return 401, {}, _error_body("UNAUTHORIZED", "authentication required")
        return 401, dict(CHALLENGE), b""

    check = check_image_pull(IMAGE, fetcher=refuses)

    assert check.status == "no_credentials"
    assert "SKYPILOT_DOCKER_PASSWORD" in check.remedy


def _docker_secret_result(registry: str, *, name: str = "pull-secret"):
    auth = base64.b64encode(b"target-user:target-password").decode()
    config = base64.b64encode(
        json.dumps({"auths": {registry: {"auth": auth}}}).encode()
    ).decode()
    payload = {
        "metadata": {"name": name},
        "type": "kubernetes.io/dockerconfigjson",
        "data": {".dockerconfigjson": config},
    }
    return subprocess.CompletedProcess(
        ["kubectl"], 0, stdout=json.dumps(payload), stderr=""
    )


def _verified_target_pull(**kwargs) -> KubernetesPullCheck:  # noqa: ANN003
    assert kwargs["namespace"] == "default"
    assert kwargs["context"] == "target-context"
    return KubernetesPullCheck(status="verified", digest=DIGEST)


def _configure_private_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_REGISTRY_SERVER", REGISTRY)
    monkeypatch.setenv("NPA_REGISTRY_USERNAME", "svc")
    monkeypatch.setenv("NPA_REGISTRY_PASSWORD", "private-token")


def test_host_credentials_do_not_replace_declared_target_pull_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_private_registry(monkeypatch)
    registry = FakeRegistry(
        manifest_status=200,
        manifest_headers={"docker-content-digest": DIGEST},
    )
    target_calls: list[list[str]] = []

    def missing_in_selected_target(cmd, **kwargs):  # noqa: ANN001
        target_calls.append(cmd)
        assert cmd[:5] == [
            "kubectl",
            "--context",
            "wrong-context",
            "--namespace",
            "wrong-namespace",
        ]
        return subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="secret not found in selected target"
        )

    checks = check_image_pulls_with_credentials(
        [IMAGE],
        fetcher=registry,
        pull_secrets_by_image={IMAGE: ("missing-secret",)},
        context="wrong-context",
        namespace="wrong-namespace",
        secret_runner=missing_in_selected_target,
    )

    assert len(target_calls) == 1
    assert checks[0].status == "target_pull_unverified"
    assert checks[0].operator_status == "verified"
    assert checks[0].target_status == "pull_secret_unverified"
    assert checks[0].authority == "none"
    assert checks[0].http_status == 200
    assert checks[0].digest == DIGEST
    assert "private-token" not in checks[0].render()


@pytest.mark.parametrize(
    "diagnostic",
    (
        "Authorization: Bearer synthetic-review-secret",
        '{"kind":"Secret","data":{".dockerconfigjson":"synthetic-review-secret"}}',
        "NPA_REGISTRY_PASSWORD=synthetic-review-secret",
        'error: "Bearer synthetic-review-secret"',
    ),
)
def test_host_credentials_bound_target_lookup_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    diagnostic: str,
) -> None:
    _configure_private_registry(monkeypatch)
    secret = "synthetic-review-secret"

    checks = check_image_pulls_with_credentials(
        [IMAGE],
        fetcher=FakeRegistry(manifest_status=200),
        pull_secret_names=("unreadable-secret",),
        context="target-context",
        namespace="default",
        secret_runner=lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd,
            1,
            stdout="",
            stderr=diagnostic,
        ),
    )

    assert checks[0].status == "target_pull_unverified"
    assert secret not in checks[0].render()
    assert ".dockerconfigjson" not in checks[0].render()
    assert "Kubernetes rejected the secret lookup (exit 1)" in checks[0].detail


def test_target_lookup_exception_diagnostic_is_bounded() -> None:
    secret = "synthetic-exception-secret"

    def unavailable(*args, **kwargs):  # noqa: ANN001
        raise OSError(f"transport rejected Bearer {secret}")

    verified, detail = verify_kubernetes_pull_secret(
        REGISTRY,
        ("unreadable-secret",),
        context="target-context",
        runner=unavailable,
    )

    assert verified is False
    assert secret not in detail
    assert "Kubernetes inventory unavailable (OSError)" in detail


def test_host_credentials_require_declared_secret_to_have_docker_config_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_private_registry(monkeypatch)
    config = base64.b64encode(
        json.dumps({"auths": {REGISTRY: {"auth": "redacted-test-value"}}}).encode()
    ).decode()
    payload = {
        "type": "Opaque",
        "data": {".dockerconfigjson": config},
    }

    checks = check_image_pulls_with_credentials(
        [IMAGE],
        fetcher=FakeRegistry(
            manifest_status=200,
            manifest_headers={"docker-content-digest": DIGEST},
        ),
        pull_secret_names=("wrong-type",),
        context="target-context",
        namespace="default",
        secret_runner=lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps(payload), stderr=""
        ),
    )

    assert checks[0].status == "target_pull_unverified"
    assert "not a kubernetes.io/dockerconfigjson secret" in checks[0].detail
    assert "redacted-test-value" not in checks[0].render()


def test_host_credentials_require_declared_secret_to_cover_image_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_private_registry(monkeypatch)

    checks = check_image_pulls_with_credentials(
        [IMAGE],
        fetcher=FakeRegistry(manifest_status=200),
        pull_secret_names=("wrong-registry",),
        context="target-context",
        namespace="default",
        secret_runner=lambda *args, **kwargs: _docker_secret_result(
            "oci.example.test", name="wrong-registry"
        ),
    )

    assert checks[0].status == "target_pull_unverified"
    assert f"does not cover registry {REGISTRY}" in checks[0].detail


def test_host_and_declared_target_authorities_are_both_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_private_registry(monkeypatch)
    target_calls = 0

    def valid_target(*args, **kwargs):  # noqa: ANN001
        nonlocal target_calls
        target_calls += 1
        return _docker_secret_result(REGISTRY)

    checks = check_image_pulls_with_credentials(
        [IMAGE],
        fetcher=FakeRegistry(
            manifest_status=200,
            manifest_headers={"docker-content-digest": DIGEST},
        ),
        pull_secret_names=("pull-secret",),
        context="target-context",
        namespace="default",
        secret_runner=valid_target,
        target_pull_verifier=_verified_target_pull,
    )

    assert target_calls == 1
    assert checks[0].ok
    assert checks[0].operator_status == "verified"
    assert checks[0].target_status == "verified_pull_secret"
    assert checks[0].authority == "kubernetes_image_pull_secret"
    assert checks[0].http_status == 200
    assert checks[0].digest == DIGEST


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
        namespace="default",
        secret_runner=lambda *args, **kwargs: _docker_secret_result(registry),
        target_pull_verifier=_verified_target_pull,
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
        context="target-context",
        namespace="default",
        secret_runner=lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="forbidden: cannot get secret"
        ),
    )

    assert checks[0].status == "target_pull_unverified"
    assert checks[0].target_status == "pull_secret_unverified"
    assert "Kubernetes rejected the secret lookup (exit 1)" in checks[0].detail


def test_invalid_pull_secret_reference_runs_no_kubectl() -> None:
    called = False

    def runner(cmd, **kwargs):  # noqa: ANN001
        nonlocal called
        called = True
        raise AssertionError("kubectl ran")

    verified, detail = verify_kubernetes_pull_secret(
        "ghcr.io", ("../../secret",), context="target-context", runner=runner
    )

    assert verified is False
    assert "invalid secret reference" in detail
    assert called is False


def test_target_secret_with_wrong_registry_is_unverified() -> None:
    verified, detail = verify_kubernetes_pull_secret(
        "ghcr.io",
        ("pull-secret",),
        context="target-context",
        runner=lambda *args, **kwargs: _docker_secret_result("oci.example.test"),
    )

    assert verified is False
    assert "does not cover registry ghcr.io" in detail


@pytest.mark.parametrize(
    "docker_config_registry",
    (
        "docker.io",
        "https://index.docker.io/v1/",
        "registry-1.docker.io",
    ),
)
def test_docker_hub_pull_secret_registry_aliases_are_equivalent(
    docker_config_registry: str,
) -> None:
    verified, detail = verify_kubernetes_pull_secret(
        "docker.io",
        ("pull-secret",),
        context="target-context",
        runner=lambda *args, **kwargs: _docker_secret_result(docker_config_registry),
    )

    assert verified is True, detail


def test_target_secret_with_empty_auth_entry_is_unverified() -> None:
    config = base64.b64encode(json.dumps({"auths": {"ghcr.io": {}}}).encode()).decode()
    payload = {
        "type": "kubernetes.io/dockerconfigjson",
        "data": {".dockerconfigjson": config},
    }

    verified, detail = verify_kubernetes_pull_secret(
        "ghcr.io",
        ("pull-secret",),
        context="target-context",
        runner=lambda *args, **kwargs: subprocess.CompletedProcess(
            ["kubectl"], 0, stdout=json.dumps(payload), stderr=""
        ),
    )

    assert verified is False
    assert "contains no usable credential fields" in detail


def test_kubernetes_private_path_without_secret_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_private_registry(monkeypatch)

    checks = check_image_pulls_with_credentials(
        [IMAGE],
        fetcher=FakeRegistry(manifest_status=200),
        pull_secrets_by_image={IMAGE: ()},
        operator_images=set(),
        kubernetes_images={IMAGE},
        context="target-context",
        namespace="default",
        target_pull_verifier=lambda **kwargs: pytest.fail(
            "private Kubernetes paths need declared delivery before probing"
        ),
    )

    assert checks[0].status == "target_pull_unverified"
    assert checks[0].target_status == "pull_secret_required"
    assert checks[0].authority == "none"


def test_host_and_target_must_resolve_the_same_immutable_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_private_registry(monkeypatch)
    other_digest = f"sha256:{'b' * 64}"

    checks = check_image_pulls_with_credentials(
        [IMAGE],
        fetcher=FakeRegistry(
            manifest_status=200,
            manifest_headers={"docker-content-digest": DIGEST},
        ),
        pull_secret_names=("pull-secret",),
        context="target-context",
        namespace="default",
        secret_runner=lambda *args, **kwargs: _docker_secret_result(REGISTRY),
        target_pull_verifier=lambda **kwargs: KubernetesPullCheck(
            status="verified", digest=other_digest
        ),
    )

    assert checks[0].status == "target_pull_unverified"
    assert checks[0].target_status == "digest_mismatch"
    assert checks[0].digest == DIGEST


def test_target_pull_and_cleanup_failures_are_both_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_private_registry(monkeypatch)

    checks = check_image_pulls_with_credentials(
        [IMAGE],
        fetcher=FakeRegistry(manifest_status=200),
        pull_secret_names=("pull-secret",),
        context="target-context",
        namespace="default",
        secret_runner=lambda *args, **kwargs: _docker_secret_result(REGISTRY),
        target_pull_verifier=lambda **kwargs: KubernetesPullCheck(
            status="image_pull_failed",
            cleanup_status="unverified",
        ),
    )

    assert checks[0].status == "target_pull_unverified"
    assert checks[0].target_status == "image_pull_failed"
    assert "target image pull failed" in checks[0].detail
    assert "cleanup was not verified" in checks[0].detail


def test_target_verifier_exception_never_claims_cleanup_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_private_registry(monkeypatch)

    def unavailable(**kwargs):
        raise subprocess.TimeoutExpired(kwargs["image"], 30)

    checks = check_image_pulls_with_credentials(
        [IMAGE],
        fetcher=FakeRegistry(manifest_status=200),
        pull_secret_names=("pull-secret",),
        context="target-context",
        namespace="default",
        secret_runner=lambda *args, **kwargs: _docker_secret_result(REGISTRY),
        target_pull_verifier=unavailable,
    )

    assert checks[0].status == "target_pull_unverified"
    assert checks[0].target_status == "verifier_unavailable_TimeoutExpired"
    assert "cleanup was not verified" in checks[0].detail


def test_vm_private_path_keeps_exact_host_manifest_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_private_registry(monkeypatch)

    checks = check_image_pulls_with_credentials(
        [IMAGE],
        fetcher=FakeRegistry(manifest_status=200),
        pull_secrets_by_image={IMAGE: ()},
        operator_images={IMAGE},
        kubernetes_images=set(),
        target_pull_verifier=lambda **kwargs: pytest.fail(
            "VM path must not create a Kubernetes pull probe"
        ),
    )

    assert checks[0].ok
    assert checks[0].authority == "operator"
    assert checks[0].target_status == "not_applicable"


def test_empty_per_image_authority_without_path_classification_fails_closed() -> None:
    checks = check_image_pulls_with_credentials(
        [IMAGE],
        pull_secrets_by_image={IMAGE: ()},
        fetcher=lambda *args, **kwargs: pytest.fail(
            "unclassified path must fail before a registry request"
        ),
    )

    assert checks[0].status == "target_path_unclassified"
    assert checks[0].authority == "none"


def test_declared_secret_requires_exact_context_before_inventory() -> None:
    checks = check_image_pulls_with_credentials(
        [IMAGE],
        fetcher=FakeRegistry(manifest_status=200),
        pull_secret_names=("pull-secret",),
        namespace="default",
        secret_runner=lambda *args, **kwargs: pytest.fail(
            "ambient kubectl context must never be queried"
        ),
    )

    assert checks[0].status == "target_pull_unverified"
    assert checks[0].target_status == "exact_context_required"
    assert (
        "exact Kubernetes context and effective namespace are required"
        in checks[0].detail
    )


def test_non_base64_auth_field_is_not_target_credential_proof() -> None:
    config = base64.b64encode(
        json.dumps({"auths": {REGISTRY: {"auth": "definitely-not-base64"}}}).encode()
    ).decode()
    secret = {
        "type": "kubernetes.io/dockerconfigjson",
        "data": {".dockerconfigjson": config},
    }

    checks = check_image_pulls_with_credentials(
        [IMAGE],
        fetcher=FakeRegistry(manifest_status=200),
        pull_secret_names=("stale-secret",),
        context="target-context",
        namespace="default",
        secret_runner=lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps(secret), stderr=""
        ),
        target_pull_verifier=lambda **kwargs: pytest.fail(
            "malformed credential must fail before a target pull probe"
        ),
    )

    assert checks[0].status == "target_pull_unverified"
    assert checks[0].target_status == "pull_secret_unverified"
    assert "contains no usable credential fields" in checks[0].detail


def test_host_fetch_exception_text_is_never_rendered() -> None:
    marker = "Authorization: Bearer synthetic-host-secret"

    def unavailable(*args, **kwargs):  # noqa: ANN002,ANN003
        raise OSError(marker)

    check = check_image_pull(IMAGE, fetcher=unavailable)

    assert check.status == "unreachable"
    assert marker not in check.render()
    assert "synthetic-host-secret" not in check.render()
    assert check.detail == "registry manifest request unavailable (OSError)"


@pytest.mark.parametrize("phase", ("token", "authenticated_manifest"))
def test_later_registry_exception_text_is_never_rendered(phase: str) -> None:
    secret = "synthetic-later-registry-secret"

    def unavailable(url: str, headers: dict[str, str], timeout: int):
        if "/v2/token/" in url:
            if phase == "token":
                raise OSError(f"Bearer {secret}")
            return 200, {}, json.dumps({"token": "scoped"}).encode()
        if "Authorization" in headers and phase == "authenticated_manifest":
            raise OSError(f"Authorization: Bearer {secret}")
        return 401, dict(CHALLENGE), b""

    check = check_image_pull(IMAGE, fetcher=unavailable)

    assert check.status == "unreachable"
    assert secret not in check.render()
    assert "OSError" in check.detail


def test_config_metadata_fetch_bounds_transport_exception_text() -> None:
    secret = "synthetic-config-fetch-secret"

    def unavailable(*args, **kwargs):  # noqa: ANN002,ANN003
        raise OSError(f"Bearer {secret}")

    with pytest.raises(RegistryPreflightError) as exc_info:
        fetch_image_config_metadata(IMAGE, fetcher=unavailable)

    assert str(exc_info.value) == "registry manifest request unavailable (OSError)"
    assert secret not in str(exc_info.value)


def test_effective_target_uses_selected_context_namespace() -> None:
    def kubectl(cmd, **kwargs):  # noqa: ANN001
        assert cmd == [
            "kubectl",
            "--context",
            "target-context",
            "config",
            "view",
            "--minify",
            "-o",
            "json",
        ]
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "contexts": [
                        {
                            "name": "target-context",
                            "context": {"namespace": "team-workloads"},
                        }
                    ]
                }
            ),
            stderr="",
        )

    target = resolve_kubernetes_pull_target(context="target-context", runner=kubectl)

    assert target.namespace == "team-workloads"
    assert target.pull_secret_names == ()


def test_effective_target_applies_secret_override_in_kubeconfig_namespace(
    tmp_path,
) -> None:
    config = tmp_path / "sky.yaml"
    config.write_text(
        """
kubernetes:
  namespace: global-namespace
  pod_config:
    spec:
      imagePullSecrets:
        - name: global-secret
        - name: global-fallback
  context_configs:
    target-context:
      namespace: team-namespace
      pod_config:
        spec:
          imagePullSecrets:
            - name: context-secret
""",
        encoding="utf-8",
    )

    def kubectl(cmd, **kwargs):  # noqa: ANN001
        assert cmd == [
            "kubectl",
            "--context",
            "target-context",
            "config",
            "view",
            "--minify",
            "-o",
            "json",
        ]
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "contexts": [
                        {
                            "name": "target-context",
                            "context": {"namespace": "kubeconfig-namespace"},
                        }
                    ]
                }
            ),
            stderr="",
        )

    target = resolve_kubernetes_pull_target(
        context="target-context", global_config_path=config, runner=kubectl
    )

    assert target.namespace == "kubeconfig-namespace"
    assert target.pull_secret_names == ("context-secret", "global-fallback")


@pytest.mark.parametrize(
    "config_text",
    [
        """
kubernetes:
  namespace: team-namespace
  pod_config:
    spec:
      imagePullSecrets: []
""",
        """
kubernetes:
  namespace: team-namespace
  pod_config:
    spec:
      imagePullSecrets:
        - name: global-secret
  context_configs:
    target-context:
      pod_config:
        spec:
          imagePullSecrets: []
""",
    ],
)
def test_effective_target_rejects_explicit_empty_pull_secret_lists(
    tmp_path, config_text: str
) -> None:
    config = tmp_path / "sky.yaml"
    config.write_text(config_text, encoding="utf-8")

    with pytest.raises(RegistryPreflightError, match="must not be empty"):
        resolve_kubernetes_pull_target(
            context="target-context",
            global_config_path=config,
            runner=lambda *args, **kwargs: pytest.fail(
                "explicit SkyPilot namespace must not consult ambient context"
            ),
        )


def test_target_namespace_lookup_never_renders_kubectl_diagnostics() -> None:
    marker = "Authorization: Bearer synthetic-namespace-secret"

    with pytest.raises(RegistryPreflightError) as exc_info:
        resolve_kubernetes_pull_target(
            context="target-context",
            runner=lambda cmd, **kwargs: subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr=marker
            ),
        )

    assert marker not in str(exc_info.value)
    assert str(exc_info.value).endswith("(exit 1)")


def test_anonymous_host_success_still_requires_exact_target_probe() -> None:
    calls: list[tuple[str, ...]] = []

    def target_pull(**kwargs):
        calls.append(kwargs["secret_names"])
        return KubernetesPullCheck(status="verified", digest=DIGEST)

    checks = check_image_pulls_with_credentials(
        [IMAGE],
        fetcher=FakeRegistry(
            manifest_status=200,
            manifest_headers={"docker-content-digest": DIGEST},
        ),
        pull_secret_sets_by_image={IMAGE: ((),)},
        operator_images=set(),
        kubernetes_images={IMAGE},
        namespace="team-namespace",
        context="target-context",
        target_pull_verifier=target_pull,
    )

    assert calls == [()]
    assert checks[0].ok
    assert checks[0].authority == "kubernetes_target_pull"


def test_every_distinct_kubernetes_secret_set_is_probed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_private_registry(monkeypatch)
    probed: list[tuple[str, ...]] = []

    def target_pull(**kwargs):
        probed.append(kwargs["secret_names"])
        return KubernetesPullCheck(status="verified", digest=DIGEST)

    checks = check_image_pulls_with_credentials(
        [IMAGE],
        fetcher=FakeRegistry(
            manifest_status=200,
            manifest_headers={"docker-content-digest": DIGEST},
        ),
        pull_secret_sets_by_image={IMAGE: (("path-a-secret",), ("path-b-secret",))},
        operator_images=set(),
        kubernetes_images={IMAGE},
        namespace="team-namespace",
        context="target-context",
        secret_runner=lambda *args, **kwargs: _docker_secret_result(REGISTRY),
        target_pull_verifier=target_pull,
    )

    assert checks[0].ok
    assert checks[0].target_status == "verified_target_paths"
    assert probed == [("path-a-secret",), ("path-b-secret",)]


def _target_probe_runner(
    *,
    delete_exit: int = 0,
    image_id: str = f"docker-pullable://{REGISTRY}/{REPOSITORY}@{DIGEST}",
    waiting_reason: str = "",
    active_deadline_seconds: int | None = 30,
):
    calls: list[list[str]] = []
    deleted = False
    name = f"npa-pull-{hashlib.sha256(IMAGE.encode()).hexdigest()[:12]}-abc123"
    payload = {
        "metadata": {
            "name": name,
            "uid": "probe-uid",
            "labels": {
                "npa.nebius.com/owned": "true",
                "npa.nebius.com/purpose": "image-pull-preflight",
                "npa.nebius.com/probe-id": "abc123",
            },
        },
        "spec": {"containers": [{"name": "pull", "image": IMAGE}]},
        "status": {
            "containerStatuses": [
                {
                    "name": "pull",
                    "imageID": image_id,
                    "state": (
                        {"waiting": {"reason": waiting_reason}}
                        if waiting_reason
                        else {"terminated": {"exitCode": 0}}
                    ),
                }
            ]
        },
    }

    def run(cmd, **kwargs):  # noqa: ANN001
        nonlocal deleted
        calls.append(cmd)
        if "create" in cmd:
            manifest = json.loads(kwargs["input"])
            assert manifest["spec"]["imagePullSecrets"] == [{"name": "pull-secret"}]
            assert manifest["spec"]["tolerations"] == [
                {
                    "key": "nvidia.com/gpu",
                    "operator": "Exists",
                    "effect": "NoSchedule",
                }
            ]
            if active_deadline_seconds is None:
                assert "activeDeadlineSeconds" not in manifest["spec"]
            else:
                assert (
                    manifest["spec"]["activeDeadlineSeconds"] == active_deadline_seconds
                )
            assert ".dockerconfigjson" not in kwargs["input"]
            return subprocess.CompletedProcess(
                cmd, 0, stdout=json.dumps(payload), stderr=""
            )
        if "delete" in cmd:
            options = json.loads(kwargs["input"])
            assert options["preconditions"]["uid"] == "probe-uid"
            if delete_exit == 0:
                deleted = True
            return subprocess.CompletedProcess(cmd, delete_exit, stdout="", stderr="")
        if deleted and "--ignore-not-found=true" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps(payload), stderr=""
        )

    return run, calls


def test_exact_target_pull_probe_verifies_image_id_and_owned_cleanup() -> None:
    run, calls = _target_probe_runner()

    check = verify_kubernetes_image_pull(
        image=IMAGE,
        secret_names=("pull-secret",),
        namespace="target-namespace",
        context="target-context",
        timeout_seconds=30,
        runner=run,
        nonce_factory=lambda: "abc123",
    )

    assert check == KubernetesPullCheck(status="verified", digest=DIGEST)
    assert all(
        cmd[:5]
        == [
            "kubectl",
            "--context",
            "target-context",
            "--namespace",
            "target-namespace",
        ]
        for cmd in calls
    )
    assert sum("delete" in cmd for cmd in calls) == 1


def test_target_pull_probe_fails_when_owned_cleanup_is_unverified() -> None:
    run, _calls = _target_probe_runner(delete_exit=1)

    check = verify_kubernetes_image_pull(
        image=IMAGE,
        secret_names=("pull-secret",),
        namespace="target-namespace",
        context="target-context",
        timeout_seconds=30,
        runner=run,
        nonce_factory=lambda: "abc123",
    )

    assert check.status == "verified"
    assert check.digest == DIGEST
    assert check.cleanup_status == "unverified"
    assert check.ok is False


def test_target_pull_probe_retries_when_delete_times_out() -> None:
    run, _calls = _target_probe_runner()
    delete_attempts = 0

    def timeout_delete(cmd, **kwargs):  # noqa: ANN001
        nonlocal delete_attempts
        if "delete" in cmd:
            delete_attempts += 1
            raise subprocess.TimeoutExpired(cmd, 30)
        return run(cmd, **kwargs)

    check = verify_kubernetes_image_pull(
        image=IMAGE,
        secret_names=("pull-secret",),
        namespace="target-namespace",
        context="target-context",
        timeout_seconds=30,
        runner=timeout_delete,
        nonce_factory=lambda: "abc123",
    )

    assert check.status == "verified"
    assert check.digest == DIGEST
    assert check.cleanup_status == "unverified"
    assert check.ok is False
    assert delete_attempts == 4


def test_target_pull_probe_bounds_create_exception_text() -> None:
    secret = "synthetic-target-probe-secret"

    def unavailable(*args, **kwargs):  # noqa: ANN002,ANN003
        raise OSError(f"Bearer {secret}")

    check = verify_kubernetes_image_pull(
        image=IMAGE,
        secret_names=("pull-secret",),
        namespace="target-namespace",
        context="target-context",
        timeout_seconds=30,
        runner=unavailable,
        nonce_factory=lambda: "abc123",
    )

    assert check == KubernetesPullCheck(
        status="create_rejected", cleanup_status="unverified"
    )
    assert secret not in repr(check)


def test_indeterminate_create_still_uses_uid_preconditioned_cleanup() -> None:
    run, _calls = _target_probe_runner()
    create_attempted = False

    def timeout_after_create(cmd, **kwargs):  # noqa: ANN001
        nonlocal create_attempted
        if "create" in cmd and not create_attempted:
            create_attempted = True
            raise subprocess.TimeoutExpired(cmd, 30, output="opaque")
        return run(cmd, **kwargs)

    check = verify_kubernetes_image_pull(
        image=IMAGE,
        secret_names=("pull-secret",),
        namespace="target-namespace",
        context="target-context",
        timeout_seconds=30,
        runner=timeout_after_create,
        nonce_factory=lambda: "abc123",
    )

    assert check.status == "create_rejected"
    assert check.cleanup_status == "verified"


def test_target_pull_probe_uses_sky_tasks_default_namespace() -> None:
    commands: list[list[str]] = []
    run, _calls = _target_probe_runner()

    def recording_runner(cmd, **kwargs):  # noqa: ANN001
        commands.append(cmd)
        if "create" in cmd:
            manifest = json.loads(kwargs["input"])
            assert manifest["metadata"]["namespace"] == "default"
        return run(cmd, **kwargs)

    check = verify_kubernetes_image_pull(
        image=IMAGE,
        secret_names=("pull-secret",),
        namespace="default",
        context="target-context",
        timeout_seconds=30,
        runner=recording_runner,
        nonce_factory=lambda: "abc123",
    )

    assert check.ok
    assert all(
        command[:5]
        == ["kubectl", "--context", "target-context", "--namespace", "default"]
        for command in commands
    )


def test_operator_selected_unbounded_probe_still_cleans_up_on_interrupt() -> None:
    run, calls = _target_probe_runner(
        image_id="",
        active_deadline_seconds=None,
    )

    with pytest.raises(KeyboardInterrupt):
        verify_kubernetes_image_pull(
            image=IMAGE,
            secret_names=("pull-secret",),
            namespace="target-namespace",
            context="target-context",
            timeout_seconds=0,
            runner=run,
            sleeper=lambda seconds: (_ for _ in ()).throw(KeyboardInterrupt()),
            nonce_factory=lambda: "abc123",
        )

    assert sum("delete" in command for command in calls) == 1


@pytest.mark.parametrize("interrupted_operation", ["identity-read", "delete"])
def test_cleanup_retries_owned_deletion_after_keyboard_interrupt(
    interrupted_operation: str,
) -> None:
    run, calls = _target_probe_runner()
    interrupted = False
    operation_attempts = 0

    def interrupt_cleanup_once(cmd, **kwargs):  # noqa: ANN001
        nonlocal interrupted, operation_attempts
        is_cleanup_read = "get" in cmd and "--ignore-not-found=true" in cmd
        selected = (
            is_cleanup_read
            if interrupted_operation == "identity-read"
            else "delete" in cmd
        )
        if selected:
            operation_attempts += 1
        if selected and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return run(cmd, **kwargs)

    with pytest.raises(KeyboardInterrupt):
        verify_kubernetes_image_pull(
            image=IMAGE,
            secret_names=("pull-secret",),
            namespace="target-namespace",
            context="target-context",
            timeout_seconds=30,
            runner=interrupt_cleanup_once,
            nonce_factory=lambda: "abc123",
        )

    assert interrupted
    assert operation_attempts >= 2
    assert sum("delete" in command for command in calls) == 1


def test_malformed_probe_labels_fail_closed_without_uncaught_exception() -> None:
    run, calls = _target_probe_runner()

    def malformed_labels(cmd, **kwargs):  # noqa: ANN001
        result = run(cmd, **kwargs)
        if "get" not in cmd:
            return result
        payload = json.loads(result.stdout)
        payload["metadata"]["labels"] = ["not", "a", "mapping"]
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps(payload), stderr=""
        )

    check = verify_kubernetes_image_pull(
        image=IMAGE,
        secret_names=("pull-secret",),
        namespace="target-namespace",
        context="target-context",
        timeout_seconds=30,
        runner=malformed_labels,
        nonce_factory=lambda: "abc123",
    )

    assert check.status == "identity_mismatch"
    assert check.cleanup_status == "identity_mismatch"
    assert not any("delete" in command for command in calls)


def test_pull_failure_is_preserved_when_cleanup_is_also_unverified() -> None:
    run, _calls = _target_probe_runner(
        delete_exit=1,
        image_id="",
        waiting_reason="ErrImagePull",
    )

    check = verify_kubernetes_image_pull(
        image=IMAGE,
        secret_names=("pull-secret",),
        namespace="target-namespace",
        context="target-context",
        timeout_seconds=30,
        runner=run,
        nonce_factory=lambda: "abc123",
    )

    assert check.status == "image_pull_failed"
    assert check.cleanup_status == "unverified"


def test_non_pull_container_failure_does_not_wait_for_timeout() -> None:
    run, _calls = _target_probe_runner(
        image_id="",
        waiting_reason="CreateContainerError",
        active_deadline_seconds=1800,
    )

    check = verify_kubernetes_image_pull(
        image=IMAGE,
        secret_names=("pull-secret",),
        namespace="target-namespace",
        context="target-context",
        timeout_seconds=1800,
        runner=run,
        sleeper=lambda seconds: pytest.fail("terminal reason must not sleep"),
        nonce_factory=lambda: "abc123",
    )

    assert check.status == "target_probe_failed"
    assert check.cleanup_status == "verified"


def test_opaque_cri_image_id_proves_pull_without_false_digest_comparison() -> None:
    run, _calls = _target_probe_runner(image_id=f"cri-o://{DIGEST}")

    check = verify_kubernetes_image_pull(
        image=IMAGE,
        secret_names=("pull-secret",),
        namespace="target-namespace",
        context="target-context",
        timeout_seconds=30,
        runner=run,
        nonce_factory=lambda: "abc123",
    )

    assert check.ok
    assert check.digest == ""
